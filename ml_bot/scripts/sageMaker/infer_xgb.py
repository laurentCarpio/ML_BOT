# -*- coding: utf-8 -*-
"""
Inference validator for a frozen XGBoost model release.

Exemples :

  # Politique proba simple (seuil global ou manifest)
  python ml_bot/scripts/sageMaker/infer_xgb.py \
    --manifest-s3 s3://.../manifest.json \
    --batch s3://.../validation \
    --out s3://.../infer_runs/.../val_test

  # Politique proba + thresholds par poche (tf|side_num)
  python ml_bot/scripts/sageMaker/infer_xgb.py \
    --manifest-s3 s3://.../manifest.json \
    --batch s3://.../test \
    --thresholds-s3 s3://.../thresholds_xxx.json \
    --out s3://.../infer_runs/.../test_v1

  # Politique EV (p * RR) avec RR tiré du META Stage3 (rr_nominal)
  python ml_bot/scripts/sageMaker/infer_xgb.py \
    --manifest-s3 s3://.../manifest.json \
    --batch s3://tradebot-config-tokyo/data/stage3/v51-short/test \
    --policy ev \
    --ev-min 5.0 \
    --out s3://.../infer_runs/.../test_ev
"""

import os
import io
import json
import tarfile
import argparse
import glob
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
import json
import fsspec  # si tu l'utilises déjà pour S3 / s3fs

# s3fs (obligatoire pour S3)
try:
    import s3fs
    _HAS_S3FS = True
except Exception:
    _HAS_S3FS = False

TF_DEFAULT = "('15m',)"  # fallback si aucune info de tf trouvée

# Matplotlib est optionnel
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ----------------------------
# Utils S3 / fichiers
# ----------------------------

def _load_json_anywhere(path: str):
    """Charge un JSON depuis S3 (s3://…) ou local."""
    if path.startswith("s3://"):
        fs, _, paths = fsspec.get_fs_token_paths(path)
        with fs.open(paths[0], "rb") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _fs():
    if not _HAS_S3FS:
        raise RuntimeError("s3fs is required for S3 paths. Install s3fs.")
    return s3fs.S3FileSystem(anon=False)


def _read_text(path: str) -> str:
    if _is_s3(path):
        fs = _fs()
        with fs.open(path, "rb") as f:
            return f.read().decode("utf-8")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(_read_text(path))


def _write_bytes(path: str, data: bytes):
    if _is_s3(path):
        fs = _fs()
        with fs.open(path, "wb") as f:
            f.write(data)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def _write_json(path: str, obj: Dict[str, Any]):
    _write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))


def _list_csv_parts(prefix: str) -> List[str]:
    """Liste les part-*.csv sous prefix (S3 ou local)."""
    if prefix.endswith(".csv"):
        return [prefix]

    pattern = os.path.join(prefix, "part-*.csv")
    if _is_s3(prefix):
        fs = _fs()
        no_scheme = prefix.replace("s3://", "")
        candidates = fs.glob(no_scheme + "/part-*.csv")
        return [f"s3://{p}" for p in sorted(candidates)]
    else:
        return sorted(glob.glob(pattern))


# ----------------------------
# Chargement batch (stage3-xgb)
# ----------------------------

def _load_csv_matrix(
    prefix: str,
    label_col: str = "Y",
    drop_cols: Tuple[str, ...] = ("side_num",)
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """
    Charge un batch stage3 (train/val/test) :

    - CSVs sans header
    - colonnes supposées: [Y, side_num, f0..fN]
    """
    parts = _list_csv_parts(prefix)
    if not parts:
        raise RuntimeError(
            f"No CSV found under {prefix} (expected part-*.csv or a CSV file)"
        )

    dfs = []
    for p in parts:
        if _is_s3(p):
            fs = _fs()
            with fs.open(p, "rb") as f:
                df = pd.read_csv(f, header=None)
        else:
            df = pd.read_csv(p, header=None)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    ncol = df.shape[1]
    if ncol < 3:
        raise RuntimeError(f"CSV shape too small: {df.shape}")

    cols = ["Y", "side_num"] + [f"f{i}" for i in range(ncol - 2)]
    df.columns = cols

    y = df["Y"].astype(np.float32).values if "Y" in df.columns else None
    drop = set([c for c in drop_cols if c in df.columns])
    feat_df = df.drop(columns=list(drop.union({"Y"})), errors="ignore")
    X = feat_df.astype(np.float32).values
    return X, y, df


# ----------------------------
# Normalisation (option manifeste)
# ----------------------------

def _load_scaler_stats(uri: str) -> Any:
    return _read_json(uri)


def _pick_tf_key(batch_df: pd.DataFrame) -> str:
    for col in ("tf", "timeframe", "TF"):
        if col in batch_df.columns:
            vals = batch_df[col].dropna().astype(str).unique().tolist()
            if len(vals) == 1:
                return vals[0]
            if len(vals) > 1:
                # si plusieurs tf dans le batch, on prend le mode
                return pd.Series(vals).mode().iloc[0]
    return TF_DEFAULT


def _coerce_stats_by_feature(
    stats_list: list,
    tf_key: str,
    features: List[str]
) -> Dict[str, Dict[str, float]]:
    rows = [r for r in stats_list if str(r.get("tf")) == str(tf_key)]
    if not rows:
        rows = [r for r in stats_list if str(r.get("tf")) == TF_DEFAULT]
    by_feat = {str(r["feature"]): r for r in rows if "feature" in r}
    out: Dict[str, Dict[str, float]] = {}
    for f in features:
        r = by_feat.get(f, {})
        q1 = float(r.get("q1", 0.0))
        q99 = float(r.get("q99", 1.0))
        med = float(r.get("median", 0.0))
        sc = float(r.get("scale", 1.0)) or 1.0
        if q99 <= q1:
            q99 = q1 + 1e-9
        out[f] = {"q1": q1, "q99": q99, "median": med, "scale": sc}
    return out


def _normalize_features_robust(
    X_df: pd.DataFrame,
    features: List[str],
    stats_list: list
) -> pd.DataFrame:
    tf_key = _pick_tf_key(X_df)
    print(f"[infer] normalization using tf={tf_key}")
    per = _coerce_stats_by_feature(stats_list, tf_key, features)
    X = X_df[features].copy()
    for f in features:
        st = per[f]
        x = X[f].astype(float)
        x = x.clip(lower=st["q1"], upper=st["q99"])
        X[f] = (x - st["median"]) / (st["scale"] if st["scale"] != 0 else 1.0)
    return X


# ----------------------------
# META Stage3 helpers (EV & per-pocket)
# ----------------------------

def _infer_meta_prefix(manifest: Dict[str, Any], batch_path: str) -> Tuple[str, str, str]:
    """
    Déduit data_root / split_name / meta_prefix à partir du manifest et du chemin batch.
    data_root doit pointer sur la racine Stage3 (ex: s3://.../data/stage3/v51-short).
    """
    data_root = manifest.get("data_root")
    if not isinstance(data_root, str) or not data_root:
        raise RuntimeError("Manifest must provide 'data_root' for EV/per-pocket thresholds.")

    split_name = "test"
    if batch_path.startswith(data_root):
        rel = batch_path[len(data_root):].lstrip("/")
        split_name = rel.split("/", 1)[0] or "test"

    meta_prefix = f"{data_root.rstrip('/')}/_meta/splits_parquet/{split_name}"
    return data_root, split_name, meta_prefix


def _load_meta_frame(meta_prefix: str) -> pd.DataFrame:
    """Charge les META parquet Stage3 (row_id,symbol,tf,t,Y,side_num,weight, pnl_net_*,thresh_bps,rr_nominal...)."""
    import pyarrow.parquet as pq
    fs = _fs()
    pattern = meta_prefix.replace("s3://", "") + "/*.parquet"
    paths = fs.glob(pattern)
    tables = []
    for p in paths:
        with fs.open(f"s3://{p}", "rb") as f:
            tables.append(pq.read_table(f))
    if not tables:
        raise RuntimeError(f"No parquet meta found under {meta_prefix}")
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)


# ----------------------------
# Modèle XGBoost
# ----------------------------

def _load_booster_from_model_tar(model_tar_uri: str) -> xgb.Booster:
    """Charge model.tar.gz et retourne un Booster."""
    if _is_s3(model_tar_uri):
        fs = _fs()
        with fs.open(model_tar_uri, "rb") as f:
            raw = f.read()
    else:
        with open(model_tar_uri, "rb") as f:
            raw = f.read()

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        member = None
        for m in tar.getmembers():
            if m.name.endswith(("model.json", "xgboost-model", "model.bin")):
                member = m
                break
        if member is None:
            raise RuntimeError("No model.json/xgboost-model/model.bin inside model.tar.gz")
        model_bytes = tar.extractfile(member).read()

    booster = xgb.Booster()
    try:
        booster.load_model(io.BytesIO(model_bytes))  # xgboost>=2.0
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp.flush()
            booster.load_model(tmp.name)
    return booster


# ----------------------------
# Métriques légères (AUPRC)
# ----------------------------

def _auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_pred))
    except Exception:
        # fallback approximation
        order = np.argsort(-y_pred)
        y = y_true[order]
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(np.sum(y == 1), 1)
        return float(np.trapz(prec, rec))


# ----------------------------
# Main
# ----------------------------

def main():
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-s3", required=True,
                    help="s3://.../manifest.json or local path")
    ap.add_argument("--batch", required=True,
                    help="S3/local dir with part-*.csv or a single CSV file "
                         "(columns: Y, side_num, features...)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override global proba threshold "
                         "(default uses manifest.inference.decision_threshold)")
    ap.add_argument("--thresholds-s3", default=None,
                    help="Optional S3/local JSON with per-pocket thresholds "
                         "(keys 'tf|side_num' -> threshold) – only for policy='proba'")
    ap.add_argument("--policy", choices=["proba", "ev"], default="proba",
                    help="Decision policy: 'proba' (default) or 'ev' (p * rr_nominal from META)")
    ap.add_argument("--ev-min", type=float, default=None,
                    help="Minimum EV required when policy='ev' "
                         "(default=0.0 si non précisé)")
    ap.add_argument("--out", default=None,
                    help="S3 or local prefix for outputs (writes report.json and proba_hist.png)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional row cap for quick tests")

    ap.add_argument("--ev-thresholds-json", default=None,
                    help="Optionnel : JSON de seuils EV par poche (symbol|tf), "
                        "produit par make_ev_thresholds_per_pocket.py. "
                        "Si présent et policy='ev', remplace --ev-min."
                )
    
    args = ap.parse_args()

    y = None
    probas = None
    meta_df: Optional[pd.DataFrame] = None

    t0 = time.time()
    print(f"[infer] manifest={args.manifest_s3}")
    try:
        manifest = _read_json(args.manifest_s3)

        # --- manifest fields ---
        model_uri = manifest["model_uri"]
        infer_cfg = manifest.get("inference", {}) or {}
        manifest_thr = float(infer_cfg.get("decision_threshold", 0.5))

        # global threshold (proba) : CLI > manifest
        cli_thr = args.threshold
        use_thr = float(cli_thr if cli_thr is not None else manifest_thr)
        threshold_source = "cli" if cli_thr is not None else "manifest"

        print(f"[infer] model_uri={model_uri}")
        print(f"[infer] decision_threshold={use_thr:.6f} (source={threshold_source})")
        print(f"[infer] policy={args.policy}")

        # --- data contract ---
        data_contract = manifest.get("data_contract", {}) or {}
        label_col = data_contract.get("label_col", "Y")
        drop_cols = data_contract.get("drop_cols", ["side_num"])
        scaler_uri = data_contract.get("scaler_stats_uri")
        normalize = bool(data_contract.get("normalize_at_infer", False))

        # --- charge batch brut ---
        X_raw, y, df_raw = _load_csv_matrix(
            args.batch,
            label_col=label_col,
            drop_cols=tuple(drop_cols)
        )
        if args.limit and args.limit > 0:
            X_raw = X_raw[:args.limit]
            if y is not None:
                y = y[:args.limit]
            df_raw = df_raw.iloc[:args.limit].copy()
        print(f"[infer] batch loaded: X_raw={X_raw.shape}, y={'present' if y is not None else 'absent'}")

        # --- ordre des features : depuis le manifeste (source entraînement) ---
        def _load_feat_order(manifest_obj, fallback_cols_count: int) -> List[str]:
            dc = manifest_obj.get("data_contract", {}) or {}
            feats = dc.get("features")
            if isinstance(feats, list) and feats:
                return list(feats)
            uri = dc.get("features_list_uri")
            if uri:
                try:
                    obj = _read_json(uri)
                    if isinstance(obj, dict) and "features" in obj:
                        return list(obj["features"])
                    if isinstance(obj, list):
                        return list(obj)
                except Exception as e:
                    print(f"[warn] Could not read features_list_uri ({uri}): {e}")
            return [f"f{i}" for i in range(fallback_cols_count)]

        feat_names = _load_feat_order(manifest, fallback_cols_count=X_raw.shape[1])
        if len(feat_names) != X_raw.shape[1]:
            raise RuntimeError(
                f"[features-order] Mismatch: feat_names({len(feat_names)}) "
                f"vs X_raw width({X_raw.shape[1]})"
            )

        X_df = pd.DataFrame(X_raw, columns=feat_names)
        if set(X_df.columns) != set(feat_names):
            raise RuntimeError(
                f"[features] Set mismatch. Batch={set(X_df.columns)} "
                f"vs Manifest={set(feat_names)}"
            )
        if list(X_df.columns) != list(feat_names):
            print("[warn] Column order differs; reordering to training order.")
            X_df = X_df.loc[:, feat_names]

        # --- normalisation optionnelle ---
        def _normalize_by_tf(df, features, stats_list, tf_col="tf"):
            if tf_col not in df.columns:
                return _normalize_features_robust(df, features=features, stats_list=stats_list)
            parts = []
            for _, grp in df.groupby(tf_col, sort=False):
                g = _normalize_features_robust(grp.copy(), features=features, stats_list=stats_list)
                parts.append(g)
            out = pd.concat(parts, axis=0).loc[df.index]
            return out

        if scaler_uri and normalize:
            print(f"[infer] normalize_at_infer=True -> loading scaler stats: {scaler_uri}")
            scaler_stats_list = _load_scaler_stats(scaler_uri)
            base_cols = [c for c in df_raw.columns if c not in feat_names]
            Xn = _normalize_by_tf(
                df=pd.concat([df_raw[base_cols], X_df], axis=1),
                features=feat_names,
                stats_list=scaler_stats_list,
                tf_col="tf",
            )
            X_norm = Xn[feat_names]
        else:
            msg = "[infer] normalize_at_infer is False -> using raw features"
            if scaler_uri and not normalize:
                msg += " (scaler_stats_uri present but disabled)"
            print(msg)
            X_norm = X_df

        X_norm = X_norm.astype(np.float32)
        nan_frac = float(np.isnan(X_norm.values).mean())
        if nan_frac > 0:
            print(f"[warn] NaN in X_norm: {nan_frac:.6f} frac")

        # --- modèle ---
        print("[infer] loading model booster from S3..." if _is_s3(model_uri) else "[infer] loading model booster...")
        booster = _load_booster_from_model_tar(model_uri)

        # IMPORTANT: construire le DMatrix AVANT predict
        try:
            dmat = xgb.DMatrix(X_norm.values, feature_names=list(feat_names))
        except Exception:
            dmat = xgb.DMatrix(X_norm.values)

        # --- proba P(Y=1) ---
        print("[infer] predicting probabilities (P(Y=1))...")
        probas = booster.predict(dmat)
        probas = np.asarray(probas, dtype=np.float32)

        q = np.quantile(probas, [0, 0.5, 0.9, 0.95, 0.99, 1.0])
        print(
            f"[debug] p_pred quantiles: "
            f"min={q[0]:.4f} p50={q[1]:.4f} p90={q[2]:.4f} "
            f"p95={q[3]:.4f} p99={q[4]:.4f} max={q[5]:.4f}"
        )
        print(f"[debug] NaN in X_norm: {nan_frac:.6f} frac")

        # --- construction du vecteur de seuils proba (global ou par poche) ---
        thr_vec = np.full_like(probas, use_thr, dtype=np.float32)

        # Per-pocket thresholds (uniquement pour policy='proba')
        if args.thresholds_s3 and args.policy == "proba":
            try:
                thr_cfg = _read_json(args.thresholds_s3)
                thr_map = thr_cfg.get("thresholds", {})

                data_root, split_name, meta_prefix = _infer_meta_prefix(manifest, args.batch)
                print(f"[thr] Loading per-pocket thresholds from {args.thresholds_s3}")
                print(f"[thr] data_root={data_root} | split={split_name} | meta_prefix={meta_prefix}")

                # Charge META (tf, side_num, etc.)
                meta_df = _load_meta_frame(meta_prefix)

                if len(meta_df) != len(probas):
                    raise RuntimeError(
                        f"meta rows ({len(meta_df)}) != batch rows ({len(probas)}). "
                        "Cannot align per-pocket thresholds."
                    )

                if "tf" not in meta_df.columns or "side_num" not in meta_df.columns:
                    raise RuntimeError("meta parquet must contain 'tf' and 'side_num' columns")

                keys = (meta_df["tf"].astype(str) + "|" +
                        meta_df["side_num"].astype(int).astype(str)).tolist()

                thr_list = []
                missing = set()
                for k in keys:
                    if k in thr_map:
                        thr_list.append(float(thr_map[k]))
                    else:
                        thr_list.append(use_thr)
                        missing.add(k)

                thr_vec = np.asarray(thr_list, dtype=np.float32)
                threshold_source = "per_pocket"

                print(
                    f"[thr] per-pocket thresholds applied. "
                    f"Distinct pockets in batch: {len(set(keys))}, "
                    f"missing pockets (fallback to global thr): {len(missing)}"
                )
                if missing:
                    print(f"[thr] missing pockets (sample): {list(sorted(missing))[:10]}")

            except Exception as e:
                print(
                    f"[thr] WARN: could not apply per-pocket thresholds ({e}); "
                    "falling back to global threshold."
                )

        # --- Décision : proba vs EV ---
        ev_vec = None
        ev_min = None

        if args.policy == "proba":
            yhat = (probas >= thr_vec).astype(np.int32)
            print(
                f"[debug] count>=thr: {(probas >= thr_vec).sum()} / {probas.size} "
                f"(global_thr={use_thr:.6f}, mode={threshold_source})"
            )

        else:  # policy == "ev"
            # On a besoin de META pour rr_nominal (Stage3 v51+)
            if meta_df is None:
                data_root, split_name, meta_prefix = _infer_meta_prefix(manifest, args.batch)
                print(f"[ev] Loading META from {meta_prefix} ...")
                meta_df = _load_meta_frame(meta_prefix)

            if len(meta_df) != len(probas):
                raise RuntimeError(
                    f"[ev] Row count mismatch: META={len(meta_df)} vs batch={len(probas)}. "
                    "You must ensure META was generated from the same Stage3 split."
                )

            if "rr_nominal" not in meta_df.columns:
                raise RuntimeError(
                    "[ev] META must contain 'rr_nominal' column for EV policy. "
                    "Regenerate Stage3 (v51+) with rr_nominal."
                )

            rr_vec = meta_df["rr_nominal"].astype(np.float32).to_numpy()

            # ⚠️ Assure-toi que la formule EV ici est la même que dans
            # make_ev_thresholds_from_predictions.py
            # Si là-bas tu fais : EV = p_pred * rr_nominal - (1 - p_pred),
            # adapte ici en conséquence.
            ev_vec = probas * rr_vec

            # --- Seuils EV : global + éventuel JSON par poche ---
            # fallback global (au cas où le JSON ne se charge pas)
            ev_min_global = 0.0 if args.ev_min is None else float(args.ev_min)
            pockets = {}

            if args.ev_thresholds_json:
                try:
                    thr_obj = _load_json_anywhere(args.ev_thresholds_json)
                    # Structure attendue :
                    # {
                    #   "global": {"ev_min": 0.072406, ...},
                    #   "pockets": {
                    #       "WLDUSDT|4h": {"ev_min": 0.072406, ...},
                    #       ...
                    #   }
                    # }
                    g = thr_obj.get("global", {})
                    if "ev_min" in g:
                        ev_min_global = float(g["ev_min"])
                    pockets = thr_obj.get("pockets", {}) or {}
                    print(
                        f"[ev] Using per-pocket EV thresholds from {args.ev_thresholds_json} "
                        f"(ev_min_global={ev_min_global:.6f})"
                    )
                    if args.ev_min is not None:
                        print(
                            f"[ev] NOTE: --ev-min={args.ev_min} est ignoré "
                            f"car --ev-thresholds-json est fourni."
                        )
                except Exception as e:
                    print(
                        f"[ev] WARN: could not load ev_thresholds_json ({e}); "
                        f"falling back to global EV min={ev_min_global:.6f}"
                    )
                    pockets = {}

            else:
                print(f"[ev] Using global EV threshold only: ev_min={ev_min_global:.6f}")

            # --- Appliquer un ev_min par ligne (symbol|tf) ---
            if "symbol" not in meta_df.columns or "tf" not in meta_df.columns:
                raise RuntimeError(
                    "[ev] META must contain 'symbol' and 'tf' columns for per-pocket EV thresholds."
                )

            keys = (meta_df["symbol"].astype(str) + "|" +
                    meta_df["tf"].astype(str)).tolist()

            ev_min_used = np.empty_like(ev_vec, dtype=np.float32)
            missing = set()
            for i, k in enumerate(keys):
                pocket = pockets.get(k)
                if pocket and "ev_min" in pocket:
                    ev_min_used[i] = float(pocket["ev_min"])
                else:
                    ev_min_used[i] = ev_min_global
                    if pockets:
                        missing.add(k)

            if pockets:
                print(
                    f"[ev] per-pocket EV thresholds applied. "
                    f"Distinct pockets in batch: {len(set(keys))}, "
                    f"missing pockets (fallback to global ev_min): {len(missing)}"
                )
                if missing:
                    print(f"[ev] missing pockets (sample): {list(sorted(missing))[:10]}")

            # Décision finale
            yhat = (ev_vec >= ev_min_used).astype(np.int32)

            n_trig = int(yhat.sum())
            print(
                f"[debug] EV policy: using rr_nominal from META, "
                f"ev_min_global={ev_min_global:.6f}, "
                f"count(EV>=ev_min_used)={n_trig} / {len(ev_vec)}"
            )

            # Pour que le report sache quel seuil a été utilisé
            ev_min = ev_min_global

        # --- CSV y,p_pred(+EV) vers S3 (debug) ---
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        s3_path = f"s3://tradebot-config-tokyo/tmp/pred_debug/{ts}/predictions.csv"

        # Base d'identifiants : si META dispo et aligné, on l'utilise (plus riche)
        cols_to_dump: Dict[str, Any] = {}
        if meta_df is not None and len(meta_df) == len(probas):
            for c in [
                "row_id", "symbol", "tf", "t", "Y",
                "side_num", "weight",
                "pnl_net_max_bps", "pnl_net_min_bps",
                "thresh_bps", "rr_nominal"
            ]:
                if c in meta_df.columns:
                    cols_to_dump[c] = meta_df[c].values
        else:
            # Fallback: colonnes éventuelles du batch brut
            for c in ["symbol", "tf", "ts", "side_num", "row_id", "row_idx"]:
                if c in df_raw.columns:
                    cols_to_dump[c] = df_raw[c].values

        extra_cols = {}
        if ev_vec is not None:
            extra_cols["EV"] = ev_vec
        elif meta_df is not None and "rr_nominal" in meta_df.columns:
            # Même sans policy EV, on peut logguer rr_nominal pour debug
            extra_cols["rr_nominal"] = meta_df["rr_nominal"].astype(float).to_numpy()

        df_out = pd.DataFrame({
            **cols_to_dump,
            "y": y if y is not None else np.full(len(probas), np.nan),
            "p_pred": np.asarray(probas),
            **extra_cols,
        })

        if _is_s3(s3_path):
            with _fs().open(s3_path, "w") as f:
                df_out.to_csv(f, index=False)
        else:
            Path(s3_path).parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(s3_path, index=False)
        print(f"[ok] Écrit debug CSV : {s3_path}")

        # --- métriques + report ---
        report: Dict[str, Any] = {
            "model_uri": model_uri,
            "decision_threshold_manifest": float(manifest_thr),
            "decision_threshold_used": float(use_thr),
            "threshold_source": threshold_source,
            "policy": args.policy,
            "n_rows": int(len(probas)),
            "timestamp": ts,
            "pred_debug_csv": s3_path,
        }

        if args.policy == "ev":
            report["ev_min_global"] = ev_min
            report["ev_rr_source"] = "META.rr_nominal"
            report["ev_thresholds_json"] = args.ev_thresholds_json

        if y is not None:
            y_int = y.astype(np.int32)
            ap = _auprc(y_int, probas)
            tp = int(np.sum((y_int == 1) & (yhat == 1)))
            fp = int(np.sum((y_int == 0) & (yhat == 1)))
            tn = int(np.sum((y_int == 0) & (yhat == 0)))
            fn = int(np.sum((y_int == 1) & (yhat == 0)))
            precision = float(tp / max(tp + fp, 1))
            recall = float(tp / max(tp + fn, 1))
            f1 = float((2 * precision * recall) / max(precision + recall, 1e-12))
            trigger_rate = float(np.mean(yhat))
            pos_rate = float(np.mean(y_int))

            try:
                from sklearn.metrics import roc_auc_score
                report["roc_auc"] = float(roc_auc_score(y_int, probas))
            except Exception:
                report["roc_auc"] = None

            report.update({
                "AUPRC_batch": ap,
                "precision_at_threshold": precision,
                "recall_at_threshold": recall,
                "F1_at_threshold": f1,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "trigger_rate": trigger_rate,
                "pos_rate": pos_rate,
            })

            print(
                f"[infer] AUPRC={ap:.6f} | "
                f"P@thr={precision:.3f} R@thr={recall:.3f} F1@thr={f1:.3f} | "
                f"trigger_rate={trigger_rate:.4f} pos_rate={pos_rate:.4f}"
            )
        else:
            trigger_rate = float(np.mean(yhat))
            report.update({
                "AUPRC_batch": None,
                "precision_at_threshold": None,
                "recall_at_threshold": None,
                "F1_at_threshold": None,
                "TP": None, "FP": None, "TN": None, "FN": None,
                "trigger_rate": trigger_rate,
                "pos_rate": None,
            })
            print(f"[infer] labels not provided; trigger_rate={trigger_rate:.4f}")

        # --- artéfacts out ---
        out_prefix = args.out
        if out_prefix:
            report_uri = out_prefix.rstrip("/") + "/report.json"
            _write_json(report_uri, report)
            print(f"[infer] report written: {report_uri}")

            if _HAS_MPL:
                try:
                    fig = plt.figure(figsize=(6, 4), dpi=120)
                    ax = fig.add_subplot(111)
                    ax.hist(probas, bins=50, alpha=0.85)
                    ax.set_title("Prediction probability histogram (P(y=1))")
                    ax.set_xlabel("p(y=1)")
                    ax.set_ylabel("count")
                    if args.policy == "proba":
                        ax.axvline(use_thr, linestyle="--")
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    buf.seek(0)
                    png_uri = out_prefix.rstrip("/") + "/proba_hist.png"
                    _write_bytes(png_uri, buf.getvalue())
                    print(f"[infer] histogram written: {png_uri}")
                except Exception as e:
                    print(f"[warn] could not write histogram: {e}")
            else:
                print("[infer] matplotlib not available; skipping histogram.")
        else:
            print("[infer] no --out provided; skipping artifact write.")

        print(f"[infer] done in {time.time() - t0:.1f}s")

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)