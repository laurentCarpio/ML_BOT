# -*- coding: utf-8 -*-
"""
Inference validator for a frozen XGBoost model release.

Usage examples:

  python ml_bot/scripts/sageMaker/infer_xgb.py \
    --manifest-s3 s3://tradebot-config-tokyo/models/xgb/releases/long-20251120-140313-v45/manifest.json \
    --batch s3://tradebot-config-tokyo/data/stage3/v45-longonly/validation \
    --out s3://tradebot-config-tokyo/models/xgb/infer_runs/long-20251120-140313-v45/val_2025-11-20

   python ml_bot/scripts/sageMaker/infer_xgb.py \
    --manifest-s3 s3://tradebot-config-tokyo/models/xgb/releases/short-20251120-140340-v45/manifest.json \
    --batch s3://tradebot-config-tokyo/data/stage3/v45-shortonly/validation \
    --out s3://tradebot-config-tokyo/models/xgb/infer_runs/short-20251120-140340-v45/val_2025-11-20 
    

    
  python infer_xgb.py \
    --manifest-s3 s3://.../manifest.json \
    --batch ./local_batch_dir_with_csv_parts \
    --threshold 0.6728 \
    --out ./_infer_out/go-20251107-v1_local
"""
import os, io, json, tarfile, argparse, glob, sys, time, tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xgboost as xgb

# s3fs (obligatoire pour S3)
try:
    import s3fs
    _HAS_S3FS = True
except Exception:
    _HAS_S3FS = False

TF_DEFAULT = "('15m',)"  # ajuste si ton batch validation est sur un autre tf

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

def _load_csv_matrix(prefix: str, label_col="Y", drop_cols=("side_num",)) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    parts = _list_csv_parts(prefix)
    if not parts:
        raise RuntimeError(f"No CSV found under {prefix} (expected part-*.csv or a CSV file)")
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
                return pd.Series(vals).mode().iloc[0]
    return TF_DEFAULT

def _coerce_stats_by_feature(stats_list: list, tf_key: str, features: List[str]) -> Dict[str, Dict[str, float]]:
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

def _normalize_features_robust(X_df: pd.DataFrame, features: List[str], stats_list: list) -> pd.DataFrame:
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
# Modèle XGBoost
# ----------------------------

def _load_booster_from_model_tar(model_tar_uri: str) -> xgb.Booster:
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
        booster.load_model(io.BytesIO(model_bytes))
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp.flush()
            booster.load_model(tmp.name)
    return booster


# ----------------------------
# Métriques légères (AUC-ROC & AUPRC)
# ----------------------------

def _try_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None  # si sklearn absent

def _auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_pred))
    except Exception:
        order = np.argsort(-y_pred)
        y = y_true[order]
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(np.sum(y == 1), 1)
        return float(np.trapz(prec, rec))


# ----------------------------
# Smart proba extractor (toujours P(y=1))
# ----------------------------

def _smart_predict_proba_class1(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Renvoie P(y=1) quoi qu'il arrive.
    - Essaie d'abord predict(..., output_margin=True) + sigmoïde
    - Sinon fallback sur predict(..., output_margin=False) si déjà dans [0,1]
    - Si sortie hors [0,1], applique sigmoïde
    """
    # Essai margins -> plus robuste (indépendant de l’objective)
    try:
        dmat = xgb.DMatrix(X)
        margins = booster.predict(dmat, output_margin=True)
        p = 1.0 / (1.0 + np.exp(-margins))
        if np.all(np.isfinite(p)):
            return p.astype(np.float32, copy=False)
    except Exception:
        pass

    # Fallback: proba directe
    dmat = xgb.DMatrix(X)
    p = booster.predict(dmat, output_margin=False)
    p = np.asarray(p)
    if (p.min() < 0) or (p.max() > 1) or ~np.isfinite(p).all():
        p = 1.0 / (1.0 + np.exp(-p))
    return p.astype(np.float32, copy=False)


# ----------------------------
# Main
# ----------------------------

def main():
    import traceback
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-s3", required=True, help="s3://.../manifest.json or local path")
    ap.add_argument("--batch", required=True,
                    help="S3/local dir with part-*.csv or a single CSV file (columns: Y, side_num, features...)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override decision threshold (default uses manifest.inference.decision_threshold)")
    ap.add_argument("--out", default=None,
                    help="S3 or local prefix for outputs (will write report.json and proba_hist.png)")
    ap.add_argument("--limit", type=int, default=0, help="Optional row cap for quick tests")
    args = ap.parse_args()

    # pour éviter tout UnboundLocal en cas d'exception précoce
    y = None
    probas = None

    t0 = time.time()
    print(f"[infer] manifest={args.manifest_s3}")
    try:
        manifest = _read_json(args.manifest_s3)

        # --- manifest fields ---
        model_uri    = manifest["model_uri"]
        infer_cfg    = manifest.get("inference", {}) or {}
        allow_flip   = bool(infer_cfg.get("autoflip_allowed", True))
        manifest_thr = float(infer_cfg.get("decision_threshold", 0.5))
        cli_thr      = args.threshold  # None ou float
        use_thr      = float(cli_thr if cli_thr is not None else manifest_thr)

        print(f"[infer] model_uri={model_uri}")
        print(f"[infer] decision_threshold={use_thr:.6f} (manifest={manifest_thr:.6f})")

        # --- data contract ---
        data_contract = manifest.get("data_contract", {}) or {}
        label_col   = data_contract.get("label_col", "Y")
        drop_cols   = data_contract.get("drop_cols", ["side_num"])
        scaler_uri  = data_contract.get("scaler_stats_uri")
        normalize   = bool(data_contract.get("normalize_at_infer", False))

        # --- charge batch brut ---
        X_raw, y, df_raw = _load_csv_matrix(args.batch, label_col=label_col, drop_cols=tuple(drop_cols))
        if args.limit and args.limit > 0:
            X_raw = X_raw[:args.limit]
            if y is not None:
                y = y[:args.limit]
            df_raw = df_raw.iloc[:args.limit].copy()
        print(f"[infer] batch loaded: X_raw={X_raw.shape}, y={'present' if y is not None else 'absent'}")

        # --- ordre des features : depuis le manifeste (source entraînement) ---
        def _load_feat_order(manifest, fallback_cols_count):
            data_contract = manifest.get("data_contract", {}) or {}
            feats = data_contract.get("features")
            if isinstance(feats, list) and feats:
                return list(feats)
            uri = data_contract.get("features_list_uri")
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
            raise RuntimeError(f"[features-order] Mismatch: feat_names({len(feat_names)}) vs X_raw width({X_raw.shape[1]})")

        X_df = pd.DataFrame(X_raw, columns=feat_names)
        if set(X_df.columns) != set(feat_names):
            raise RuntimeError(f"[features] Set mismatch. Batch={set(X_df.columns)} vs Manifest={set(feat_names)}")
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
        print("[infer] loading model booster...")
        booster = _load_booster_from_model_tar(model_uri)

        # IMPORTANT: construire le DMatrix AVANT predict
        try:
            dmat = xgb.DMatrix(X_norm.values, feature_names=list(feat_names))
        except Exception:
            dmat = xgb.DMatrix(X_norm.values)

        # --- proba P(y=1) ---
        print("[infer] predicting probabilities (raw model output)...")
        probas = booster.predict(dmat)

        # --- orientation & threshold remap ---
        proba_semantics = (infer_cfg.get("proba_semantics", "") or "").strip().lower()
        thr_is_for_py1 = ("p(y=1)" in proba_semantics) or ("p = p(y=1)" in proba_semantics)
        invert_output = bool(infer_cfg.get("invert_output", False))

        print(f"[infer] semantics: proba_semantics='{proba_semantics}', allow_flip={allow_flip}, "
            f"threshold_source={'cli' if cli_thr is not None else 'manifest'}")

        inverted = False

        # 1) Inversion déterministe depuis le manifest
        if invert_output:
            print("[xform] Deterministic invert_output=true -> using 1 - raw probabilities")
            probas = 1.0 - probas
            inverted = True

        # 2) Auto-flip (seulement si pas déjà inversé, flip autorisé, et labels présents)
        if (not invert_output) and allow_flip and (y is not None):
            try:
                from sklearn.metrics import roc_auc_score
                y_int = y.astype(int)
                auc_raw = roc_auc_score(y_int, probas)
                auc_inv = roc_auc_score(y_int, 1.0 - probas)
                if auc_inv > auc_raw + 1e-6:
                    print(f"[xform] Detected better orientation with 1 - p_pred "
                        f"(AUC {auc_raw:.6f} -> {auc_inv:.6f}). Inverting probabilities.")
                    probas = 1.0 - probas
                    inverted = True
                else:
                    print("[xform] Orientation looks fine; no inversion.")
            except Exception as e:
                print(f"[warn] auto-flip check failed: {e}")
        else:
            if invert_output:
                print("[xform] Auto-flip skipped (deterministic invert already applied).")
            elif not allow_flip:
                print("[xform] Auto-flip disabled by manifest.")
            else:
                print("[xform] No labels present; auto-flip skipped.")

        # 3) Remap du seuil si on a inversé, SEULEMENT si le seuil vient du manifest et
        #    que le manifest n’indique pas explicitement P(y=1)
        if inverted and cli_thr is None:
            if thr_is_for_py1:
                print("[xform] flipped probs, but manifest threshold is defined for P(y=1); leaving threshold unchanged.")
            else:
                use_thr = 1.0 - use_thr
                print(f"[xform] remapped manifest threshold -> {use_thr:.6f} (no explicit P(y=1) semantics)")
        
        q = np.quantile(probas, [0, 0.5, 0.9, 0.95, 0.99, 1.0])
        print(f"[debug] p_pred quantiles: min={q[0]:.4f} p50={q[1]:.4f} p90={q[2]:.4f} p95={q[3]:.4f} p99={q[4]:.4f} max={q[5]:.4f}")
        print(f"[debug] NaN in X_norm: {nan_frac:.6f} frac")
        print(f"[debug] count>=thr: {(probas >= use_thr).sum()} / {probas.size} (thr={use_thr:.6f})")

        # --- CSV y,p_pred vers S3 (debug) ---
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        s3_path = f"s3://tradebot-config-tokyo/tmp/pred_debug/{ts}/predictions.csv"

        # Inclure des colonnes ID si elles existent dans le batch brut
        id_cols = [c for c in ["symbol", "tf", "ts", "side_num", "row_id", "row_idx"] if c in df_raw.columns]
        cols_to_dump = {c: df_raw[c].values for c in id_cols}

        df_out = pd.DataFrame({
            **cols_to_dump,
            "y": y if y is not None else np.full(len(probas), np.nan),
            "p_pred": np.asarray(probas),
        })
        with _fs().open(s3_path, "w") as f:
            df_out.to_csv(f, index=False)
        print(f"[ok] Écrit sur S3 : {s3_path}")

        # --- métriques + report ---
        yhat = (probas >= use_thr).astype(np.int32)
        report: Dict[str, Any] = {
            "model_uri": model_uri,
            "decision_threshold_manifest": float(manifest_thr),
            "decision_threshold_used": float(use_thr),
            "threshold_source": ("cli" if cli_thr is not None else ("manifest_adjusted" if inverted else "manifest")),
            "autoflip_allowed": bool(allow_flip),
            "autoflip_applied": bool(inverted),
            "n_rows": int(len(probas)),
            "timestamp": ts,
            "pred_debug_csv": s3_path,
        }

        if y is not None:
            y_int = y.astype(np.int32)
            ap = _auprc(y_int, probas)
            tp = int(np.sum((y_int == 1) & (yhat == 1)))
            fp = int(np.sum((y_int == 0) & (yhat == 1)))
            tn = int(np.sum((y_int == 0) & (yhat == 0)))
            fn = int(np.sum((y_int == 1) & (yhat == 0)))
            precision = float(tp / max(tp + fp, 1))
            recall    = float(tp / max(tp + fn, 1))
            f1        = float((2 * precision * recall) / max(precision + recall, 1e-12))
            trigger_rate = float(np.mean(yhat))
            pos_rate     = float(np.mean(y_int))
            # AUC(s) optionnels
            try:
                from sklearn.metrics import roc_auc_score
                report["roc_auc_as_is"] = float(auc_raw) if auc_raw is not None else float(roc_auc_score(y_int, probas if not inverted else 1.0 - probas))
                report["roc_auc_final"] = float(roc_auc_score(y_int, probas))                
            except Exception:
                report["roc_auc_final"] = None
                report["roc_auc_as_is"] = None

            report.update({
                "decision_threshold_manifest": float(manifest_thr),
                "decision_threshold_used": float(use_thr),
                "AUPRC_batch": ap,
                "precision_at_threshold": precision,
                "recall_at_threshold": recall,
                "F1_at_threshold": f1,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "trigger_rate": trigger_rate,
                "pos_rate": pos_rate,
                "proba_semantics": proba_semantics,
                "invert_output_flag": bool(infer_cfg.get("invert_output", False)),
                "autoflip_allowed": bool(allow_flip),
                "autoflip_applied": bool(inverted),
                "threshold_source": ("cli" if cli_thr is not None else ("manifest_adjusted" if inverted and not thr_is_for_py1 else "manifest")),
            })
            
            print(f"[infer] AUPRC={ap:.6f} | P@thr={precision:.3f} R@thr={recall:.3f} F1@thr={f1:.3f} | "
                  f"trigger_rate={trigger_rate:.4f} pos_rate={pos_rate:.4f}")
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
                    ax.axvline(use_thr, linestyle="--")
                    ax.set_title("Prediction probability histogram (P(y=1))")
                    ax.set_xlabel("p(y=1)")
                    ax.set_ylabel("count")
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

        print(f"[infer] done in {time.time()-t0:.1f}s")

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