#!/usr/bin/env python3
# validate_stage3_csv.py — Sanity-check des CSV (no header) pour SageMaker XGBoost (format v2)
from __future__ import annotations
import argparse, json, io, sys, math
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

# =============================
# 1) Seuils side-aware (pooled)
# =============================
SIDE_THRESHOLDS = {
    "_default": {"rmse_max": 0.15, "corr_min": 0.95},
    "obi_15":   {"rmse_max": 0.26, "corr_min": 0.94},
    "aggr_ratio_10s": {"rmse_max": 0.70, "corr_min": 0.50},
    "slope_bid_5":  {"rmse_max": 0.80, "corr_min": 0.70},
    "slope_ask_5":  {"rmse_max": 0.80, "corr_min": 0.70},
    "slope_bid_15": {"rmse_max": 0.80, "corr_min": 0.75},
    "slope_ask_15": {"rmse_max": 0.80, "corr_min": 0.75},
}

# Colonnes non-features dans les CSV v2
NON_FEATURE_COLS = {"Y", "side_num", "weight"}

# --- helper: side-projection attendu selon la feature
SIDE_NEUTRAL_05 = {"aggr_ratio_10s", "aggr_ratio_15s"}  # étends si besoin

def _expected_side_projection(base_name: str, base: np.ndarray, side: np.ndarray) -> np.ndarray:
    x = base.astype(np.float64)
    s = side.astype(np.float64)
    if base_name in SIDE_NEUTRAL_05:
        return s * (x - 0.5)      # neutre à 0.5
    else:
        return s * x              # neutre à 0.0
    
# =============================
# Helpers classes (online stats)
# =============================
def _detect_constant_flags(feat_stats: pd.DataFrame) -> list[str]:
    const = []
    for _, r in feat_stats.iterrows():
        name, std, minv, maxv = r["name"], r["std"], r["min"], r["max"]
        if name.endswith(("_gt_0p08","_gt_0p05","_gt_0p12")) or name.endswith("_isconstant"):
            if std < 1e-9 or minv == maxv:
                const.append(name)
    return const

class ColStats:
    __slots__ = ("n","mean","M2","nan_count","posinf","neginf","minv","maxv")
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
        self.nan_count = 0; self.posinf = 0; self.neginf = 0
        self.minv = math.inf; self.maxv = -math.inf
    def update(self, x: np.ndarray):
        self.nan_count += int(np.isnan(x).sum())
        self.posinf += int(np.isposinf(x).sum())
        self.neginf += int(np.isneginf(x).sum())
        x = x[np.isfinite(x)]
        if x.size == 0: return
        self.minv = float(min(self.minv, float(np.min(x))))
        self.maxv = float(max(self.maxv, float(np.max(x))))
        n2 = self.n + x.size
        xm = float(x.mean())
        delta = xm - self.mean
        self.mean += delta * (x.size / max(n2,1))
        self.M2 += float(((x - xm)**2).sum()) + (delta**2) * (self.n * x.size / max(n2,1))
        self.n = n2
    def finalize(self) -> Tuple[float,float]:
        if self.n <= 1: return (self.mean, 0.0)
        var = self.M2 / self.n
        return (self.mean, math.sqrt(var) if var > 0 else 0.0)

class PairStats:
    __slots__ = ("n","sse","sum_x","sum_y","sum_x2","sum_y2","sum_xy")
    def __init__(self):
        self.n = 0; self.sse = 0.0
        self.sum_x = 0.0; self.sum_y = 0.0
        self.sum_x2 = 0.0; self.sum_y2 = 0.0; self.sum_xy = 0.0
    def update(self, x: np.ndarray, y: np.ndarray):
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]
        if x.size == 0: return
        self.n += x.size
        self.sse += float(np.sum((y - x)**2))
        self.sum_x  += float(np.sum(x)); self.sum_y  += float(np.sum(y))
        self.sum_x2 += float(np.sum(x*x)); self.sum_y2 += float(np.sum(y*y)); self.sum_xy += float(np.sum(x*y))
    def finalize(self) -> Tuple[float,float]:
        if self.n == 0: return (float("nan"), float("nan"))
        rmse = math.sqrt(self.sse / self.n); n = float(self.n)
        num = self.sum_xy - (self.sum_x * self.sum_y) / n
        den_x = self.sum_x2 - (self.sum_x**2) / n
        den_y = self.sum_y2 - (self.sum_y**2) / n
        corr = float("nan") if den_x <= 0 or den_y <= 0 else num / math.sqrt(den_x * den_y)
        return (rmse, corr)

# =============================
# S3
# =============================
def _fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _strip_s3(uri: str) -> str:
    assert uri.startswith("s3://"), "Only s3:// supported"
    return uri[len("s3://"):]

def _list_csv_files(fs: pafs.S3FileSystem, dir_s3: str, max_files: int, exclude_weight: bool = True) -> List[str]:
    sel = pafs.FileSelector(_strip_s3(dir_s3), recursive=True)
    out = []
    for info in fs.get_file_info(sel):
        if not info.is_file:
            continue
        if not info.path.lower().endswith(".csv"):
            continue
        path = "s3://" + info.path
        if exclude_weight and ("/train_weight/" in path or "/validation_weight/" in path or "/test_weight/" in path):
            continue
        out.append(path)
        if len(out) >= max_files:
            break
    return sorted(out)

def _read_csv_chunks(fs: pafs.S3FileSystem, path_s3: str, chunksize: int):
    with fs.open_input_stream(_strip_s3(path_s3)) as f:
        bio = io.BytesIO(f.read())
    for chunk in pd.read_csv(bio, header=None, dtype=np.float64, chunksize=chunksize):
        yield chunk

def _maybe_list_dir(fs: pafs.S3FileSystem, dir_s3: str) -> Optional[list[str]]:
    try:
        sel = pafs.FileSelector(_strip_s3(dir_s3), recursive=False)
        files = [ "s3://" + info.path for info in fs.get_file_info(sel)
                  if info.is_file and info.path.lower().endswith(".csv") ]
        return sorted(files) if files else None
    except Exception:
        return None

# =============================
# Meta loader (v2)
# =============================
def _load_meta(fs, root: str):
    """
    columns.json (v2) :
      {
        "features": [...],
        "label": "Y",
        "side": "side_num"
      }
    """
    columns_json_path = f"{root.rstrip('/')}/_meta/columns.json"
    with fs.open_input_stream(_strip_s3(columns_json_path)) as f:
        meta = json.loads(f.read().decode("utf-8"))

    features = meta.get("features")
    label = meta.get("label", "Y")
    side = meta.get("side", "side_num")
    if not isinstance(features, list) or len(features) == 0:
        raise RuntimeError("columns.json invalide: 'features' doit être une liste non vide.")
    if label != "Y" or side != "side_num":
        raise RuntimeError("columns.json invalide: attendu label='Y' et side='side_num'.")

    col_names = ["Y", "side_num"] + features

    # (optionnel) train_pos_weight.json si présent
    tpw_path = f"{root.rstrip('/')}/_meta/train_pos_weight.json"
    pos_weight = None
    try:
        with fs.open_input_stream(_strip_s3(tpw_path)) as f:
            pj = json.loads(f.read().decode("utf-8"))
            pos_weight = pj.get("pos_weight", None)
    except Exception:
        pass

    return col_names, pos_weight, features

def _check_scaler_stats(fs, root: str, features: list[str]) -> int:
    path = f"{root.rstrip('/')}/_meta/scaler_stats.json"
    try:
        with fs.open_input_stream(_strip_s3(path)) as f:
            stats = json.loads(f.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ scaler_stats.json introuvable ou illisible: {e}")
        return 0

    # stats: liste de dicts {"tf": "...", "feature": "...", "q1":..., "q99":..., "median":..., "scale":...}
    df = pd.DataFrame(stats)
    if df.empty or not {"tf","feature"}.issubset(df.columns):
        print("⚠️ scaler_stats.json mal formé (pas de colonnes tf/feature).")
        return 0

    # TF présents dans les CSV (on peut les inférer plus tard dans _validate_one_split si besoin)
    # Ici, on accepte un check "global": chaque feature doit exister pour au moins un TF.
    missing_any = [f for f in features if f not in set(df["feature"].unique())]
    if missing_any:
        print(f"⛔ features sans stats de scaling: {missing_any}")
        return 2

    # Option (plus strict) : quand on scanne un split, on détecte les TF vus
    # et on vérifie qu'il existe une entrée (tf, feature) pour chacun.
    # On fera ce check *dans* _validate_one_split (voir patch d ci-dessous).
    print("[meta] scaler_stats.json: OK (présence globale par feature).")
    return 0

def _probe_align(fs, path):
    with fs.open_input_stream(_strip_s3(path)) as f:
        df = pd.read_csv(f)
    need = {"y","p_pred","symbol","tf","side_num"}
    if not need.issubset(df.columns):
        print("⚠️ predictions.csv: colonnes manquantes pour align (requis y,p_pred,symbol,tf,side_num)"); return
    from sklearn.metrics import roc_auc_score
    def best_lag(y, p, kmax=10):
        from sklearn.metrics import roc_auc_score
        def safe_auc(a, b):
            if a.size < 2 or b.size < 2: 
                return float("nan")
            u = np.unique(a)
            if u.size < 2:
                return float("nan")
            return roc_auc_score(a, b)

        base = safe_auc(y, p)
        best = (0, base if np.isfinite(base) else -1.0)
        n = min(len(y), len(p))
        for k in range(1, kmax+1):
            auc_f = safe_auc(y[k:n], p[:n-k])
            if np.isfinite(auc_f) and auc_f > best[1]:
                best = (k, auc_f)
            auc_b = safe_auc(y[:n-k], p[k:n])
            if np.isfinite(auc_b) and auc_b > best[1]:
                best = (-k, auc_b)
        return best
    rows=[]
    for (sym, tf, side), sd in df.groupby(["symbol","tf","side_num"]):
        y = sd["y"].astype(int).to_numpy()
        p = sd["p_pred"].astype(float).to_numpy()
        lag, auc = best_lag(y,p)
        rows.append((sym,tf,int(sd["side_num"].iloc[0]),lag,auc))
    rpt = pd.DataFrame(rows, columns=["symbol","tf","side_num","best_lag","auc"])
    print("\n[align] best_lag par (symbol,tf,side_num):")
    print(rpt.sort_values("auc", ascending=False).head(40).to_string(index=False))
    off = rpt[rpt["best_lag"].abs()>=2]
    if not off.empty:
        print("\n⚠️ groupes avec best_lag |≥2| :")
        print(off.sort_values("best_lag").to_string(index=False))

# =============================
# CLI
# =============================
def parse_args():
    ap = argparse.ArgumentParser(description="Validate Stage3 XGBoost CSV shards (no header) — format v2.")
    ap.add_argument("--root", required=True, help="s3://…/data/stage3-xgb/vX-…")
    ap.add_argument("--split", default="all", choices=["all","train","validation","test"], help="Par défaut: all")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--chunksize", type=int, default=500_000)

    # Seuils pooled (doivent matcher la normalisation globale de stage3)
    ap.add_argument("--mean-tol", type=float, default=0.12)
    ap.add_argument("--std-low", type=float, default=0.75)
    ap.add_argument("--std-high", type=float, default=1.30)

    # Validation poids
    ap.add_argument("--check-weights", action="store_true", default=True,
                    help="Vérifie les shards de poids parallèles (*/_weight).")

    # Fait échouer le run s’il y a des anomalies fortes sur TRAIN
    ap.add_argument("--fail-on-strong", action="store_true")

    # Meta parquet: résumé TF/side/weights si présent
    ap.add_argument("--check-meta-parquet", action="store_true",
                    help="Si _meta/splits_parquet existe, résume Y/poids par tf/side.")
    
    ap.add_argument("--predictions-csv", help="s3://…/predictions.csv (colonnes y,p_pred,symbol,tf,side_num)")

    return ap.parse_args()

# =============================
# Split runner
# =============================
def _validate_one_split(fs, root: str, split: str, col_names: List[str], args, features: List[str]):
    print("\n" + "="*80)
    print(f"🔎 Split: {split}")
    print("="*80)

    name_to_idx = {c:i for i,c in enumerate(col_names)}

    # fichiers CSV
    split_dir = f"{root.rstrip('/')}/{split}"
    files = _list_csv_files(fs, split_dir, args.max_files, exclude_weight=True)
    if not files:
        print("⛔ Aucun fichier CSV trouvé sous", split_dir)
        return 2
    print(f"[scan] {len(files)} files under {split_dir}")

    # paires base/_side
    pair_indices: List[Tuple[int,int]] = []
    for base, ib in name_to_idx.items():
        if ib < 2 or base.endswith("_isnan") or base.endswith("_side") or base in NON_FEATURE_COLS:
            continue
        side_name = f"{base}_side"
        if side_name in name_to_idx:
            pair_indices.append((ib, name_to_idx[side_name]))
    if pair_indices:
        preview = [(col_names[ib], col_names[iside]) for (ib,iside) in pair_indices[:12]]
        print(f"[side] {len(pair_indices)} paires détectées (extraits) :", preview)
    else:
        print("[side] aucune paire base/_side détectée")

    # Accumulateurs
    col_stats: Dict[int, ColStats] = {}
    label_counts: Dict[int, int] = {}
    side_anomalies = 0
    total_rows = 0
    ncols_seen = None
    side_pairs: Dict[Tuple[int,int], PairStats] = {p: PairStats() for p in pair_indices}

    # Scan
    for path in files:
        # Sécurité : ignorer tout shard sous *_weight/
        if ("/train_weight/" in path) or ("/validation_weight/" in path) or ("/test_weight/" in path):
            continue
        for chunk in _read_csv_chunks(fs, path, chunksize=args.chunksize):
            if ncols_seen is None:
                ncols_seen = chunk.shape[1]
                if ncols_seen != len(col_names):
                    print(f"⛔ nb colonnes CSV ({ncols_seen}) ≠ nb colonnes meta ({len(col_names)}).")
                    return 2
                print(f"[shape] detected {ncols_seen} columns (matches meta)")

            if chunk.shape[1] < 2:
                print("⛔ CSV shard with <2 columns (Y, side_num manquants):", path)
                return 2

            total_rows += len(chunk)

            # Label
            y = pd.to_numeric(chunk.iloc[:,0], errors="coerce").fillna(-1).astype("int64").to_numpy()
            vals, cnts = np.unique(y, return_counts=True)
            for v, c in zip(vals, cnts):
                label_counts[int(v)] = label_counts.get(int(v), 0) + int(c)

            # side_num
            side_nm = pd.to_numeric(chunk.iloc[:,1], errors="coerce").to_numpy()
            side_anomalies += int(np.isnan(side_nm).sum())
            side_anomalies += int((~np.isin(side_nm, [-1, 1])).sum())

            # Stats globales par colonne
            for j in range(chunk.shape[1]):
                if j not in col_stats: col_stats[j] = ColStats()
                col_stats[j].update(pd.to_numeric(chunk.iloc[:,j], errors="coerce").to_numpy(dtype=np.float64))

            # Side-aware: base_side ≈ side_num * base
            if pair_indices:
                s = side_nm.astype(np.float64)
                for (ib, iside), acc in side_pairs.items():
                    base_name = col_names[ib]
                    base  = pd.to_numeric(chunk.iloc[:, ib],   errors="coerce").to_numpy(dtype=np.float64)
                    sidev = pd.to_numeric(chunk.iloc[:, iside], errors="coerce").to_numpy(dtype=np.float64)
                    exp   = _expected_side_projection(base_name, base, s)
                    acc.update(exp, sidev)

    # Reporting pooled
    print(f"\n[rows] total scanned: {total_rows}")
    if total_rows == 0:
        print("⛔ Aucun enregistrement lu.")
        return 2

    print("\nℹ️ Label balance:")
    total_lbl = sum(label_counts.values())
    for k in sorted(label_counts.keys()):
        r = label_counts[k] / max(total_lbl,1)
        print(f"label={k}: count={label_counts[k]} ratio={r:.4f}")

    if side_anomalies > 0:
        print(f"\n⚠️ anomalies side_num détectées (NaN ou valeurs ≠ -1/1): ~{side_anomalies}")
    
    if split == "train":
        pos = int(label_counts.get(1, 0))
        neg = int(label_counts.get(0, 0))
        if pos > 0:
            spw = float(neg) / float(pos)
            print(f"[meta] scale_pos_weight (train) ≈ {spw:.4f}")
            try:
                path = f"{root.rstrip('/')}/_meta/train_pos_weight.json"
                with fs.open_output_stream(_strip_s3(path)) as out:
                    out.write(json.dumps({
                        "pos_weight": spw,
                        "pos_count": pos,
                        "neg_count": neg
                    }, indent=2).encode("utf-8"))
                print(f"[meta] Écrit {path}")
            except Exception as e:
                print(f"[meta] Écriture pos_weight skip ({e})")
        else:
            print("[meta] Impossible de calculer scale_pos_weight: aucun positif en TRAIN.")

    # stats features (indices >=2), en excluant weight et *_isnan
    def _ignore_for_zscore(name: str) -> bool:
        return name.endswith("_isnan") or name in NON_FEATURE_COLS

    ncols_seen = len(col_names)  # sécurisé
    feat_idx = [i for i in range(2, ncols_seen) if not _ignore_for_zscore(col_names[i])]
    mask_idx = [i for i in range(2, ncols_seen) if col_names[i].endswith("_isnan")]

    def _finalize(indices: List[int]) -> pd.DataFrame:
        rows = []
        for j in indices:
            cs = col_stats.get(j)
            if cs is None:
                rows.append((j, col_names[j], float("nan"), float("nan"), 0, 0, 0, 0, float("nan"), float("nan")))
                continue
            mu, sd = cs.finalize()
            rows.append((j, col_names[j], mu, sd, cs.n, cs.nan_count, cs.posinf, cs.neginf, cs.minv, cs.maxv))
        return pd.DataFrame(rows, columns=["col","name","mean","std","n","nan","posinf","neginf","min","max"])

    feat_stats = _finalize(feat_idx)

    const_flags = _detect_constant_flags(feat_stats)
    if const_flags:
        print("\n[flags] Colonnes constantes à ignorer pour l’entraînement :", const_flags)
        # (facultatif) Sauvegarde sous _meta/constant_flags.json
        try:
            meta_dir = f"{root.rstrip('/')}/_meta"
            path = f"{meta_dir}/constant_flags.json"
            with fs.open_output_stream(_strip_s3(path)) as out:
                out.write(json.dumps({"drop_cols": const_flags}, indent=2).encode("utf-8"))
            print(f"[flags] Écrit {path}")
        except Exception as e:
            print(f"[flags] Écriture skip ({e})")

    print("\nGLOBAL feature stats (%s): z-scored → |mean|<=%.2f, std∈[%.2f, %.2f]."
      % (split, args.mean_tol, args.std_low, args.std_high))
    print(feat_stats.head(80).to_string(index=False))

    # ===== Auto-détection du mode de scaling (Z-MODE vs WIDE-MODE) =====
    means = feat_stats["mean"].abs().to_numpy()
    stds  = feat_stats["std"].to_numpy()
    n     = max(len(stds), 1)

    # heuristique : si ≥60% des features ont std dans [0.6,1.5] ET |mean|≤0.5 → on considère “z-scored pooled”
    ok_mask = (stds >= 0.6) & (stds <= 1.5) & (means <= 0.5)
    ratio_ok = float(ok_mask.sum()) / n
    is_z_mode = (ratio_ok >= 0.60)

    print(f"\n[scaling] auto-detect → {'Z-MODE (pooled)' if is_z_mode else 'WIDE-MODE (no strict z)'} "
        f"(ok_ratio={ratio_ok:.2f})")

    # en Z-MODE, on applique les seuils serrés; sinon, reporting uniquement (pas de “bad” bloquant)
    if is_z_mode:
        bad = feat_stats[(feat_stats["mean"].abs() > args.mean_tol) |
                        (feat_stats["std"] < args.std_low) |
                        (feat_stats["std"] > args.std_high)]
        if not bad.empty and split == "train":
            print("\n⚠️ anomalies (TRAIN, Z-MODE):")
            print(bad.to_string(index=False))
    else:
        bad = pd.DataFrame()  # pas d’anomalies bloquantes en WIDE-MODE

    qconst = feat_stats[feat_stats["std"] < 1e-6]
    if not qconst.empty:
        print("\n⚠️ quasi-constantes (std < 1e-6):")
        print(qconst.to_string(index=False))

    # Report *_isnan
    if mask_idx:
        mask_stats = _finalize(mask_idx)
        print("\nMasques *_isnan (mean = taux de NaN imputés):")
        print(mask_stats[["name","mean","n"]].to_string(index=False))

    # Side-aware pooled
    if pair_indices:
        rows = []
        for (ib, iside), acc in side_pairs.items():
            rmse, corr = acc.finalize()
            rows.append((col_names[ib], col_names[iside], acc.n, rmse, corr))
        df_pairs = pd.DataFrame(rows, columns=["base","side","n","rmse","corr"])
        print("\nCohérence side-aware (side ≈ side_num * base):")
        print(df_pairs.to_string(index=False))

        def _is_bad(row):
            thr = SIDE_THRESHOLDS.get(row["base"], SIDE_THRESHOLDS["_default"])
            return (row["rmse"] > thr["rmse_max"]) or (row["corr"] < thr["corr_min"])

        bad_pairs = df_pairs[df_pairs.apply(_is_bad, axis=1)]
        if not bad_pairs.empty:
            print("\n⚠️ anomalies side-aware (seuils spécifiques par feature):")
            print(bad_pairs.to_string(index=False))

    # ===== Weights validation =====
    if args.check_weights:
        w_dir = f"{root.rstrip('/')}/{split}_weight"
        w_files = _maybe_list_dir(fs, w_dir)
        if not w_files:
            print(f"\n[weights] {split}: aucun shard trouvé sous {w_dir} (OK si tu ne les as pas écrits).")
        else:
            print(f"\n[weights] {split}: {len(w_files)} fichiers détectés sous {w_dir}")
            w_stats = ColStats(); total_w_rows = 0
            for p in w_files[:args.max_files]:
                for chunk in _read_csv_chunks(fs, p, chunksize=args.chunksize):
                    if chunk.shape[1] != 1:
                        print(f"⚠️ {p} : attendu 1 colonne de poids, trouvé {chunk.shape[1]}")
                    w = pd.to_numeric(chunk.iloc[:,0], errors="coerce").to_numpy(dtype=np.float64)
                    w_stats.update(w); total_w_rows += len(w)
            w_mean, w_std = w_stats.finalize()
            print(f"[weights] rows={total_w_rows} | mean={w_mean:.4f} std={w_std:.4f} "
                  f"min={w_stats.minv:.4f} max={w_stats.maxv:.4f} "
                  f"nan={w_stats.nan_count} +inf={w_stats.posinf} -inf={w_stats.neginf}")

            # tolérances simples
            if split == "train":
                if not (0.95 <= w_mean <= 1.05):
                    print("⚠️ TRAIN: mean(weight) attendu ≈1 (±5%).")
                if not (w_stats.minv >= 0.0):
                    print("⚠️ TRAIN: weight doit être ≥ 0.")
            else:
                if not (0.80 <= w_mean <= 1.20):
                    print("ℹ️ OOS: mean(weight) atypique (attendu ~[0.8,1.2]).")
                if not (w_stats.minv >= 0.0):
                    print("⚠️ OOS: weight doit être ≥ 0.")

            # parité lignes (approx. sur l’échantillon scanné)
            if total_rows and total_w_rows:
                delta = abs(total_rows - total_w_rows)
                if delta == 0:
                    print(f"[weights] parité lignes OK (features={total_rows}, weights={total_w_rows}).")
                else:
                    print(f"⚠️ mismatch lignes features({total_rows}) vs weights({total_w_rows}) "
                          f"(max-files={args.max_files}).")

    # ===== Meta Parquet résumé (facultatif) =====
    if args.check_meta_parquet:
        meta_dir = f"{root.rstrip('/')}/_meta/splits_parquet/{split}"
        try:
            sel = pafs.FileSelector(_strip_s3(meta_dir), recursive=True)
            infos = [i for i in fs.get_file_info(sel) if i.is_file and i.path.endswith(".parquet")]
            if not infos:
                print(f"\n[meta] {split}: pas de fichiers dans {meta_dir} — skip")
            else:
                # lecture rapide (concat all; meta est petit vs CSV)
                tables = []
                for inf in infos:
                    with fs.open_input_file(inf.path) as f:  # seekable
                        tables.append(pq.read_table(f))
                meta_df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)

                # en fin de _validate_one_split, si args.check_meta_parquet est activé et meta parquet dispo:
                # après avoir lu meta_df
                if args.check_meta_parquet and "tf" in meta_df.columns:
                    tf_seen = sorted(set(meta_df["tf"].astype(str).unique()))
                    # charger scaler_stats.json une seule fois
                    try:
                        with fs.open_input_stream(_strip_s3(f"{root.rstrip('/')}/_meta/scaler_stats.json")) as f:
                            ss = pd.DataFrame(json.loads(f.read().decode("utf-8")))
                        have = set((ss["tf"].astype(str) + "|" + ss["feature"]).tolist())
                        missing_pairs = []
                        for tfv in tf_seen:
                            for feat in features:
                                key = f"{tfv}|{feat}"
                                if key not in have:
                                    missing_pairs.append((tfv, feat))
                        if missing_pairs:
                            print(f"\n⛔ scaler_stats coverage manquante pour {split} : {len(missing_pairs)} (TF, feature) pairs")
                            print(pd.DataFrame(missing_pairs, columns=["tf","feature"]).head(40).to_string(index=False))
                            return 2
                        else:
                            print(f"\n[meta] scaler_stats coverage OK pour TF vus dans {split}.")
                    except Exception as e:
                        print(f"[meta] Impossible de vérifier scaler_stats coverage: {e}")


                # Attendu: row_id, symbol, tf, t, Y, side_num, weight
                cols_ok = {"Y","side_num","weight"}
                if not cols_ok.issubset(set(meta_df.columns)):
                    print(f"[meta] colonnes attendues manquantes dans {meta_dir} — trouvé: {list(meta_df.columns)}")
                else:
                    print("\n[meta] Résumé par tf (label/poids):")
                    if "tf" in meta_df.columns:
                        g = meta_df.groupby("tf").agg(
                            n=("Y","size"),
                            y_pos=("Y","sum"),
                            w_mean=("weight","mean"),
                            w_std=("weight","std")
                        ).reset_index()
                        g["y_rate"] = g["y_pos"] / g["n"].clip(lower=1)
                        print(g.to_string(index=False))
                    else:
                        print("[meta] colonne 'tf' absente — skip résumé TF.")
                    if "side_num" in meta_df.columns:
                        gs = meta_df.groupby("side_num").agg(n=("Y","size"), y_pos=("Y","sum")).reset_index()
                        gs["y_rate"] = gs["y_pos"] / gs["n"].clip(lower=1)
                        print("\n[meta] Répartition par side_num:")
                        print(gs.to_string(index=False))
        except Exception as e:
            print(f"[meta] Erreur lecture meta parquet: {e}")

    # si on est en Z-MODE, 'bad' contient les anomalies strictes; en WIDE-MODE il est vide
    strict_bad = bad
    if args.fail_on_strong and split == "train" and (not strict_bad.empty):
        print("\n⛔ strong anomalies detected, failing run:")
        print(strict_bad.to_string(index=False))
        return 2

    if split == "train":
        if is_z_mode:
            print("\n✅ TRAIN: Z-MODE — z-scored ≈ mean 0 / std 1 (pooled TF).")
        else:
            print("\nℹ️ TRAIN: WIDE-MODE — pas de z-scoring pooled strict, checks informatifs uniquement.")
    elif split == "validation":
        print("\nℹ️ VALIDATION: de légers décalages mean/std peuvent arriver (semi-OOS).")
    else:
        print("\nℹ️ TEST: des décalages de mean/std sont normaux (OOS).")

    return 0

# =============================
# Main
# =============================
def main():
    args = parse_args()
    fs = _fs(args.aws_region)

    # meta
    try:
        col_names, pos_weight, features = _load_meta(fs, args.root)
    except Exception as e:
        print(f"⛔ Problème lecture meta sous {args.root}:\n{e}")
        sys.exit(2)

    print("[meta] columns.json OK")
    if pos_weight is not None:
        print(f"[meta] pos_weight (train) ≈ {pos_weight:.4f}")

    FORBID = {"THRESH_BPS","pnl_net_max_bps","pnl_net_min_bps"}
    forbid_found = sorted(FORBID & set(features))
    if forbid_found:
        print(f"⛔ features interdits détectés dans columns.json: {forbid_found}")
        sys.exit(2)
    
    rc = _check_scaler_stats(fs, args.root, features)
    if rc != 0:
        sys.exit(rc)

    splits = ["train","validation","test"] if args.split == "all" else [args.split]
    
    if args.predictions_csv:
        _probe_align(fs, args.predictions_csv)

    rc_sum = 0
    for sp in splits:
        rc_sum += _validate_one_split(fs, args.root, sp, col_names, args, features)

    sys.exit(0 if rc_sum == 0 else 2)

if __name__ == "__main__":
    main()