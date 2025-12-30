#!/usr/bin/env python3
# validate_stage3_go.py — Validation des CSV Stage3_GO (no header)
#
# Stage3_GO format:
#   - X shards: [Y] + features (no side_num)
#   - splits: train / val / test
#   - weights: train_weight / val_weight / test_weight
# Meta:
#   - _meta/columns.json (features + label)
#   - _meta/scaler_stats.json (robust params per tf/feature)
#   - _meta/train_class_balance.json (pos/neg + pos_weight_raw)
#
from __future__ import annotations

import argparse, json, io, sys, math
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.fs as pafs

# =============================
# Z-score: listes de contrôle
# =============================

# Colonnes qu'on ne check PAS en z-score strict (si tu en ajoutes plus tard)
Z_IGNORE_EXACT = {"microprice_bias"}

# Si une feature a une variance trop faible, le check Z-score strict n’a pas de sens.
LOW_VAR_STD_FLOOR = 0.05   # seuil sur std global (après scaling Stage3)
LOW_VAR_FEATURES = {
    "microprice_bias",
}

def _is_mask_col(name: str) -> bool:
    return name.endswith("_isnan") or name.endswith("_mask") or name.endswith("_flag")

# Features pour lesquelles on accepte des stats Z plus "relax"
HEAVY_TAIL_RELAXED = {
    "ret_stdev_1s_10s_bps",
    "quote_churn_10s",
    "spread_bps_entry",
    "wall_opp_share_5",
    "wall_opp_share_15",
    "bb_width",
    "adx",
    "atr_percentile",
    "atr_bps",
    "bt_dom_3s",
    "bt_dom_5s",
    "bt_dom_10s",
    "lf_bb_width_pct",

    # Ajouts: slopes (drift pooled fréquent) → RELAX
    "slope_bid_5",
    "slope_ask_5",
    "slope_bid_15",
    "slope_ask_15",

    # Optionnel (vu dans tes logs OOS): OBI peut bouger un peu → RELAX
    "obi_5",
    "obi_15",
}

# =============================
# Helpers classes (online stats)
# =============================

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
        if x.size == 0:
            return
        self.minv = float(min(self.minv, float(np.min(x))))
        self.maxv = float(max(self.maxv, float(np.max(x))))

        n2 = self.n + x.size
        xm = float(x.mean())
        delta = xm - self.mean
        self.mean += delta * (x.size / max(n2, 1))
        self.M2 += float(((x - xm) ** 2).sum()) + (delta**2) * (self.n * x.size / max(n2, 1))
        self.n = n2

    def finalize(self) -> Tuple[float,float]:
        if self.n <= 1:
            return (self.mean, 0.0)
        var = self.M2 / self.n
        return (self.mean, math.sqrt(var) if var > 0 else 0.0)

# =============================
# S3 helpers
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
        if exclude_weight and (
            "/train_weight/" in path or "/val_weight/" in path or "/test_weight/" in path
            or "/train_ids/" in path or "/val_ids/" in path or "/test_ids/" in path
        ):
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
        files = ["s3://" + info.path for info in fs.get_file_info(sel)
                 if info.is_file and info.path.lower().endswith(".csv")]
        return sorted(files) if files else None
    except Exception:
        return None

# =============================
# Meta loader (GO)
# =============================

def _load_meta(fs, root: str):
    """
    columns.json Stage3_GO:
      {
        "task": "go",
        "label": "is_tradeable",
        "features": [...],
        ...
      }
    """
    columns_json_path = f"{root.rstrip('/')}/_meta/columns.json"
    with fs.open_input_stream(_strip_s3(columns_json_path)) as f:
        meta = json.loads(f.read().decode("utf-8"))

    features = meta.get("features")
    label = meta.get("label", "Y")

    if not isinstance(features, list) or len(features) == 0:
        raise RuntimeError("columns.json invalide: 'features' doit être une liste non vide.")

    # ✅ on accepte le nouveau label
    if label not in ("Y", "is_tradeable"):
        raise RuntimeError(f"columns.json invalide: label inattendu '{label}' (attendu 'Y' ou 'is_tradeable').")

    col_names = [label] + features

    # train_class_balance.json (optionnel)
    bal_path = f"{root.rstrip('/')}/_meta/train_class_balance.json"
    balance = None
    try:
        with fs.open_input_stream(_strip_s3(bal_path)) as f:
            balance = json.loads(f.read().decode("utf-8"))
    except Exception:
        pass

    return col_names, features, balance, label

def _check_scaler_stats(fs, root: str, features: list[str]) -> int:
    path = f"{root.rstrip('/')}/_meta/scaler_stats.json"
    try:
        with fs.open_input_stream(_strip_s3(path)) as f:
            stats = json.loads(f.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ scaler_stats.json introuvable ou illisible: {e}")
        return 0

    df = pd.DataFrame(stats)
    if df.empty or not {"tf","feature"}.issubset(df.columns):
        print("⚠️ scaler_stats.json mal formé (pas de colonnes tf/feature).")
        return 0

    # Ici Stage3_GO n'embarque pas tf dans les CSV, donc on vérifie la présence globale des features.
    missing_any = [f for f in features if f not in set(df["feature"].unique())]
    if missing_any:
        print(f"⛔ features sans stats de scaling: {missing_any}")
        return 2

    print("[meta] scaler_stats.json: OK (présence globale par feature).")
    return 0

# =============================
# CLI
# =============================

def parse_args():
    ap = argparse.ArgumentParser("Validate Stage3_GO CSV shards (no header) — aligned with Stage3_GO format.")
    ap.add_argument("--root", default="s3://tradebot-config-tokyo/data/stage3/go")
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--chunksize", type=int, default=500_000)

    # Seuils pooled (si le scaling donne un mode ~Z)
    ap.add_argument("--mean-tol", type=float, default=0.12)
    ap.add_argument("--std-low", type=float, default=0.75)
    ap.add_argument("--std-high", type=float, default=1.30)

    ap.add_argument("--check-weights", action="store_true", default=True)
    ap.add_argument("--fail-on-strong", action="store_true",
                    help="Échec si anomalies STRICT sur TRAIN en Z-MODE.")
    return ap.parse_args()

# =============================
# Split runner
# =============================

def _validate_one_split(fs, root: str, split: str, col_names: List[str], args, features: List[str]):
    print("\n" + "="*80)
    print(f"🔎 Split: {split}")
    print("="*80)

    label_name = col_names[0]

    split_dir = f"{root.rstrip('/')}/{split}"
    files = _list_csv_files(fs, split_dir, args.max_files, exclude_weight=True)
    if not files:
        print("⛔ Aucun fichier CSV trouvé sous", split_dir)
        return 2
    print(f"[scan] {len(files)} files under {split_dir}")

    total_rows = 0
    label_counts: Dict[int, int] = {}
    col_stats: Dict[int, ColStats] = {}
    ncols_seen = None

    # Scan
    for path in files:
        for chunk in _read_csv_chunks(fs, path, chunksize=args.chunksize):
            if chunk.shape[0] == 0:
                continue

            if ncols_seen is None:
                ncols_seen = chunk.shape[1]
                if ncols_seen != len(col_names):
                    print(f"⛔ nb colonnes CSV ({ncols_seen}) ≠ nb colonnes meta ({len(col_names)}).")
                    return 2
                print(f"[shape] detected {ncols_seen} columns (matches meta)")

            total_rows += len(chunk)

            # Label
            y = pd.to_numeric(chunk.iloc[:,0], errors="coerce").fillna(-1).astype("int64").to_numpy()
            vals, cnts = np.unique(y, return_counts=True)
            for v, c in zip(vals, cnts):
                label_counts[int(v)] = label_counts.get(int(v), 0) + int(c)

            # Stats globales par colonne
            for j in range(chunk.shape[1]):
                if j not in col_stats:
                    col_stats[j] = ColStats()
                col_stats[j].update(
                    pd.to_numeric(chunk.iloc[:,j], errors="coerce").to_numpy(dtype=np.float64)
                )

    print(f"\n[rows] total scanned: {total_rows:,}")
    if total_rows == 0:
        print("⛔ Aucun enregistrement lu.")
        return 2

    print(f"\nℹ️ Label balance ({label_name}):")
    total_lbl = sum(label_counts.values())
    for k in sorted(label_counts.keys()):
        r = label_counts[k] / max(total_lbl, 1)
        print(f"label={k}: count={label_counts[k]:,} ratio={r:.4f}")

    # train class balance
    if split == "train":
        pos = int(label_counts.get(1, 0))
        neg = int(label_counts.get(0, 0))
        spw = (float(neg) / float(pos)) if pos > 0 else float("nan")
        print(f"[train] pos={pos:,} neg={neg:,} scale_pos_weight≈{spw:.6g}")

    def _ignore_for_zscore(name: str) -> bool:
        if name in Z_IGNORE_EXACT:
            return True
        if _is_mask_col(name):
            return True
        return False

    ncols_seen = len(col_names)
    feat_idx = [i for i in range(1, ncols_seen) if not _ignore_for_zscore(col_names[i])]

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

    print("\nGLOBAL feature stats (%s): pooled checks (auto Z-MODE detect)." % split)
    print(feat_stats.head(120).to_string(index=False))

    # ===== Auto-détection du mode de scaling (Z-MODE vs WIDE-MODE) =====
    means = feat_stats["mean"].abs().to_numpy()
    stds  = feat_stats["std"].to_numpy()
    n     = max(len(stds), 1)

    ok_mask = (stds >= 0.6) & (stds <= 1.5) & (means <= 0.5)
    ratio_ok = float(ok_mask.sum()) / n
    is_z_mode = (ratio_ok >= 0.60)

    print(f"\n[scaling] auto-detect → {'Z-MODE (pooled)' if is_z_mode else 'WIDE-MODE'} "
          f"(ok_ratio={ratio_ok:.2f})")

    strict_bad = pd.DataFrame()

    if is_z_mode:
        strict_rows  = []
        relaxed_rows = []

        for _, row in feat_stats.iterrows():
            col  = str(row["name"])
            mean = float(row["mean"])
            std  = float(row["std"])

            if _ignore_for_zscore(col):
                continue

            # Low-variance: pas de check STRICT (ça n'a pas de sens en "z-mode")
            if (col in LOW_VAR_FEATURES) or (std < LOW_VAR_STD_FLOOR):
                # info-only: si mean est énorme malgré std tiny, on peut le signaler
                continue

            if col in HEAVY_TAIL_RELAXED:
                relax_mean_tol = max(args.mean_tol, 0.5)
                relax_std_low  = min(args.std_low, 0.4)
                relax_std_high = max(args.std_high, 3.5)
                if (abs(mean) > relax_mean_tol) or (std < relax_std_low) or (std > relax_std_high):
                    relaxed_rows.append(row)
                continue

            if (abs(mean) > args.mean_tol) or (std < args.std_low) or (std > args.std_high):
                strict_rows.append(row)

        if strict_rows:
            strict_bad = pd.DataFrame(strict_rows).reset_index(drop=True)
            print("\n⚠️ anomalies STRICT (Z-MODE):")
            print(strict_bad.to_string(index=False))

        if relaxed_rows:
            relaxed_bad = pd.DataFrame(relaxed_rows).reset_index(drop=True)
            print("\nℹ️ anomalies RELAX (heavy-tail): warnings only")
            print(relaxed_bad.to_string(index=False))

    else:
        print("\nℹ️ WIDE-MODE: pas de checks Z stricts (informatif).")

    qconst = feat_stats[feat_stats["std"] < LOW_VAR_STD_FLOOR]
    if not qconst.empty:
        print(f"\n⚠️ low-variance features (std < {LOW_VAR_STD_FLOOR}):")
        print(qconst.to_string(index=False))

    # ===== Weights validation =====
    if args.check_weights:
        w_dir = f"{root.rstrip('/')}/{split}_weight"
        w_files = _maybe_list_dir(fs, w_dir)
        if not w_files:
            print(f"\n[weights] {split}: aucun shard trouvé sous {w_dir} (OK si tu ne les as pas écrits).")
        else:
            print(f"\n[weights] {split}: {len(w_files)} fichiers détectés sous {w_dir}")
            w_stats = ColStats()
            total_w_rows = 0
            for p in w_files[:args.max_files]:
                for chunk in _read_csv_chunks(fs, p, chunksize=args.chunksize):
                    if chunk.shape[1] != 1:
                        print(f"⚠️ {p} : attendu 1 colonne de poids, trouvé {chunk.shape[1]}")
                    w = pd.to_numeric(chunk.iloc[:,0], errors="coerce").to_numpy(dtype=np.float64)
                    w_stats.update(w)
                    total_w_rows += len(w)

            w_mean, w_std = w_stats.finalize()
            print(f"[weights] rows={total_w_rows:,} | mean={w_mean:.4f} std={w_std:.4f} "
                  f"min={w_stats.minv:.4f} max={w_stats.maxv:.4f} "
                  f"nan={w_stats.nan_count} +inf={w_stats.posinf} -inf={w_stats.neginf}")

            if split == "train":
                if not (0.95 <= w_mean <= 1.05):
                    print("⚠️ TRAIN: mean(weight) attendu ≈1 (±5%).")
                if not (w_stats.minv >= 0.0):
                    print("⚠️ TRAIN: weight doit être ≥ 0.")
            else:
                # val/test -> weights sont ones => mean≈1
                if not (0.95 <= w_mean <= 1.05):
                    print("⚠️ OOS: mean(weight) attendu ≈1 (val/test = ones).")
                if not (w_stats.minv >= 0.0):
                    print("⚠️ OOS: weight doit être ≥ 0.")

            if total_rows and total_w_rows:
                delta = abs(total_rows - total_w_rows)
                if delta == 0:
                    print(f"[weights] parité lignes OK (X={total_rows:,}, W={total_w_rows:,}).")
                else:
                    print(f"⚠️ mismatch lignes X({total_rows:,}) vs W({total_w_rows:,}) (max-files={args.max_files}).")

    # ===== IDs validation =====
    ids_dir = f"{root.rstrip('/')}/{split}_ids"
    ids_files = _maybe_list_dir(fs, ids_dir)
    if not ids_files:
        print(f"\n[ids] {split}: aucun shard trouvé sous {ids_dir} (⛔ attendu si Stage3 écrit les ids).")
    else:
        print(f"\n[ids] {split}: {len(ids_files)} fichiers détectés sous {ids_dir}")
        total_ids_rows = 0
        bad_shape = 0
        empty_id = 0

        for p in ids_files[:args.max_files]:
            with fs.open_input_stream(_strip_s3(p)) as f:
                bio = io.BytesIO(f.read())

            for chunk in pd.read_csv(bio, header=None, dtype=str, chunksize=args.chunksize):
                if chunk.shape[1] != 2:
                    bad_shape += 1
                    continue
                total_ids_rows += len(chunk)

                # check non-empty (soft)
                a = chunk.iloc[:,0].fillna("").astype(str)
                b = chunk.iloc[:,1].fillna("").astype(str)
                empty_id += int((a.str.len() == 0).sum() + (b.str.len() == 0).sum())

        if bad_shape:
            print(f"⛔ IDs: {bad_shape} chunk(s) avec shape != 2 colonnes.")
            return 2

        print(f"[ids] rows={total_ids_rows:,} | empty_fields={empty_id:,}")

        if total_rows and total_ids_rows:
            delta = abs(total_rows - total_ids_rows)
            if delta == 0:
                print(f"[ids] parité lignes OK (X={total_rows:,}, IDS={total_ids_rows:,}).")
            else:
                print(f"⚠️ mismatch lignes X({total_rows:,}) vs IDS({total_ids_rows:,}) (max-files={args.max_files}).")

    if args.fail_on_strong and split == "train" and is_z_mode and (not strict_bad.empty):
        print("\n⛔ strong anomalies detected on STRICT Z-features, failing run:")
        print(strict_bad.to_string(index=False))
        return 2

    print("\n✅ split OK (warnings possibles selon heavy-tail / OOS).")
    return 0

# =============================
# Main
# =============================

def main():
    args = parse_args()
    fs = _fs(args.aws_region)

    # meta
    try:
        col_names, features, balance, label_name = _load_meta(fs, args.root)
    except Exception as e:
        print(f"⛔ Problème lecture meta sous {args.root}:\n{e}")
        sys.exit(2)

    print("[meta] columns.json OK")
    if balance:
        try:
            print(f"[meta] train_class_balance: pos={balance.get('pos')} neg={balance.get('neg')} posw_raw={balance.get('scale_pos_weight_raw')}")
        except Exception:
            pass
    
    print(f"[meta] label = {label_name}")

    # sécurité: features interdites
    FORBID = {"THRESH_BPS","pnl_net_max_bps","pnl_net_min_bps"}
    forbid_found = sorted(FORBID & set(features))
    if forbid_found:
        print(f"⛔ features interdits détectés dans columns.json: {forbid_found}")
        sys.exit(2)

    rc = _check_scaler_stats(fs, args.root, features)
    if rc != 0:
        sys.exit(rc)

    splits = ["train","val","test"] if args.split == "all" else [args.split]

    rc_sum = 0
    for sp in splits:
        rc_sum += _validate_one_split(fs, args.root, sp, col_names, args, features)

    sys.exit(0 if rc_sum == 0 else 2)

if __name__ == "__main__":
    main()