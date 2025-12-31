#!/usr/bin/env python3
# validate_stage3_go.py — Validation des CSV Stage3_GO (no header) — streaming + parity checks
#
# Stage3_GO format:
#   - X shards: [is_tradeable] + features (no header)
#   - splits: train / val / test
#   - ids: train_ids / val_ids / test_ids  (row_id,event_id)
#   - weights: train_weight / val_weight / test_weight (single col)
# Meta:
#   - _meta/columns.json (features + label)
#   - _meta/scaler_stats.json (robust params per tf/feature)
#   - _meta/train_class_balance.json (pos/neg + pos_weight_raw)

from __future__ import annotations

import argparse, json, io, sys, math, re
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.fs as pafs

# =============================
# Z-score: listes de contrôle
# =============================

Z_IGNORE_EXACT = {"microprice_bias"}

LOW_VAR_STD_FLOOR = 0.05
LOW_VAR_FEATURES = {"microprice_bias"}

def _is_mask_col(name: str) -> bool:
    return name.endswith("_isnan") or name.endswith("_mask") or name.endswith("_flag")

HEAVY_TAIL_RELAXED = {
    "ret_stdev_1s_10s_bps",
    "quote_churn_10s",
    "spread_bps_entry",
    "bb_width",
    "adx",
    "atr_percentile",
    "atr_bps",
    "bt_dom_3s",
    "bt_dom_5s",
    "bt_dom_10s",
    "lf_bb_width_pct",
    "slope_bid_5",
    "slope_ask_5",
    "slope_bid_15",
    "slope_ask_15",
    "obi_5",
    "obi_15",
    "wall_opp_share_5_buy",
    "wall_opp_share_5_sell",
    "wall_opp_share_15_buy",
    "wall_opp_share_15_sell",
    "slope_opp_5_buy",
    "slope_opp_5_sell",
    "slope_opp_15_buy",
    "slope_opp_15_sell",
    "bt_dom_3s_side_buy",
    "bt_dom_3s_side_sell",
}

# =============================
# Online stats
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

_PART_RE = re.compile(r"part-(\d+)\.csv$", re.IGNORECASE)

def _extract_part_num(path_s3: str) -> Optional[int]:
    m = _PART_RE.search(path_s3)
    return int(m.group(1)) if m else None

def _list_all_csv_files_flat(fs: pafs.S3FileSystem, dir_s3: str) -> List[str]:
    sel = pafs.FileSelector(_strip_s3(dir_s3), recursive=False)
    out = []
    for info in fs.get_file_info(sel):
        if info.is_file and info.path.lower().endswith(".csv"):
            out.append("s3://" + info.path)
    return sorted(out)

def _read_csv_chunks_numeric(fs: pafs.S3FileSystem, path_s3: str, chunksize: int):
    # streaming: no BytesIO(f.read())
    with fs.open_input_stream(_strip_s3(path_s3)) as fbin:
        ftxt = io.TextIOWrapper(fbin, encoding="utf-8", newline="")
        for chunk in pd.read_csv(ftxt, header=None, dtype=np.float64, chunksize=chunksize):
            yield chunk

def _read_csv_chunks_str(fs: pafs.S3FileSystem, path_s3: str, chunksize: int):
    with fs.open_input_stream(_strip_s3(path_s3)) as fbin:
        ftxt = io.TextIOWrapper(fbin, encoding="utf-8", newline="")
        for chunk in pd.read_csv(ftxt, header=None, dtype=str, chunksize=chunksize):
            yield chunk

# =============================
# Meta loader
# =============================

def _load_meta(fs, root: str):
    columns_json_path = f"{root.rstrip('/')}/_meta/columns.json"
    with fs.open_input_stream(_strip_s3(columns_json_path)) as f:
        meta = json.loads(f.read().decode("utf-8"))

    features = meta.get("features")
    label = meta.get("label", "is_tradeable")

    if not isinstance(features, list) or len(features) == 0:
        raise RuntimeError("columns.json invalide: 'features' doit être une liste non vide.")

    if label not in ("Y", "is_tradeable"):
        raise RuntimeError(f"columns.json invalide: label inattendu '{label}' (attendu 'Y' ou 'is_tradeable').")

    col_names = [label] + features

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
    ap = argparse.ArgumentParser("Validate Stage3_GO CSV shards (no header) — streaming + shard parity.")
    ap.add_argument("--root", default="s3://tradebot-config-tokyo/data/stage3/go")
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    ap.add_argument("--aws-region", default="ap-northeast-1")

    ap.add_argument("--max-files", type=int, default=50, help="Nombre max de parts à scanner par split (pour vitesse).")
    ap.add_argument("--chunksize", type=int, default=500_000)

    ap.add_argument("--mean-tol", type=float, default=0.12)
    ap.add_argument("--std-low", type=float, default=0.75)
    ap.add_argument("--std-high", type=float, default=1.30)

    ap.add_argument("--check-weights", action="store_true", default=True)
    ap.add_argument("--check-ids", action="store_true", default=True)

    ap.add_argument("--require-parity", action="store_true", default=True,
                    help="Fail si mismatch parts/rows entre X / ids / weights (sur les fichiers scannés).")

    ap.add_argument("--fail-on-strong", action="store_true",
                    help="Échec si anomalies STRICT sur TRAIN en Z-MODE.")
    return ap.parse_args()

# =============================
# Split runner
# =============================

def _ignore_for_zscore(name: str) -> bool:
    if name in Z_IGNORE_EXACT:
        return True
    if _is_mask_col(name):
        return True
    return False

def _drift_score(row, mean_tol: float, std_low: float, std_high: float) -> float:
    """
    Score simple pour trier les dérives OOS :
    - pénalise mean trop loin
    - pénalise std trop bas/haut
    """
    m = abs(float(row["mean"]))
    s = float(row["std"])
    score = 0.0
    if m > mean_tol:
        score += (m - mean_tol)
    if s < std_low:
        score += (std_low - s)
    if s > std_high:
        score += (s - std_high)
    return score

def _print_table(df: pd.DataFrame, title: str, top_k: int = 25):
    if df.empty:
        return
    print(f"\n{title}")
    if len(df) > top_k:
        df = df.head(top_k)
    print(df.to_string(index=False))

def _validate_one_split(fs, root: str, split: str, col_names: List[str], args, features: List[str]):
    print("\n" + "="*90)
    print(f"🔎 Split: {split}")
    print("="*90)

    label_name = col_names[0]
    n_expected_cols = len(col_names)

    # ----- list X parts -----
    x_dir = f"{root.rstrip('/')}/{split}"
    x_files_all = _list_all_csv_files_flat(fs, x_dir)
    if not x_files_all:
        print("⛔ Aucun fichier CSV trouvé sous", x_dir)
        return 2

    x_files = x_files_all[:args.max_files]
    x_parts = { _extract_part_num(p): p for p in x_files if _extract_part_num(p) is not None }
    print(f"[X] found {len(x_files_all)} file(s) total, scanning {len(x_files)}")
    if len(x_parts) == 0:
        print("⛔ Aucun fichier part-xxxxx.csv reconnu dans", x_dir)
        return 2

    # ----- list IDS/WEIGHTS parts -----
    ids_parts = {}
    w_parts = {}

    if args.check_ids:
        ids_dir = f"{root.rstrip('/')}/{split}_ids"
        ids_files_all = _list_all_csv_files_flat(fs, ids_dir)
        ids_files = ids_files_all[:args.max_files] if ids_files_all else []
        ids_parts = { _extract_part_num(p): p for p in ids_files if _extract_part_num(p) is not None }
        print(f"[IDS] found {len(ids_files_all)} file(s) total, scanning {len(ids_files)}")

    if args.check_weights:
        w_dir = f"{root.rstrip('/')}/{split}_weight"
        w_files_all = _list_all_csv_files_flat(fs, w_dir)
        w_files = w_files_all[:args.max_files] if w_files_all else []
        w_parts = { _extract_part_num(p): p for p in w_files if _extract_part_num(p) is not None }
        print(f"[W] found {len(w_files_all)} file(s) total, scanning {len(w_files)}")

    # ----- parity on part numbers (scanned subset) -----
    scanned_parts = sorted([p for p in x_parts.keys() if p is not None])

    if args.require_parity:
        if args.check_ids:
            missing_ids = [p for p in scanned_parts if p not in ids_parts]
            if missing_ids:
                print(f"⛔ Missing IDS parts for: {missing_ids[:20]} ... (count={len(missing_ids)})")
                return 2
        if args.check_weights:
            missing_w = [p for p in scanned_parts if p not in w_parts]
            if missing_w:
                print(f"⛔ Missing WEIGHT parts for: {missing_w[:20]} ... (count={len(missing_w)})")
                return 2

    # ----- scan X + compute stats -----
    total_rows = 0
    label_counts: Dict[int, int] = {}
    col_stats: Dict[int, ColStats] = {}
    ncols_seen = None

    per_part_rows: Dict[int, int] = {}

    for part in scanned_parts:
        x_path = x_parts[part]
        part_rows = 0

        for chunk in _read_csv_chunks_numeric(fs, x_path, chunksize=args.chunksize):
            if chunk.shape[0] == 0:
                continue

            if ncols_seen is None:
                ncols_seen = chunk.shape[1]
                if ncols_seen != n_expected_cols:
                    print(f"⛔ nb colonnes CSV ({ncols_seen}) ≠ nb colonnes meta ({n_expected_cols}).")
                    print(f"   -> meta = [label]+features = {n_expected_cols} cols")
                    return 2
                print(f"[shape] detected {ncols_seen} columns (matches meta)")

            if chunk.shape[1] != n_expected_cols:
                print(f"⛔ {x_path}: nb colonnes inattendu: {chunk.shape[1]} (attendu {n_expected_cols})")
                return 2

            total_rows += len(chunk)
            part_rows += len(chunk)

            # Label strict: only 0/1
            y = pd.to_numeric(chunk.iloc[:,0], errors="coerce")
            if y.isna().any():
                print(f"⛔ {x_path}: label NaN détecté (chunk).")
                return 2
            yi = y.astype("int64").to_numpy()
            bad = ~np.isin(yi, [0, 1])
            if bad.any():
                ex = yi[bad][:10]
                print(f"⛔ {x_path}: label hors {{0,1}} détecté. Exemples: {ex.tolist()}")
                return 2

            vals, cnts = np.unique(yi, return_counts=True)
            for v, c in zip(vals, cnts):
                label_counts[int(v)] = label_counts.get(int(v), 0) + int(c)

            # Stats globales par colonne
            for j in range(n_expected_cols):
                if j not in col_stats:
                    col_stats[j] = ColStats()
                col_stats[j].update(chunk.iloc[:,j].to_numpy(dtype=np.float64, copy=False))

        per_part_rows[part] = part_rows

    print(f"\n[rows] total scanned: {total_rows:,}")
    if total_rows == 0:
        print("⛔ Aucun enregistrement lu.")
        return 2

    # ----- label balance -----
    print(f"\nℹ️ Label balance ({label_name}):")
    total_lbl = sum(label_counts.values())
    for k in sorted(label_counts.keys()):
        r = label_counts[k] / max(total_lbl, 1)
        print(f"label={k}: count={label_counts[k]:,} ratio={r:.6f}")

    if split == "train":
        pos = int(label_counts.get(1, 0))
        neg = int(label_counts.get(0, 0))
        spw = (float(neg) / float(pos)) if pos > 0 else float("nan")
        print(f"[train] pos={pos:,} neg={neg:,} scale_pos_weight≈{spw:.6g}")

    # ----- pooled feature stats -----
    feat_idx = [i for i in range(1, n_expected_cols) if not _ignore_for_zscore(col_names[i])]

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

    print("\nGLOBAL feature stats (pooled):")
    print(feat_stats.head(140).to_string(index=False))

    # ----- auto-detect z-mode vs wide-mode -----
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

            # low-variance => pas de check strict
            if (col in LOW_VAR_FEATURES) or (std < LOW_VAR_STD_FLOOR):
                continue

            # side_* => drift fréquent => relax (warnings)
            if col.endswith("_side_buy") or col.endswith("_side_sell"):
                relax_mean_tol = max(args.mean_tol, 0.6)
                relax_std_low  = min(args.std_low, 0.35)
                relax_std_high = max(args.std_high, 3.5)
                if (abs(mean) > relax_mean_tol) or (std < relax_std_low) or (std > relax_std_high):
                    relaxed_rows.append(row)
                continue

            # heavy-tail/regime-sensitive => relax (warnings)
            if col in HEAVY_TAIL_RELAXED:
                relax_mean_tol = max(args.mean_tol, 0.5)
                relax_std_low  = min(args.std_low, 0.4)
                relax_std_high = max(args.std_high, 3.5)
                if (abs(mean) > relax_mean_tol) or (std < relax_std_low) or (std > relax_std_high):
                    relaxed_rows.append(row)
                continue

            # STRICT candidates
            if (abs(mean) > args.mean_tol) or (std < args.std_low) or (std > args.std_high):
                strict_rows.append(row)

        # --------------------------
        # Split-aware reporting:
        # - TRAIN: strict_rows => STRICT (can fail with --fail-on-strong)
        # - VAL/TEST: strict_rows => DRIFT report only (never fail)
        # --------------------------
        if strict_rows:
            df_strict = pd.DataFrame(strict_rows).reset_index(drop=True)

            if split == "train":
                strict_bad = df_strict
                _print_table(strict_bad, "\n⚠️ anomalies STRICT (TRAIN, Z-MODE):", top_k=200)
            else:
                df_drift = df_strict.copy()
                df_drift["drift_score"] = df_drift.apply(
                    lambda r: _drift_score(r, args.mean_tol, args.std_low, args.std_high), axis=1
                )
                df_drift = df_drift.sort_values("drift_score", ascending=False)
                _print_table(df_drift, f"\nℹ️ DRIFT (OOS={split}, ex-STRICT Z-check):", top_k=35)

        if relaxed_rows:
            relaxed_bad = pd.DataFrame(relaxed_rows).reset_index(drop=True)
            _print_table(relaxed_bad, "\nℹ️ anomalies RELAX (warnings only):", top_k=200)

    else:
        print("\nℹ️ WIDE-MODE: pas de checks Z stricts (informatif).")

    qconst = feat_stats[feat_stats["std"] < LOW_VAR_STD_FLOOR]
    if not qconst.empty:
        print(f"\n⚠️ low-variance features (std < {LOW_VAR_STD_FLOOR}):")
        print(qconst.to_string(index=False))

    # =========================
    # Weights validation (per-part parity)
    # =========================
    if args.check_weights:
        if not w_parts:
            print(f"\n[weights] {split}: aucun shard trouvé (OK si tu ne les as pas écrits).")
        else:
            w_stats = ColStats()
            total_w_rows = 0
            per_w_rows: Dict[int,int] = {}

            for part in scanned_parts:
                wp = w_parts.get(part)
                if wp is None:
                    continue
                pr = 0
                for chunk in _read_csv_chunks_numeric(fs, wp, chunksize=args.chunksize):
                    if chunk.shape[1] != 1:
                        print(f"⛔ {wp}: attendu 1 colonne de poids, trouvé {chunk.shape[1]}")
                        return 2
                    w = chunk.iloc[:,0].to_numpy(dtype=np.float64, copy=False)
                    w_stats.update(w)
                    pr += len(w)
                    total_w_rows += len(w)
                per_w_rows[part] = pr

            w_mean, w_std = w_stats.finalize()
            print(f"\n[weights] rows={total_w_rows:,} | mean={w_mean:.4f} std={w_std:.4f} "
                  f"min={w_stats.minv:.4f} max={w_stats.maxv:.4f} "
                  f"nan={w_stats.nan_count} +inf={w_stats.posinf} -inf={w_stats.neginf}")

            if w_stats.minv < 0:
                print("⛔ weights négatifs détectés.")
                return 2

            if split == "train":
                if not (0.95 <= w_mean <= 1.05):
                    print("⚠️ TRAIN: mean(weight) attendu ≈1 (±5%).")
            else:
                if not (0.95 <= w_mean <= 1.05):
                    print("⚠️ OOS: mean(weight) attendu ≈1 (val/test = ones).")

            if args.require_parity:
                bad = []
                for part in scanned_parts:
                    xr = per_part_rows.get(part, 0)
                    wr = per_w_rows.get(part, 0)
                    if xr != wr:
                        bad.append((part, xr, wr))
                if bad:
                    print("\n⛔ Parité X vs W cassée sur parts (part, X_rows, W_rows):")
                    print(bad[:20])
                    return 2
                print("[weights] parité X/W OK (par part).")

    # =========================
    # IDs validation (per-part parity)
    # =========================
    if args.check_ids:
        if not ids_parts:
            print(f"\n[ids] {split}: aucun shard trouvé (⛔ attendu si Stage3 écrit les ids).")
            return 2
        else:
            total_ids_rows = 0
            per_ids_rows: Dict[int,int] = {}
            empty_fields = 0
            bad_shape = 0

            for part in scanned_parts:
                ip = ids_parts.get(part)
                if ip is None:
                    continue
                pr = 0
                for chunk in _read_csv_chunks_str(fs, ip, chunksize=args.chunksize):
                    if chunk.shape[1] != 2:
                        bad_shape += 1
                        continue
                    a = chunk.iloc[:,0].fillna("").astype(str)
                    b = chunk.iloc[:,1].fillna("").astype(str)
                    empty_fields += int((a.str.len() == 0).sum() + (b.str.len() == 0).sum())
                    pr += len(chunk)
                    total_ids_rows += len(chunk)
                per_ids_rows[part] = pr

            if bad_shape:
                print(f"⛔ IDs: {bad_shape} chunk(s) avec shape != 2 colonnes.")
                return 2

            print(f"\n[ids] rows={total_ids_rows:,} | empty_fields={empty_fields:,}")

            if args.require_parity:
                bad = []
                for part in scanned_parts:
                    xr = per_part_rows.get(part, 0)
                    ir = per_ids_rows.get(part, 0)
                    if xr != ir:
                        bad.append((part, xr, ir))
                if bad:
                    print("\n⛔ Parité X vs IDS cassée sur parts (part, X_rows, IDS_rows):")
                    print(bad[:20])
                    return 2
                print("[ids] parité X/IDS OK (par part).")

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
    print(f"[meta] n_features = {len(features)} ; n_cols_expected = {len(col_names)}")

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