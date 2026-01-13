#!/usr/bin/env python3
# validate_stage3_go.py — Validation des CSV Stage3_GO (no header) — streaming + parity checks (+ optional *_audit)


from __future__ import annotations

import argparse, json, io, sys, math, re, time
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# =============================
# Z-score: listes de contrôle
# =============================
Z_IGNORE_EXACT = {"aggrbt_missing"}  # raw-only columns to skip in z checks
LOW_VAR_FEATURES = set()              # ou garde vide, microprice_bias ne doit plus être exemptée

LOW_VAR_STD_FLOOR = 0.05

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
    with fs.open_input_stream(_strip_s3(path_s3)) as fbin:
        ftxt = io.TextIOWrapper(fbin, encoding="utf-8", newline="")
        for chunk in pd.read_csv(ftxt, header=None, dtype=np.float64, chunksize=chunksize):
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

def _check_scaler_stats(fs, root: str, features: list[str]) -> set[str]:
    path = f"{root.rstrip('/')}/_meta/scaler_stats.json"
    try:
        with fs.open_input_stream(_strip_s3(path)) as f:
            stats = json.loads(f.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ scaler_stats.json introuvable ou illisible: {e}")
        return set()

    df = pd.DataFrame(stats)
    if df.empty or not {"tf","feature"}.issubset(df.columns):
        print("⚠️ scaler_stats.json mal formé (pas de colonnes tf/feature).")
        return set()

    scaled_set = set(df["feature"].unique())
    expected_scaled = set(features) - Z_IGNORE_EXACT   # si tu veux exclure aggrbt_missing
    missing_scaled = sorted(expected_scaled - scaled_set)
    if missing_scaled:
        print(f"⚠️ scaler_stats missing for features: {missing_scaled[:20]} (count={len(missing_scaled)})")

    raw_features = [f for f in features if f not in scaled_set]
    if raw_features:
        print(f"[meta] RAW (non-normalized) features (no scaler stats expected): {raw_features}")

    if len(scaled_set) == 0:
        print("⚠️ scaler_stats.json vide (aucune feature scalée).")
        return set()

    print("[meta] scaler_stats.json: OK (scaled features present).")
    return scaled_set

# =============================
# CLI
# =============================

def parse_args():
    ap = argparse.ArgumentParser("Validate Stage3_GO CSV shards (no header) — streaming + shard parity.")
    ap.add_argument("--root", default="s3://tradebot-config-tokyo/data/stage3/go")
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    ap.add_argument("--aws-region", default="ap-northeast-1")

    ap.add_argument("--max-files", type=int, default=50, help="Nombre max de parts à scanner par split (pour vitesse).")
    ap.add_argument("--chunksize", type=int, default=50000)

    # IMPORTANT: 0 or negative = unlimited
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Limite globale de lignes à lire pour X (0=illimité). W/IDS seront tronqués à la même lecture effective par part.")
    ap.add_argument("--progress-every", type=int, default=0,
                    help="Log progress toutes les N chunks (0=off).")

    ap.add_argument("--mean-tol", type=float, default=0.12)
    ap.add_argument("--std-low", type=float, default=0.75)
    ap.add_argument("--std-high", type=float, default=1.30)

    ap.add_argument("--check-weights", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--check-ids", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-parity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--check-audit", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--audit-cols", type=str, default="fp_cost_bps,audit_early_abort,audit_timeout,audit_market_toxic",
        help="Colonnes attendues dans *_audit, séparées par des virgules (ordre important)."
    )
    
    # Stage4-ish audit checks
    ap.add_argument("--audit-fp-zero-on-pos", action=argparse.BooleanOptionalAction, default=True,
                    help="Si fp_cost_bps existe: check que fp_cost_bps≈0 quand label==1 (bon trade).")
    ap.add_argument("--audit-fp-zero-tol", type=float, default=1e-6,
                    help="Tolérance pour fp_cost_bps≈0 quand label==1.")

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

def _validate_one_split(fs, root: str, split: str, col_names: List[str], args, features: List[str], scaled_set: set[str]):
    print("\n" + "="*90)
    print(f"🔎 Split: {split}")
    print("="*90)

    label_name = col_names[0]
    n_expected_cols = len(col_names)

    # max_rows: 0 or negative means unlimited
    max_rows = int(args.max_rows)
    if max_rows <= 0:
        max_rows = None  # unlimited

    # ----- list X parts -----
    x_dir = f"{root.rstrip('/')}/{split}"
    x_files_all = _list_all_csv_files_flat(fs, x_dir)
    if not x_files_all:
        print("⛔ Aucun fichier CSV trouvé sous", x_dir)
        return 2

    x_files = x_files_all[:args.max_files]
    x_parts = {_extract_part_num(p): p for p in x_files if _extract_part_num(p) is not None}
    print(f"[X] found {len(x_files_all)} file(s) total, scanning {len(x_files)}")
    print(f"[X] example: {x_files_all[0]}")
    if len(x_parts) == 0:
        print("⛔ Aucun fichier part-xxxxx.csv reconnu dans", x_dir)
        return 2

    # ----- list IDS/WEIGHTS/AUDIT parts -----
    ids_parts: Dict[int, str] = {}
    w_parts: Dict[int, str] = {}
    audit_parts: Dict[int, str] = {}

    if args.check_ids:
        ids_dir = f"{root.rstrip('/')}/{split}_ids"
        ids_files_all = _list_all_csv_files_flat(fs, ids_dir)
        ids_files = ids_files_all[:args.max_files] if ids_files_all else []
        ids_parts = {_extract_part_num(p): p for p in ids_files if _extract_part_num(p) is not None}
        print(f"[IDS] found {len(ids_files_all)} file(s) total, scanning {len(ids_files)}")
        if ids_files_all:
            print(f"[IDS] example: {ids_files_all[0]}")

    if args.check_weights:
        w_dir = f"{root.rstrip('/')}/{split}_weight"
        w_files_all = _list_all_csv_files_flat(fs, w_dir)
        w_files = w_files_all[:args.max_files] if w_files_all else []
        w_parts = {_extract_part_num(p): p for p in w_files if _extract_part_num(p) is not None}
        print(f"[W] found {len(w_files_all)} file(s) total, scanning {len(w_files)}")
        if w_files_all:
            print(f"[W] example: {w_files_all[0]}")

    audit_cols = [c.strip() for c in str(args.audit_cols).split(",") if c.strip()]
    if args.check_audit:
        audit_dir = f"{root.rstrip('/')}/{split}_audit"
        audit_files_all = _list_all_csv_files_flat(fs, audit_dir)
        audit_files = audit_files_all[:args.max_files] if audit_files_all else []
        audit_parts = {_extract_part_num(p): p for p in audit_files if _extract_part_num(p) is not None}
        print(f"[AUDIT] found {len(audit_files_all)} file(s) total, scanning {len(audit_files)}")
        if audit_files_all:
            print(f"[AUDIT] example: {audit_files_all[0]}")
        print(f"[AUDIT] expected columns (no header): {audit_cols}")

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
        if args.check_audit:
            missing_a = [p for p in scanned_parts if p not in audit_parts]
            if missing_a:
                print(f"⛔ Missing AUDIT parts for: {missing_a[:20]} ... (count={len(missing_a)})")
                return 2

    # ----- scan X + compute stats -----
    total_rows = 0
    label_counts: Dict[int, int] = {}
    col_stats: Dict[int, ColStats] = {}
    ncols_seen = None

    per_part_rows: Dict[int, int] = {}
    per_part_lbl: Dict[int, Dict[int, int]] = {}  # part -> {0:count,1:count}

    for part in scanned_parts:
        x_path = x_parts[part]
        t0 = time.time()
        print(f"\n[X] scanning part={part:05d} path={x_path}", flush=True)
        part_rows = 0
        chunk_idx = 0

        for chunk in _read_csv_chunks_numeric(fs, x_path, chunksize=args.chunksize):
            if chunk.shape[0] == 0:
                continue
            chunk_idx += 1

            # Apply max_rows globally on X
            if max_rows is not None:
                remaining = max_rows - total_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining, :]

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
            y = pd.to_numeric(chunk.iloc[:, 0], errors="coerce")
            if y.isna().any():
                print(f"⛔ {x_path}: label NaN détecté (chunk).")
                return 2
            yi = y.astype("int64").to_numpy(copy=False)
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
                col_stats[j].update(chunk.iloc[:, j].to_numpy(dtype=np.float64, copy=False))

            if args.progress_every and (chunk_idx % args.progress_every == 0):
                print(f"[X] part={part:05d} chunk={chunk_idx} rows_part={part_rows:,} rows_total={total_rows:,}")

            if max_rows is not None and total_rows >= max_rows:
                print(f"[X] max_rows reached ({max_rows}), stopping scan early.")
                break

        per_part_rows[part] = part_rows
        print(f"[X] done part={part:05d} rows={part_rows:,} elapsed={time.time()-t0:.2f}s")

        if max_rows is not None and total_rows >= max_rows:
            break

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

    # ----- pooled feature stats (only for scaled features; raw features skipped) -----
    feat_idx = [
        i for i in range(1, n_expected_cols)
        if (col_names[i] in scaled_set) and (not _ignore_for_zscore(col_names[i]))
    ]

    def _finalize(indices: List[int]) -> pd.DataFrame:
        rows = []
        for j in indices:
            cs = col_stats.get(j)
            if cs is None:
                rows.append((j, col_names[j], float("nan"), float("nan"), 0, 0, 0, 0, float("nan"), float("nan")))
                continue
            mu, sd = cs.finalize()
            rows.append((j, col_names[j], mu, sd, cs.n, cs.nan_count, cs.posinf, cs.neginf, cs.minv, cs.maxv))
        return pd.DataFrame(rows, columns=["col", "name", "mean", "std", "n", "nan", "posinf", "neginf", "min", "max"])

    feat_stats = _finalize(feat_idx)

    if len(scaled_set) > 0 and not feat_stats.empty:
        print("\nGLOBAL SCALED feature stats (pooled, informative):")
        print(feat_stats.head(140).to_string(index=False))
        print("\n[scaling] mode → SCALED (robust norm per-tf); pooled mean/std are informative only.")
        is_z_mode = True
    else:
        print("\n[scaling] mode → NO-SCALER / RAW-ONLY (skip Z checks).")
        is_z_mode = False

    strict_bad = pd.DataFrame()

    if is_z_mode:
        strict_rows = []
        relaxed_rows = []

        for _, row in feat_stats.iterrows():
            col = str(row["name"])
            mean = float(row["mean"])
            std = float(row["std"])

            if _ignore_for_zscore(col):
                continue

            if (col in LOW_VAR_FEATURES) or (std < LOW_VAR_STD_FLOOR):
                continue

            if col.endswith("_side_buy") or col.endswith("_side_sell"):
                relax_mean_tol = max(args.mean_tol, 0.6)
                relax_std_low = min(args.std_low, 0.35)
                relax_std_high = max(args.std_high, 3.5)
                if (abs(mean) > relax_mean_tol) or (std < relax_std_low) or (std > relax_std_high):
                    relaxed_rows.append(row)
                continue

            if col in HEAVY_TAIL_RELAXED:
                relax_mean_tol = max(args.mean_tol, 0.5)
                relax_std_low = min(args.std_low, 0.4)
                relax_std_high = max(args.std_high, 3.5)
                if (abs(mean) > relax_mean_tol) or (std < relax_std_low) or (std > relax_std_high):
                    relaxed_rows.append(row)
                continue

            if (abs(mean) > args.mean_tol) or (std < args.std_low) or (std > args.std_high):
                strict_rows.append(row)

        if strict_rows:
            df_strict = pd.DataFrame(strict_rows).reset_index(drop=True)
            if split == "train":
                strict_bad = df_strict
                _print_table(strict_bad, "\n⚠️ anomalies STRICT (TRAIN, SCALED-features):", top_k=200)
            else:
                df_drift = df_strict.copy()
                df_drift["drift_score"] = df_drift.apply(
                    lambda r: _drift_score(r, args.mean_tol, args.std_low, args.std_high), axis=1
                )
                df_drift = df_drift.sort_values("drift_score", ascending=False)
                _print_table(df_drift, f"\nℹ️ DRIFT (OOS={split}, ex-STRICT scaled-check):", top_k=35)

        if relaxed_rows:
            relaxed_bad = pd.DataFrame(relaxed_rows).reset_index(drop=True)
            _print_table(relaxed_bad, "\nℹ️ anomalies RELAX (warnings only):", top_k=200)
    else:
        print("\nℹ️ no-scaler: pas de checks Z (informatif).")

    # =========================
    # Weights validation (aligned truncation per-part)
    # =========================
    if args.check_weights:
        if not w_parts:
            print(f"\n[weights] {split}: aucun shard trouvé (OK si tu ne les as pas écrits).")
        else:
            w_stats = ColStats()
            total_w_rows = 0
            per_w_rows: Dict[int, int] = {}

            for part in scanned_parts:
                x_target = per_part_rows.get(part, 0)
                if x_target <= 0:
                    continue

                wp = w_parts.get(part)
                if wp is None:
                    continue

                print(f"\n[W] scanning part={part:05d} path={wp}")
                pr = 0
                chunk_idx = 0
                for chunk in _read_csv_chunks_numeric(fs, wp, chunksize=args.chunksize):
                    if chunk.shape[1] != 1:
                        print(f"⛔ {wp}: attendu 1 colonne de poids, trouvé {chunk.shape[1]}")
                        return 2
                    if pr >= x_target:
                        break

                    remaining = x_target - pr
                    if len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining, :]

                    w = chunk.iloc[:, 0].to_numpy(dtype=np.float64, copy=False)
                    w_stats.update(w)

                    pr += len(w)
                    total_w_rows += len(w)
                    chunk_idx += 1

                    if args.progress_every and (chunk_idx % args.progress_every == 0):
                        print(f"[W] part={part:05d} chunk={chunk_idx} rows_part={pr:,} rows_total={total_w_rows:,}")

                per_w_rows[part] = pr
                print(f"[W] done part={part:05d} rows={pr:,}")

            w_mean, w_std = w_stats.finalize()
            print(f"\n[weights] rows={total_w_rows:,} | mean={w_mean:.4f} std={w_std:.4f} "
                  f"min={w_stats.minv:.4f} max={w_stats.maxv:.4f} "
                  f"nan={w_stats.nan_count} +inf={w_stats.posinf} -inf={w_stats.neginf}")

            if w_stats.minv < 0:
                print("⛔ weights négatifs détectés.")
                return 2

            if args.require_parity:
                bad = []
                for part, xr in per_part_rows.items():
                    if xr <= 0:
                        continue
                    wr = per_w_rows.get(part, 0)
                    if xr != wr:
                        bad.append((part, xr, wr))
                if bad:
                    print("\n⛔ Parité X vs W cassée sur parts (part, X_rows, W_rows):")
                    print(bad[:20])
                    return 2
                print("[weights] parité X/W OK (par part).")

    # =========================
    # AUDIT validation (LOCKSTEP with X; no buffering)
    # Layout: {root}/{split}_audit/part-*.csv  (no header)
    # =========================
    if args.check_audit:
        if not audit_cols:
            print("\n⛔ audit-cols vide. Donne au moins une colonne via --audit-cols.")
            return 2

        if not audit_parts:
            print(f"\n[audit] {split}: aucun shard trouvé (⛔ attendu si tu actives --check-audit).")
            return 2

        total_a_rows = 0
        per_a_rows: Dict[int, int] = {}
        bad_shape = 0

        a_stats: Dict[int, ColStats] = {j: ColStats() for j in range(len(audit_cols))}

        idx_fp  = audit_cols.index("fp_cost_bps") if "fp_cost_bps" in audit_cols else None
        idx_A   = audit_cols.index("audit_early_abort") if "audit_early_abort" in audit_cols else None
        idx_B   = audit_cols.index("audit_timeout") if "audit_timeout" in audit_cols else None
        idx_C   = audit_cols.index("audit_market_toxic") if "audit_market_toxic" in audit_cols else None

        # split_stats (always ON)
        A_sum = 0
        B_sum = 0
        C_sum = 0
        fp_sum = 0.0
        fp_n = 0

        C_total = 0
        C_ones = 0
        fp_sum_C1 = 0.0
        fp_sum_C0 = 0.0
        fp_n_C1 = 0
        fp_n_C0 = 0

        fp_pos_bad = 0
        fp_pos_n = 0

        for part in scanned_parts:
            x_target = per_part_rows.get(part, 0)
            if x_target <= 0:
                continue

            apath = audit_parts.get(part)
            if apath is None:
                continue

            print(f"\n[AUDIT] scanning part={part:05d} path={apath}")
            pr = 0
            chunk_idx = 0

            x_iter = _read_csv_chunks_numeric(fs, x_parts[part], chunksize=args.chunksize)
            a_iter = _read_csv_chunks_numeric(fs, apath, chunksize=args.chunksize)

            while pr < x_target:
                try:
                    xchunk = next(x_iter)
                except StopIteration:
                    break
                try:
                    achunk = next(a_iter)
                except StopIteration:
                    print(f"⛔ {apath}: audit ended before X for part={part:05d}")
                    return 2

                if xchunk.shape[0] == 0:
                    continue

                if achunk.shape[1] != len(audit_cols):
                    bad_shape += 1
                    break

                remaining = x_target - pr
                nmin = min(len(xchunk), len(achunk), remaining)
                if nmin <= 0:
                    break

                if len(xchunk) != nmin:
                    xchunk = xchunk.iloc[:nmin, :]
                if len(achunk) != nmin:
                    achunk = achunk.iloc[:nmin, :]

                arr = achunk.to_numpy(dtype=np.float64, copy=False)

                # y (déjà validé plus haut, mais on reste safe)
                yv = xchunk.iloc[:, 0].to_numpy(dtype=np.float64, copy=False)
                yi = yv.astype(np.int64, copy=False)

                # split_stats (always ON)
                if idx_A is not None:
                    vA = arr[:, idx_A]
                    mA = np.isfinite(vA)
                    if mA.any():
                        A_sum += int(np.round(vA[mA]).astype(np.int64).sum())

                if idx_B is not None:
                    vB = arr[:, idx_B]
                    mB = np.isfinite(vB)
                    if mB.any():
                        B_sum += int(np.round(vB[mB]).astype(np.int64).sum())

                if idx_C is not None:
                    vC = arr[:, idx_C]
                    mC = np.isfinite(vC)
                    if mC.any():
                        C_sum += int(np.round(vC[mC]).astype(np.int64).sum())

                if idx_fp is not None:
                    fpv = arr[:, idx_fp]
                    mf = np.isfinite(fpv)
                    if mf.any():
                        fp_sum += float(fpv[mf].sum())
                        fp_n += int(mf.sum())

                # fp_cost_bps mean | market_toxic=1 vs 0
                if (idx_fp is not None) and (idx_C is not None):
                    fpv = arr[:, idx_fp]
                    Cv  = arr[:, idx_C]
                    m = np.isfinite(fpv) & np.isfinite(Cv)
                    if m.any():
                        fpv2 = fpv[m]
                        Cv2  = Cv[m].astype(np.int64, copy=False)

                        m1 = (Cv2 == 1)
                        if m1.any():
                            fp_sum_C1 += float(fpv2[m1].sum())
                            fp_n_C1   += int(m1.sum())

                        m0 = ~m1
                        if m0.any():
                            fp_sum_C0 += float(fpv2[m0].sum())
                            fp_n_C0   += int(m0.sum())

                # market_toxic_rate (indépendant de fp_cost_bps)
                if idx_C is not None:
                    Cv = arr[:, idx_C]
                    mC = np.isfinite(Cv)
                    if mC.any():
                        Cv2 = Cv[mC].astype(np.int64, copy=False)
                        C_total += int(Cv2.size)
                        C_ones  += int((Cv2 == 1).sum())
                        
                if args.audit_fp_zero_on_pos and (idx_fp is not None):
                    fpv = arr[:, idx_fp]
                    m = np.isfinite(fpv)
                    if m.any():
                        fpv2 = fpv[m]
                        y2 = yi[m]
                        pos = (y2 == 1)
                        if pos.any():
                            fp_pos_n += int(pos.sum())
                            fp_pos_bad += int((np.abs(fpv2[pos]) > float(args.audit_fp_zero_tol)).sum())

                # domain checks + per-col stats
                for j, name in enumerate(audit_cols):
                    v = arr[:, j]
                    a_stats[j].update(v)

                    if name == "fp_cost_bps":
                        bad = (~np.isfinite(v)) | (v < -1e-9)
                        if bad.any():
                            ex = v[bad][:10]
                            print(f"⛔ {apath}: fp_cost_bps invalid (NaN/inf or <0). Examples: {ex.tolist()}")
                            return 2

                    if name == "audit_market_toxic":
                        vv = v[np.isfinite(v)]
                        if vv.size:
                            bad = ~np.isin(vv.astype(np.int64), [0, 1]) | (np.abs(vv - np.round(vv)) > 1e-9)
                            if bad.any():
                                ex = vv[bad][:10]
                                print(f"⛔ {apath}: audit_market_toxic not in {{0,1}}. Examples: {ex.tolist()}")
                                return 2

                pr += nmin
                total_a_rows += nmin
                chunk_idx += 1

                if args.progress_every and (chunk_idx % args.progress_every == 0):
                    print(f"[AUDIT] part={part:05d} chunk={chunk_idx} rows_part={pr:,} rows_total={total_a_rows:,}")

            per_a_rows[part] = pr
            print(f"[AUDIT] done part={part:05d} rows={pr:,}")

        if bad_shape:
            print(f"⛔ AUDIT: {bad_shape} part(s) avec shape != {len(audit_cols)} colonnes.")
            return 2

        print(f"\n[audit] rows={total_a_rows:,} | cols={audit_cols}")
        for j, name in enumerate(audit_cols):
            mu, sd = a_stats[j].finalize()
            print(f"[audit] {name}: mean={mu:.6g} std={sd:.6g} min={a_stats[j].minv:.6g} max={a_stats[j].maxv:.6g} "
                  f"nan={a_stats[j].nan_count} +inf={a_stats[j].posinf} -inf={a_stats[j].neginf}")
        
        if idx_C is not None and C_total > 0:
            print(f"[audit] market_toxic_rate={C_ones/max(C_total,1):.6g} (ones={C_ones} total={C_total})")
        
        if (idx_fp is not None) and (idx_C is not None):
            m1 = fp_sum_C1 / max(fp_n_C1, 1)
            m0 = fp_sum_C0 / max(fp_n_C0, 1)
            print(f"[audit] fp_cost_bps mean | market_toxic=1: {m1:.6g} (n={fp_n_C1}) | market_toxic=0: {m0:.6g} (n={fp_n_C0})")

        if args.audit_fp_zero_on_pos and (idx_fp is not None) and fp_pos_n > 0:
            bad_rate = fp_pos_bad / max(fp_pos_n, 1)
            print(f"[audit] fp_cost_bps≈0 on y==1: bad={fp_pos_bad}/{fp_pos_n} rate={bad_rate:.6g} tol={args.audit_fp_zero_tol}")
            if fp_pos_bad > 0:
                print("⛔ audit: fp_cost_bps non-zero détecté sur label==1 (check audit-fp-zero-on-pos).")
                return 2

        # split_stats summary (always ON)
        fp_mean = (fp_sum / max(fp_n, 1)) if idx_fp is not None else float("nan")
        print(f"[audit][split_stats] sum_A={A_sum} sum_B={B_sum} sum_C={C_sum} fp_cost_mean={fp_mean:.6g} (n={fp_n})")

        if args.require_parity:
            bad = []
            for part, xr in per_part_rows.items():
                if xr <= 0:
                    continue
                ar = per_a_rows.get(part, 0)
                if xr != ar:
                    bad.append((part, xr, ar))
            if bad:
                print("\n⛔ Parité X vs AUDIT cassée sur parts (part, X_rows, AUDIT_rows):")
                print(bad[:20])
                return 2
            print("[audit] parité X/AUDIT OK (par part).")

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

    scaled_set = _check_scaler_stats(fs, args.root, features)

    splits = ["train","val","test"] if args.split == "all" else [args.split]

    rc_sum = 0
    for sp in splits:
        try:
            rc_sum += _validate_one_split(fs, args.root, sp, col_names, args, features, scaled_set)
        except Exception as e:
            print(f"⛔ Exception in split={sp}: {type(e).__name__}: {e}")
            raise

    sys.exit(0 if rc_sum == 0 else 2)

if __name__ == "__main__":
    main()