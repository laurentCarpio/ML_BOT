#!/usr/bin/env python3
# make_stage3_go.py — Stage2_split(go/{train,val,test}) -> Stage3 GO (no internal split)
#
# Input (Stage2 v4.1):
#  s3://.../data/stage2/split/{train,val,test}/go/{SYMBOL}/{YEAR}/parts/part-*.parquet
#
# Output:
#   - {dst_root}/{split}/part-*.csv          : X (first col = is_tradeable, then features)
#   - {dst_root}/{split}_ids/part-*.csv      : ids (row_id,event_id) aligned with X parts
#   - {dst_root}/{split}_weight/part-*.csv   : sample weights aligned with X parts
#   - {dst_root}/_meta/*
#
# Notes:
#   - aggrbt_missing is INCLUDED in X as a raw (non-normalized) feature.

from __future__ import annotations

import argparse, io, json, sys, random
from typing import Optional, Dict, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

# =======================
# GO feature policy (Stage2 v4.1 names)
# =======================
# IMPORTANT:
# - Features in GO_FEATURES_NORM are robust-normalized.
# - Features in GO_FEATURES_RAW are appended to X without normalization.

GO_FEATURES = [
    "spread_bps_entry",
    "atr_bps",
    "quote_churn_10s",
    "ret_stdev_1s_10s_bps",
    "microprice_bias",
    "obi_5",
    "obi_15",

    "wall_opp_share_5_buy",
    "wall_opp_share_5_sell",
    "wall_opp_share_15_buy",
    "wall_opp_share_15_sell",

    "cum_depth_within_5bps_opp_buy",
    "cum_depth_within_5bps_opp_sell",
    "cum_depth_within_10bps_opp_buy",
    "cum_depth_within_10bps_opp_sell",

    "slope_bid_5",
    "slope_ask_5",
    "slope_bid_15",
    "slope_ask_15",

    "slope_opp_5_buy",
    "slope_opp_5_sell",
    "slope_opp_15_buy",
    "slope_opp_15_sell",

    "aggr_ratio_10s",
    "aggr_ratio_15s",
    "bt_dom_3s",
    "bt_dom_5s",
    "bt_dom_10s",

    "obi_5_side_buy",
    "obi_5_side_sell",
    "obi_15_side_buy",
    "obi_15_side_sell",
    "microprice_bias_side_buy",
    "microprice_bias_side_sell",
    "aggr_ratio_10s_side_buy",
    "aggr_ratio_10s_side_sell",
    "aggr_ratio_15s_side_buy",
    "aggr_ratio_15s_side_sell",
    "bt_dom_3s_side_buy",
    "bt_dom_3s_side_sell",
    "bt_dom_5s_side_buy",
    "bt_dom_5s_side_sell",
    "bt_dom_10s_side_buy",
    "bt_dom_10s_side_sell",

    "bb_width",
    "bb_width_pctl",
    "lf_bb_width_pct",
    "atr_percentile",
    "lf_atr_rank_30m",
    "atr_pct_rank_30m",
    "adx",
]

GO_FEATURES_RAW = [
     "aggrbt_missing",   # raw flag (0/1) indicating aggr/bt imputation happened on the row
]
 
GO_FEATURES_NORM = GO_FEATURES  # backward-friendly alias (explicitly: normalized list)


LOG1P_FEATURES = [
    "cum_depth_within_5bps_opp_buy",
    "cum_depth_within_5bps_opp_sell",
    "cum_depth_within_10bps_opp_buy",
    "cum_depth_within_10bps_opp_sell",
    "quote_churn_10s",
    "ret_stdev_1s_10s_bps",
]

ASINH_FEATURES = [
    "microprice_bias",
    "slope_bid_5", "slope_ask_5",
    "slope_bid_15", "slope_ask_15",
    "slope_opp_5_buy", "slope_opp_5_sell",
    "slope_opp_15_buy", "slope_opp_15_sell",
    "microprice_bias_side_buy", "microprice_bias_side_sell",
]

ROBUST_CLIP_Q = (0.005, 0.995)
ROBUST_EPS = 1e-9

FEATURE_FLOOR_Q = 0.10
FEATURE_FLOOR_EPS = 1e-6

META_DIR = "_meta"

AGGR_BT_COLS = [
    "aggr_ratio_10s","aggr_ratio_15s","bt_dom_3s","bt_dom_5s","bt_dom_10s",
    "aggr_ratio_10s_side_buy","aggr_ratio_10s_side_sell",
    "aggr_ratio_15s_side_buy","aggr_ratio_15s_side_sell",
    "bt_dom_3s_side_buy","bt_dom_3s_side_sell",
    "bt_dom_5s_side_buy","bt_dom_5s_side_sell",
    "bt_dom_10s_side_buy","bt_dom_10s_side_sell",
]

AGGR_BT_NEUTRAL_01 = 0.5   # neutre pour colonnes dans [0,1]
AGGR_BT_NEUTRAL_M11 = 0.0  # neutre pour colonnes dans [-1,1]

def _neutral_fill_value(col: str) -> float:
    # convention: *_side_* sont dans [-1,1], le reste aggr/bt est dans [0,1]
    return AGGR_BT_NEUTRAL_M11 if "_side_" in col else AGGR_BT_NEUTRAL_01

# =======================
# Utils
# =======================

def _norm_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore").strip()
    return str(x).strip()

def _tf_key(x) -> str:
    return _norm_str(x).lower().strip()

def _task_key(x) -> str:
    return _norm_str(x).lower().strip()

def _fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _strip_s3(uri: str) -> str:
    assert uri.startswith("s3://")
    return uri[len("s3://"):]

def _open_s3_output(fs: pafs.S3FileSystem, path_s3: str):
    return fs.open_output_stream(_strip_s3(path_s3))

def _dataset_from_split_go(fs: pafs.S3FileSystem, src_stage2_split_root: str, split: str) -> ds.Dataset:
    """
    Stage2 split layout (v4.1):
      {src_stage2_split_root}/{split}/go/{SYMBOL}/{YEAR}/parts/part-*.parquet
    """
    base = _strip_s3(f"{src_stage2_split_root.rstrip('/')}/{split}/go")
    return ds.dataset(base, filesystem=fs, format="parquet")

def _optional_filter(schema: pa.Schema,
                     symbols: Optional[List[str]],
                     tfs: Optional[List[str]]) -> Optional[ds.Expression]:
    expr = None
    if symbols and "symbol" in schema.names:
        expr = ds.field("symbol").isin([str(x) for x in symbols])
    if tfs and "tf" in schema.names:
        tf_norm = [str(x).lower().strip() for x in tfs]
        tf_expr = ds.field("tf").isin(tf_norm)
        expr = tf_expr if expr is None else (expr & tf_expr)
    return expr

def _rows_to_csv_bytes(arr2d: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savetxt(buf, arr2d, delimiter=",", fmt="%.6g")
    return buf.getvalue()

def _rows_to_csv_bytes_str(arr2d: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savetxt(buf, arr2d, delimiter=",", fmt="%s")
    return buf.getvalue()

# =======================
# Transforms + Robust norm
# =======================

def _pre_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in LOG1P_FEATURES:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df[c] = np.log1p(v.clip(lower=0))
    for c in ASINH_FEATURES:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df[c] = np.arcsinh(v)
    return df

def _reservoir_add(arr: List[float], x: float, k: int, seen: int, rng: random.Random):
    if len(arr) < k:
        arr.append(x)
        return
    j = rng.randrange(seen)
    if j < k:
        arr[j] = x

def _mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def _fit_robust_params_train(
    dset_train: ds.Dataset,
    filt: Optional[ds.Expression],
    features: List[str],
    batch_size: int,
    sample_per_tf_feature: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    avail = set(dset_train.schema.names)

    proj = ["tf"] + [c for c in features if c in avail]
    proj = sorted(set([c for c in proj if c in avail]))

    scanner = dset_train.scanner(columns=proj, filter=filt, batch_size=batch_size)

    qlo, qhi = ROBUST_CLIP_Q
    res = defaultdict(list)   # (tf, feat) -> reservoir
    seen = defaultdict(int)   # (tf, feat) -> count

    for b in scanner.to_batches():
        df = b.to_pandas()
        if df.empty:
            continue

        df["tf"] = df["tf"].map(_tf_key)
        df = _pre_transforms(df)

        for tfv, g in df.groupby("tf", sort=False):
            tfv = str(tfv)
            for f in features:
                if f not in g.columns:
                    continue
                x = pd.to_numeric(g[f], errors="coerce").to_numpy(dtype="float64", copy=False)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue
                key = (tfv, f)
                for v in x:
                    seen[key] += 1
                    _reservoir_add(res[key], float(v), sample_per_tf_feature, seen[key], rng)

    rows_raw = []
    for (tfv, f), xs in res.items():
        x = np.asarray(xs, dtype="float64")
        if x.size == 0:
            rows_raw.append((tfv, f, np.nan, np.nan, np.nan, 0.0, 0.0))
            continue

        q1 = float(np.nanquantile(x, qlo))
        q99 = float(np.nanquantile(x, qhi))
        xc = np.clip(x, q1, q99)

        med = float(np.nanmedian(xc))
        q25 = float(np.nanquantile(xc, 0.25))
        q75 = float(np.nanquantile(xc, 0.75))
        iqr = max(q75 - q25, 0.0)
        scale_iqr = (iqr / 1.349) if iqr > 0 else 0.0

        mad = _mad(xc)
        scale_mad = 1.4826 * mad if mad > 0 else 0.0

        scale_raw = max(scale_iqr, scale_mad)
        if not np.isfinite(scale_raw) or scale_raw < ROBUST_EPS:
            scale_raw = np.nan

        rows_raw.append((tfv, f, q1, q99, med, scale_raw, scale_mad))

    raw_df = pd.DataFrame(rows_raw, columns=["tf", "feature", "q1", "q99", "median", "scale_raw", "scale_mad"])

    floors = {}
    for feat, s in raw_df.groupby("feature")["scale_raw"]:
        v = s.to_numpy(dtype="float64")
        v = v[np.isfinite(v) & (v > 0)]
        floors[feat] = max(float(np.quantile(v, FEATURE_FLOOR_Q)), FEATURE_FLOOR_EPS) if v.size else FEATURE_FLOOR_EPS

    raw_df["scale_floor"] = raw_df["feature"].map(lambda f: max(float(floors.get(f, 0.0)), FEATURE_FLOOR_EPS))

    sr = raw_df["scale_raw"].to_numpy(dtype="float64")
    sf = raw_df["scale_floor"].to_numpy(dtype="float64")
    raw_df["scale"] = np.where(np.isfinite(sr) & (sr > 0), sr, sf)

    return raw_df[["tf", "feature", "q1", "q99", "median", "scale", "scale_floor", "scale_raw", "scale_mad"]].copy()

def _apply_norm(
    df: pd.DataFrame,
    params_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
    features: List[str]
) -> pd.DataFrame:
    # NOTE: Only pass "normalized" features here.
    df = df.copy()
    df["tf"] = df["tf"].map(_tf_key)
    df = _pre_transforms(df)

    groups = df.groupby("tf", sort=False).groups
    for tfv, idx in groups.items():
        pm = params_map.get(str(tfv))
        if not pm:
            continue
        for f in features:
            if f not in df.columns or f not in pm:
                continue
            q1, q99, med, scale = pm[f]
            x = pd.to_numeric(df.loc[idx, f], errors="coerce").astype("float64")
            x = np.clip(x, q1, q99)
            denom = scale if (np.isfinite(scale) and scale >= ROBUST_EPS) else 1.0
            df.loc[idx, f] = (x - med) / denom

    # final sanitize: no NaN/inf survives
    for f in features:
        if f in df.columns:
            x = pd.to_numeric(df[f], errors="coerce").astype("float64")
            x[~np.isfinite(x)] = 0.0
            df[f] = x

    return df

# =======================
# Single-pass shard writer: X + ids + weights
# =======================

def _write_x_ids_w_audit_shards_stream(
    fs: pafs.S3FileSystem,
    base_out_x: str,
    base_out_ids: str,
    base_out_w: str,
    base_out_audit: str,
    iterator_x_ids_w_audit,
    rows_per_shard: int
) -> Tuple[int, int, int, int]:
    part = 0
    written_x = written_ids = written_w = written_a = 0

    acc_x: List[np.ndarray] = []
    acc_ids: List[np.ndarray] = []
    acc_w: List[np.ndarray] = []
    acc_a: List[np.ndarray] = []
    cur = 0

    def _dump_x(path, arr2d):
        with _open_s3_output(fs, path) as out:
            out.write(_rows_to_csv_bytes(arr2d))

    def _dump_ids(path, arr2d):
        with _open_s3_output(fs, path) as out:
            out.write(_rows_to_csv_bytes_str(arr2d))

    def _dump_w(path, w1d):
        v = w1d.reshape(-1, 1)
        with _open_s3_output(fs, path) as out:
            out.write(_rows_to_csv_bytes(v))

    def _dump_audit(path, arr2d):
        with _open_s3_output(fs, path) as out:
            out.write(_rows_to_csv_bytes(arr2d))

    for X, ids, w, audit2d in iterator_x_ids_w_audit:
        if X is None or X.size == 0: continue
        if ids is None or ids.size == 0: continue
        if w is None or w.size == 0: continue
        if audit2d is None or audit2d.size == 0: continue

        if not (X.shape[0] == ids.shape[0] == w.shape[0] == audit2d.shape[0]):
            raise ValueError(f"Row mismatch: X={X.shape[0]} ids={ids.shape[0]} w={w.shape[0]} audit={audit2d.shape[0]}")

        acc_x.append(X)
        acc_ids.append(ids)
        acc_w.append(w.astype("float32", copy=False))
        acc_a.append(audit2d.astype("float32", copy=False))
        cur += X.shape[0]

        while cur >= rows_per_shard:
            big_x = np.vstack(acc_x)
            big_i = np.vstack(acc_ids)
            big_w = np.concatenate(acc_w, axis=0)
            big_a = np.vstack(acc_a)

            head_x, tail_x = big_x[:rows_per_shard], big_x[rows_per_shard:]
            head_i, tail_i = big_i[:rows_per_shard], big_i[rows_per_shard:]
            head_w, tail_w = big_w[:rows_per_shard], big_w[rows_per_shard:]
            head_a, tail_a = big_a[:rows_per_shard], big_a[rows_per_shard:]

            px = f"{base_out_x.rstrip('/')}/part-{part:05d}.csv"
            pi = f"{base_out_ids.rstrip('/')}/part-{part:05d}.csv"
            pw = f"{base_out_w.rstrip('/')}/part-{part:05d}.csv"
            pa_ = f"{base_out_audit.rstrip('/')}/part-{part:05d}.csv"

            _dump_x(px, head_x)
            _dump_ids(pi, head_i)
            _dump_w(pw, head_w)
            _dump_audit(pa_, head_a)

            written_x += head_x.shape[0]
            written_ids += head_i.shape[0]
            written_w += head_w.shape[0]
            written_a += head_a.shape[0]
            part += 1

            acc_x = [tail_x] if tail_x.size else []
            acc_ids = [tail_i] if tail_i.size else []
            acc_w = [tail_w] if tail_w.size else []
            acc_a = [tail_a] if tail_a.size else []
            cur = tail_x.shape[0] if tail_x.size else 0

    if acc_x:
        big_x = np.vstack(acc_x)
        big_i = np.vstack(acc_ids)
        big_w = np.concatenate(acc_w, axis=0)
        big_a = np.vstack(acc_a)

        px = f"{base_out_x.rstrip('/')}/part-{part:05d}.csv"
        pi = f"{base_out_ids.rstrip('/')}/part-{part:05d}.csv"
        pw = f"{base_out_w.rstrip('/')}/part-{part:05d}.csv"
        pa_ = f"{base_out_audit.rstrip('/')}/part-{part:05d}.csv"

        _dump_x(px, big_x)
        _dump_ids(pi, big_i)
        _dump_w(pw, big_w)
        _dump_audit(pa_, big_a)

        written_x += big_x.shape[0]
        written_ids += big_i.shape[0]
        written_w += big_w.shape[0]
        written_a += big_a.shape[0]

    return written_x, written_ids, written_w, written_a

# =======================
# Core: build one split (now yields X, ids, y for class balance)
# =======================

AUDIT_PNL_COL = "audit_pnl_net_bps"

def _derive_toxic_from_exit_reason(aer: pd.Series) -> np.ndarray:
    s = aer.astype(str).str.upper()
    return s.str.contains("TOXIC", na=False).to_numpy(dtype=np.int8)

def _iter_x_ids_y_from_split(
    dset: ds.Dataset,
    base_filt: Optional[ds.Expression],
    feats: List[str],
    params_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
    batch_size: int,
    stats: Optional[dict] = None,
):
    schema_names = set(dset.schema.names)

    needed = sorted(set([
        "tf", "y_go", "row_id", "event_id", "task",
        "audit_exit_reason",
        "audit_early_abort",
        "audit_timeout",
        "audit_market_toxic",
        AUDIT_PNL_COL,
    ]) | set(feats) | set(AGGR_BT_COLS))
    needed = [c for c in needed if c in schema_names]

    scanner = dset.scanner(columns=needed, filter=base_filt, batch_size=batch_size)

    for b in scanner.to_batches():
        df = b.to_pandas()
        if df.empty:
            continue

        if df[["row_id", "event_id", "task", "tf"]].isna().any(axis=1).any():
            bad = df[df[["row_id", "event_id", "task", "tf"]].isna().any(axis=1)].head(5)
            raise ValueError(f"NULL in (row_id,event_id,task,tf) detected. Examples:\n{bad}")

        task_norm = df["task"].map(_task_key)
        if not (task_norm == "go").all():
            bad = df.loc[task_norm != "go", ["row_id", "task", "event_id", "tf"]].head(5)
            raise ValueError(f"Stage3 GO expects task=='go' only. Examples:\n{bad}")

        tf_norm = df["tf"].map(_tf_key)
        exp = df["row_id"].astype(str) + "|go|" + tf_norm.astype(str)
        eid = df["event_id"].astype(str)
        if not (eid == exp).all():
            bad = df.loc[eid != exp, ["row_id", "event_id", "tf", "task"]].head(5)
            raise ValueError(f"event_id formula mismatch. Examples:\n{bad}")

        if stats is not None:
            stats["rows_in"] = stats.get("rows_in", 0) + int(len(df))

        # --- aggr/bt NaN safety net ---
        present = [c for c in AGGR_BT_COLS if c in df.columns]
        if present:
            nan_mask = df[present].isna().any(axis=1)
            df["aggrbt_missing"] = nan_mask.astype("int8")
            if stats is not None:
                stats["imputed_nan_aggrbt"] = stats.get("imputed_nan_aggrbt", 0) + int(nan_mask.sum())
            for c in present:
                v = pd.to_numeric(df[c], errors="coerce").astype("float64")
                df[c] = v.fillna(_neutral_fill_value(c))
        else:
            df["aggrbt_missing"] = 0

        tf_norm = tf_norm.loc[df.index]

        ids2d = np.column_stack([
            df["row_id"].astype(str).to_numpy(dtype=object, copy=False),
            df["event_id"].astype(str).to_numpy(dtype=object, copy=False),
        ])

        # --- base label y_go sanity ---
        y_raw = pd.to_numeric(df["y_go"], errors="coerce").fillna(0).astype("int8").clip(0, 1)
        if not y_raw.isin([0, 1]).all():
            bad = df.loc[~y_raw.isin([0, 1]), ["row_id", "event_id", "y_go"]].head(5)
            raise ValueError(f"y_go not in {{0,1}}. Examples:\n{bad}")

        # --- coverage gate for positives ONLY ---
        if "audit_exit_reason" in df.columns:
            aer = df["audit_exit_reason"].astype(str)
            pos_has_exit = (aer != "NONE").to_numpy(dtype=bool)

            # force y_go=1 but exit_reason==NONE -> y=0 (invalid positive)
            invalid_pos = (y_raw.to_numpy(dtype=np.int8) == 1) & (~pos_has_exit)
            if stats is not None:
                stats["invalid_pos_exit_reason_NONE"] = stats.get("invalid_pos_exit_reason_NONE", 0) + int(invalid_pos.sum())

            y_raw = pd.Series((y_raw.to_numpy(dtype=np.int8) & pos_has_exit).astype("int8"), index=df.index)
        else:
            if stats is not None:
                stats["missing_audit_exit_reason_col"] = stats.get("missing_audit_exit_reason_col", 0) + 1

        # label final = y_go brut (après sanity exit_reason!=NONE)
        y = y_raw

        # normalize features
        df["tf"] = tf_norm
        df = _apply_norm(df, params_map, feats)

        for c in feats:
            if c not in df.columns:
                df[c] = 0.0

        for c in GO_FEATURES_RAW:
            if c not in df.columns:
                df[c] = 0
        feats_out = list(feats) + [c for c in GO_FEATURES_RAW if c not in feats]

        Xf = (
            df[feats_out]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype="float32")
        )
        X = np.hstack([y.to_numpy(dtype="float32").reshape(-1, 1), Xf])
        X = np.nan_to_num(X, copy=False, posinf=0.0, neginf=0.0)

        # ----- AUDIT (aligned) -----
        avail_cols = set(df.columns)

        # A/B/C flags (default 0 if missing)
        A = (pd.to_numeric(df["audit_early_abort"], errors="coerce").fillna(0).astype("int8").clip(0, 1).to_numpy()
             if "audit_early_abort" in avail_cols else np.zeros(len(df), dtype=np.int8))

        B = (pd.to_numeric(df["audit_timeout"], errors="coerce").fillna(0).astype("int8").clip(0, 1).to_numpy()
             if "audit_timeout" in avail_cols else np.zeros(len(df), dtype=np.int8))

        C = (pd.to_numeric(df["audit_market_toxic"], errors="coerce").fillna(0).astype("int8").clip(0, 1).to_numpy()
             if "audit_market_toxic" in avail_cols else np.zeros(len(df), dtype=np.int8))

        # fp_cost_bps: STRICT from audit_pnl_net_bps
        if AUDIT_PNL_COL not in df.columns:
            raise ValueError(
                f"Stage3 audit: colonne manquante '{AUDIT_PNL_COL}'. "
                f"Colonnes dispo (sample): {sorted(list(df.columns))[:50]} ..."
            )

        pnl = pd.to_numeric(df[AUDIT_PNL_COL], errors="coerce").fillna(0.0).astype("float64").to_numpy()
        fp_cost = np.maximum(0.0, -pnl).astype("float32")

        audit2d = np.column_stack([fp_cost, A.astype("float32"), B.astype("float32"), C.astype("float32")])
        
        # ---- split_stats (audit) ----
        if stats is not None:
            # counts
            stats["audit_rows"] = stats.get("audit_rows", 0) + int(len(df))
            stats["audit_A_sum"] = stats.get("audit_A_sum", 0) + int(A.sum())
            stats["audit_B_sum"] = stats.get("audit_B_sum", 0) + int(B.sum())
            stats["audit_C_sum"] = stats.get("audit_C_sum", 0) + int(C.sum())

            # fp_cost running mean (store sum + n, compute mean at end)
            fp_sum = float(fp_cost.sum())  # fp_cost is float32 array
            stats["audit_fp_cost_sum"] = stats.get("audit_fp_cost_sum", 0.0) + fp_sum
            stats["audit_fp_cost_n"] = stats.get("audit_fp_cost_n", 0) + int(fp_cost.size)
            stats["audit_fp_cost_mean"] = (
                stats["audit_fp_cost_sum"] / max(1, stats["audit_fp_cost_n"])
            )

            # keep source (only once)
            stats.setdefault("audit_fp_cost_source", AUDIT_PNL_COL)

        if stats is not None and "audit_fp_cost_source" not in stats:
            stats["audit_fp_cost_source"] = AUDIT_PNL_COL
  
        yield X, ids2d, y, audit2d

# =======================
# CLI
# =======================

def parse_args():
    ap = argparse.ArgumentParser("Stage2_split(go/{train,val,test}) -> Stage3 GO (no internal split)")
    ap.add_argument("--mode", choices=["build", "train"], default="build",
        help="Mode d'exécution. 'train' est un alias rétrocompatible pour 'build' (génère Stage3).",
    )
    ap.add_argument("--src-stage2-split", default="s3://tradebot-config-tokyo/data/stage2/split")
    ap.add_argument("--dst-root", default="s3://tradebot-config-tokyo/data/stage3/go")
    ap.add_argument("--aws-region", default="ap-northeast-1")

    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--tfs", nargs="*", default=None)

    ap.add_argument("--batch-size", type=int, default=200_000)
    ap.add_argument("--rows-per-shard", type=int, default=1_000_000)

    ap.add_argument("--sample-per-tf-feature", type=int, default=200_000,
                    help="Reservoir sample size per (tf, feature) for robust stats (TRAIN only).")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()

def main():
    args = parse_args()
    if getattr(args, "mode", "build") == "train":
        args.mode = "build"
    
    fs = _fs(args.aws_region)

    dset_train = _dataset_from_split_go(fs, args.src_stage2_split, "train")
    dset_val   = _dataset_from_split_go(fs, args.src_stage2_split, "val")
    dset_test  = _dataset_from_split_go(fs, args.src_stage2_split, "test")

    schema = dset_train.schema
    base_f = _optional_filter(schema, args.symbols, args.tfs)

    required = {"tf","y_go","row_id","event_id","task"}
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"Stage2 split/go/train missing required columns: {sorted(missing)}")

    avail = set(schema.names)
    # Normalized features must EXCLUDE raw extras like aggrbt_missing
    feats_norm = [c for c in GO_FEATURES_NORM if c in avail]
    if not feats_norm:
        raise RuntimeError("Aucune feature GO trouvée (GO_FEATURES ne match pas le schema).")

    # Raw extras are appended to X (non-normalized). Only keep those available OR computed (aggrbt_missing is computed).
    feats_raw = list(GO_FEATURES_RAW)
 
    # Output feature order in X (after label col)
    feats_out = list(feats_norm) + [c for c in feats_raw if c not in feats_norm]

    print("Fitting robust scaler params on TRAIN (GO)...")
    rob = _fit_robust_params_train(
        dset_train, base_f, feats_norm,
        batch_size=args.batch_size,
        sample_per_tf_feature=args.sample_per_tf_feature,
        seed=args.seed
    )
    rob["tf"] = rob["tf"].map(_tf_key)

    params_map: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {}
    for tfv, grp in rob.groupby("tf"):
        tfv = _tf_key(tfv)
        params_map[tfv] = {
            str(r.feature): (float(r.q1), float(r.q99), float(r.median), float(r.scale))
            for r in grp.itertuples(index=False)
        }
    
    def write_split(split_name: str, dset: ds.Dataset, posw: float):
        out_x   = f"{args.dst_root.rstrip('/')}/{split_name}"
        out_ids = f"{args.dst_root.rstrip('/')}/{split_name}_ids"
        out_w   = f"{args.dst_root.rstrip('/')}/{split_name}_weight"
        
        out_a = f"{args.dst_root.rstrip('/')}/{split_name}_audit"

        pos_total = 0
        neg_total = 0

        split_stats = {}
        
        def iter_x_ids_w_audit():
            nonlocal pos_total, neg_total
            for X, ids2d, y, audit2d in _iter_x_ids_y_from_split(
                dset, base_f, feats_norm, params_map, batch_size=args.batch_size,
                stats=split_stats
            ):
                # class counts
                pos = int((y == 1).sum())
                neg = int((y == 0).sum())
                pos_total += pos
                neg_total += neg

                # weights aligned with X
                if split_name == "train":
                    w = np.where(y.to_numpy(dtype="int8", copy=False) == 1, float(posw), 1.0).astype("float32")
                    m = float(w.mean()) if np.isfinite(w.mean()) and w.mean() > 0 else 1.0
                    w = (w / m).astype("float32")
                else:
                    w = np.ones(len(y), dtype="float32")

                yield X, ids2d, w, audit2d
        
        nx, nids, nw, na = _write_x_ids_w_audit_shards_stream(
            fs, out_x, out_ids, out_w, out_a,
            iter_x_ids_w_audit(),
            args.rows_per_shard
        )

        print(f"[{split_name}] write stats:", split_stats)

        return nx, nids, nw, pos_total, neg_total, split_stats

    print("Writing TRAIN (pass 1 for class balance) ...")
    # First, compute train balance cheaply (still single scan, but we need posw before writing weights)
    pos_total = 0
    neg_total = 0
    train_stats = {}
    for _, _, y, _ in _iter_x_ids_y_from_split(dset_train, base_f, feats_norm, params_map,
                                                batch_size=args.batch_size,
                                                stats=train_stats):
        pos_total += int((y == 1).sum())
        neg_total += int((y == 0).sum())

    if pos_total == 0:
        print("⛔ TRAIN: pos_total=0 (aucun tradeable) -> stop", file=sys.stderr)
        sys.exit(2)

    posw = float(neg_total / pos_total)
    print(f"TRAIN class balance: pos={pos_total:,} neg={neg_total:,} => scale_pos_weight={posw:.6g}")

    print("Writing TRAIN / VAL / TEST (single-pass per split)...")
    ntrain, _, _, pos_tr, neg_tr, train_write_stats = write_split("train", dset_train, posw)
    nval,   _, _, pos_va, neg_va, val_stats         = write_split("val",   dset_val,   1.0)
    ntest,  _, _, pos_te, neg_te, test_stats        = write_split("test",  dset_test,  1.0)

    if ntrain == 0:
        print("⛔ TRAIN vide -> stop", file=sys.stderr)
        sys.exit(2)

    meta_root = f"{args.dst_root.rstrip('/')}/{META_DIR}"

    cutoffs_payload = None
    try:
        cutoffs_path = f"{args.src_stage2_split.rstrip('/')}/_meta/split_cutoffs.json"
        with fs.open_input_file(_strip_s3(cutoffs_path)) as f:
            cutoffs_payload = json.loads(f.read().decode("utf-8"))
    except Exception:
        cutoffs_payload = None

    with _open_s3_output(fs, f"{meta_root}/columns.json") as f:
        f.write(json.dumps({
            "task": "go",
            "label": "is_tradeable",
            "label_source": "y_go (stage1) with sanity gate: positives require audit_exit_reason != 'NONE'",
            "features": feats_out,
            "row_format": "CSV, first column is is_tradeable then features in listed order",
            "ids_format": "CSV, columns: row_id,event_id (aligned with X shards, same part numbers)",
            "label_rule": "GO: is_tradeable = y_go after sanity gate: if y_go==1 then audit_exit_reason!='NONE' else unchanged.",
            "feature_notes": {
                "aggrbt_missing": "Raw (non-normalized) flag: 1 if any aggr/bt col was NaN and imputed on the row, else 0."
            },
            "splits_source": "stage2/split/{train,val,test}/go",
            "ids_paths": {
                "train_ids": "train_ids/part-*.csv",
                "val_ids": "val_ids/part-*.csv",
                "test_ids": "test_ids/part-*.csv"
            },
            "audit_format": "CSV, no header, columns: fp_cost_bps,audit_early_abort,audit_timeout,audit_market_toxic (aligned with X shards)",
            "audit_paths": {
                "train_audit": "train_audit/part-*.csv",
                "val_audit": "val_audit/part-*.csv",
                "test_audit": "test_audit/part-*.csv"
            },
        }, indent=2).encode("utf-8"))

    with _open_s3_output(fs, f"{meta_root}/scaler_stats.json") as f:
        f.write(rob.to_json(orient="records").encode("utf-8"))

    with _open_s3_output(fs, f"{meta_root}/train_class_balance.json") as f:
        f.write(json.dumps({
            "pos": int(pos_tr),
            "neg": int(neg_tr),
            "scale_pos_weight_raw": float(posw),
            "note": "Weights saved in train_weight shards are normalized to mean=1.",
        }, indent=2).encode("utf-8"))

    with _open_s3_output(fs, f"{meta_root}/write_stats.json") as f:
        f.write(json.dumps({
            "train_pass1_balance_scan": train_stats,
            "train_write": train_write_stats,
            "val_write": val_stats,
            "test_write": test_stats,
        }, indent=2).encode("utf-8"))

    if cutoffs_payload is not None:
        with _open_s3_output(fs, f"{meta_root}/time_split_cutoffs.json") as f:
            f.write(json.dumps(cutoffs_payload, indent=2).encode("utf-8"))

    print("Done.")

if __name__ == "__main__":
    main()