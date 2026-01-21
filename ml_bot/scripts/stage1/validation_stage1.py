#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validation_stage1.py — Validate + summarize Stage1 labels (RR + GO + DIR), chunked & S3-safe.

Supports Stage1 outputs that may contain:
  - RR labels:   y_<tf> in {-1,0,1} + optional pnl_net_bps_<tf>, exit_reason_<tf>,
                tp_bps_<tf>, sl_bps_<tf>, rr_min_<tf>, risk_r_bps_<tf>, entry_spread_half_bps
  - GO labels:   y_go_<tf> in {0,1}
  - DIR labels:  y_dir_<tf> in {-1,0,1} (0 usually means NO-GO)

Cross-consistency checks (when both exist, per TF horizon):
  - y_go_<tf> should match:
        if exit_reason exists: (exit_reason contains "TP_")
        else:                  (y_<tf> != 0)   [legacy TP-only RR convention]
  - y_dir_<tf> should match y_<tf> (and 0 when y_<tf> == 0)

RR-specific checks (when columns exist):
  - y vs exit_reason consistency (TP_LONG/TP_SHORT vs y=+1/-1 ; y=0 allows TIME/NOFILL/SL/NONE)
  - tp_bps ≈ rr_min * risk_r_bps ; sl_bps ≈ risk_r_bps
  - entry_spread_half_bps ≈ 0.5 * spread_bps_entry
  - pnl_net_bps:
      * exit_reason SL_* => pnl < 0
      * exit_reason TP_* => pnl > 0
      * exit_reason TIME/NOFILL/NONE => pnl ≈ 0 (optional)

Outputs:
  metrics_overall.csv
  metrics_by_label.csv
  monthly_by_label.csv
  integrity_issues.csv
  recommendations.txt

Layout:
  <root>/<SYMBOL>/<YEAR>_signals.csv
root can be local path or s3://bucket/prefix
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import fsspec


# ============================
# Logging
# ============================

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("stage1_validation")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


# ============================
# Path helpers (local/S3)
# ============================

def is_s3(path: str) -> bool:
    return str(path).startswith("s3://")

def s3_storage_options(region: Optional[str]) -> dict:
    return {"client_kwargs": {"region_name": region}} if region else {}

def normalize_root(root: str) -> str:
    root = str(root).strip()
    if is_s3(root):
        return root.rstrip("/")
    return os.path.abspath(root)

def join_path(root: str, *parts: str) -> str:
    if is_s3(root):
        return "/".join([root.rstrip("/")] + [p.strip("/").replace("\\", "/") for p in parts])
    return os.path.join(root, *parts)

def list_signal_files(root: str, so: dict) -> List[str]:
    if is_s3(root):
        fs = fsspec.filesystem("s3", **so)
        rest = root[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        base = f"{bucket}/{prefix}".rstrip("/") if prefix else bucket
        matches = fs.glob(f"{base}/*/[0-9][0-9][0-9][0-9]_signals.csv")
        return ["s3://" + m for m in sorted(matches)]
    else:
        out = []
        for sym in os.listdir(root):
            p_sym = os.path.join(root, sym)
            if not os.path.isdir(p_sym):
                continue
            for fn in os.listdir(p_sym):
                if fn.endswith("_signals.csv") and len(fn) >= len("YYYY_signals.csv"):
                    y = fn.split("_", 1)[0]
                    if y.isdigit() and len(y) == 4:
                        out.append(os.path.join(p_sym, fn))
        return sorted(out)

def open_any(path: str, so: dict):
    if is_s3(path):
        return fsspec.open(
            path, "rb",
            block_size=64 * 1024 * 1024,
            cache_type="none",
            **so
        ).open()
    return open(path, "rb")

def write_text(path: str, text: str, so: dict):
    if is_s3(path):
        with fsspec.open(path, "wb", **so) as f:
            f.write(text.encode("utf-8"))
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

def write_df_csv(path: str, df: pd.DataFrame, so: dict):
    if is_s3(path):
        with fsspec.open(path, "wb", **so) as f:
            df.to_csv(f, index=False)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

# ============================
# Basic utils
# ============================

def to_utc_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def month_key(dt: pd.Series) -> pd.Series:
    return dt.dt.strftime("%Y-%m")

def safe_rate(k: int, n: int) -> float:
    return float(k / n) if n > 0 else float("nan")

def _tf_sort_key(tf: str) -> int:
    tf = str(tf).strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    return 10**9


# ============================
# Label discovery (family-aware)
# ============================

@dataclass(frozen=True)
class LabelSpec:
    col: str          # e.g. y_15m, y_go_15m, y_dir_15m
    family: str       # rr | go | dir
    horizon: str      # 15m, 1h, ...
    key: str          # stable id: f"{family}:{horizon}"

def discover_label_specs(cols: Iterable[str]) -> List[LabelSpec]:
    cols = list(cols)
    out: List[LabelSpec] = []
    for c in cols:
        if not c.startswith("y_"):
            continue
        tail = c[2:]  # after "y_"
        if tail.startswith("go_"):
            h = tail[len("go_"):]
            out.append(LabelSpec(c, "go", h, f"go:{h}"))
        elif tail.startswith("dir_"):
            h = tail[len("dir_"):]
            out.append(LabelSpec(c, "dir", h, f"dir:{h}"))
        else:
            out.append(LabelSpec(c, "rr", tail, f"rr:{tail}"))

    fam_ord = {"rr": 0, "go": 1, "dir": 2}
    out.sort(key=lambda s: (fam_ord.get(s.family, 9), _tf_sort_key(s.horizon), s.col))
    return out


# ============================
# Reservoir sampling (approx quantiles)
# ============================

@dataclass
class Reservoir:
    max_n: int = 200_000
    n_seen: int = 0
    data: List[float] = field(default_factory=list)

    def add_array(self, arr: np.ndarray, rng: np.random.Generator):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        for v in arr:
            self.n_seen += 1
            if len(self.data) < self.max_n:
                self.data.append(float(v))
            else:
                j = rng.integers(0, self.n_seen)
                if j < self.max_n:
                    self.data[int(j)] = float(v)

    def quantiles(self, qs=(0.5, 0.75, 0.9)) -> Dict[str, float]:
        if not self.data:
            return {f"q{int(q*100)}": float("nan") for q in qs}
        a = np.asarray(self.data, dtype=np.float64)
        return {f"q{int(q*100)}": float(np.quantile(a, q)) for q in qs}


# ============================
# Integrity issues
# ============================

@dataclass
class IntegrityIssue:
    path: str
    symbol: str
    year: int
    issue: str
    details: str


# ============================
# Aggregators
# ============================

@dataclass
class AggLabel:
    n: int = 0
    n_pos: int = 0         # GO: y==1 ; RR/DIR: y==+1
    n_neg: int = 0         # RR/DIR: y==-1
    n_zero: int = 0        # y==0

    pnl_trade: Reservoir = field(default_factory=lambda: Reservoir(120_000))

    # RR-only exit reasons counts
    n_tp_long: int = 0
    n_tp_short: int = 0
    n_sl_long: int = 0
    n_sl_short: int = 0
    n_time: int = 0
    n_nofill: int = 0
    n_none: int = 0

    # RR-only checks
    n_bad_y_exit_reason: int = 0
    n_bad_rr_formula: int = 0
    n_bad_entry_spread_half: int = 0
    n_bad_pnl_sl_sign: int = 0
    n_bad_pnl_tp_sign: int = 0

    # EV threshold checks (p_thr_ev0_<tf>)
    n_bad_p_thr_ev0_range: int = 0
    n_bad_p_thr_ev0_nan: int = 0
    p_thr_ev0_samples: Reservoir = field(default_factory=lambda: Reservoir(120_000))
    n_bad_p_thr_ev0_formula: int = 0

@dataclass
class AggAudit:
    n_checked: int = 0
    n_bad: int = 0
    n_bad_range: int = 0
    n_bad_monotonic: int = 0
    n_bad_first_touch: int = 0
    n_bad_first_touch_step: int = 0
    n_bad_mfe_mae: int = 0

@dataclass
class AggFile:
    n_dup_row_id: int = 0
    n_rows: int = 0
    n_bad_ts: int = 0
    n_dup_ts: int = 0
    n_nonmonotonic_chunks: int = 0

    # base range checks
    n_bad_spread_lt0: int = 0
    n_bad_atr_lt0: int = 0
    n_bad_entry_px_le0: int = 0

    step_seconds_samples: Reservoir = field(default_factory=lambda: Reservoir(50_000))
    by_label: Dict[str, AggLabel] = field(default_factory=dict)  # key = family:horizon

    # cross-consistency counts
    n_bad_go_vs_rr: int = 0
    n_bad_dir_vs_rr: int = 0
    n_checked_go_vs_rr: int = 0
    n_checked_dir_vs_rr: int = 0

     # spread diagnostics
    spread_min: float = float("inf")
    spread_min_count: int = 0  # combien de lignes == min (approx, sur les chunks)
    n_bad_spread_recalc_mismatch: int = 0
    n_checked_spread_recalc: int = 0

    # monthly counts: (key, month, y) -> count
    monthly: Dict[Tuple[str, str, int], int] = field(default_factory=dict)

    # Stage1 v3 audit stats (optional)
    audit: AggAudit = field(default_factory=AggAudit)

# ============================
# Helpers: robust exit_reason matching
# ============================

# ============================
# Stage1 v3 AUDIT checks (optional)
# ============================

AUDIT_POS_LEVELS = (1, 2, 3, 4, 5)
AUDIT_NEG_LEVELS = (1, 2, 3)  # -1R/-2R/-3R

AUDIT_FIRST_TOUCH_ALLOWED = set([-3, -2, -1, 0, 1, 2, 3, 4, 5])

def _audit_col(prefix: str, name: str, h: str) -> str:
    # prefix: "auditL" or "auditS"
    return f"{prefix}_{name}_{h}"

def _audit_hit_col(prefix: str, sign: str, k: int, h: str) -> str:
    # sign: "p" for +kR, "m" for -kR
    return f"{prefix}_hit_{sign}{k}R_{h}"

def _norm_reason(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip().upper()

def _reason_ok(y: int, reason: str) -> bool:
    r = _norm_reason(reason)
    if y == 1:
        return ("TP_LONG" in r) or (r == "TPLONG")
    if y == -1:
        return ("TP_SHORT" in r) or (r == "TPSHORT")
    if y == 0:
        if r == "" or r in {"NA", "NAN", "NONE"}:
            return True
        return ("TIME" in r) or ("NOFILL" in r) or ("SL_" in r)  # allow SL even if y is TP-only
    return False


# ============================
# Core per-file processing (chunked)
# ============================

def infer_symbol_year_from_path(path: str) -> Tuple[str, int]:
    symbol = os.path.basename(os.path.dirname(path.rstrip("/")))
    base = os.path.basename(path)
    year = -1
    try:
        year = int(base.split("_", 1)[0])
    except Exception:
        pass
    return symbol, year

def process_signals_file(
    path: str,
    so: dict,
    chunksize: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Tuple[str, int, AggFile, List[IntegrityIssue]]:
    issues: List[IntegrityIssue] = []
    agg = AggFile()

    symbol, year = infer_symbol_year_from_path(path)
    logger.info(f"READ {path} (symbol={symbol}, year={year})")

    # Peek header
    t_peek0 = time.perf_counter()
    logger.info(f"PEEK header start: {path}")
    try:
        if is_s3(path):
            head = pd.read_csv(path, nrows=0, storage_options=so)
        else:
            with open_any(path, so) as f:
                head = pd.read_csv(f, nrows=0)
    except Exception as e:
        issues.append(IntegrityIssue(path, symbol, year, "read_header_failed", f"{type(e).__name__}: {e}"))
        logger.exception(f"FAILED header read: {path}")
        return symbol, year, agg, issues
    logger.info(f"PEEK header done in {time.perf_counter()-t_peek0:.2f}s cols={len(head.columns)}")
    cols = list(head.columns)

    # Must-have base cols (minimal)
    base_required = ["row_id", "t", "spread_bps_entry", "atr_bps"]
    miss = [c for c in base_required if c not in cols]
    if miss:
        issues.append(IntegrityIssue(path, symbol, year, "missing_base_cols", ",".join(miss)))
        return symbol, year, agg, issues

    has_bidask = ("bid_entry" in cols and "ask_entry" in cols)
    has_mid = ("mid_entry" in cols)
    if not has_bidask and not has_mid:
        issues.append(IntegrityIssue(path, symbol, year, "missing_entry_price_cols",
                                     "need bid_entry+ask_entry or mid_entry"))
        return symbol, year, agg, issues

    # Discover labels
    specs = discover_label_specs(cols)
    if not specs:
        issues.append(IntegrityIssue(path, symbol, year, "no_label_cols", "No y_* columns found"))
        return symbol, year, agg, issues
    for s in specs:
        agg.by_label.setdefault(s.key, AggLabel())

    # cross-check availability maps (by horizon)
    rr_col_by_h = {s.horizon: s.col for s in specs if s.family == "rr"}
    go_col_by_h = {s.horizon: s.col for s in specs if s.family == "go"}
    dir_col_by_h = {s.horizon: s.col for s in specs if s.family == "dir"}

    use_global_dup_set = False
    seen_ts = set()
    seen_rid = set()

    # ---------------------------
    # Chunked read (visible timing + S3-safe)
    # ---------------------------
    logger.info(f"CHUNK read start: chunksize={chunksize} path={path}")
    t_first_chunk0 = time.perf_counter()
    
    f = None 
    try:
        # IMPORTANT: pour S3, pandas lit mieux via le path + storage_options
        if is_s3(path):
            reader = pd.read_csv(path, chunksize=chunksize, storage_options=so)
            f = None
        else:
            f = open_any(path, so)
            reader = pd.read_csv(f, chunksize=chunksize)
    except Exception as e:
        issues.append(IntegrityIssue(path, symbol, year, "read_chunked_init_failed", f"{type(e).__name__}: {e}"))
        logger.exception(f"FAILED init chunked read: {path}")
        return symbol, year, agg, issues

    rid_to_t = {}  # row_id -> first seen t (string)

    got_first = False
    last_beat = time.perf_counter()

    try:
        for ci, df in enumerate(reader, 1):
            # log au 1er chunk pour savoir si ça bloque avant
            if not got_first:
                logger.info(f"FIRST chunk received in {time.perf_counter()-t_first_chunk0:.2f}s rows={len(df):,}")
                got_first = True

            # heartbeat toutes les 15s (évite le "silence")
            now = time.perf_counter()
            if now - last_beat > 15:
                logger.info(f"HEARTBEAT: chunk={ci} rows_so_far={agg.n_rows:,}")
                last_beat = now

            if df is None or df.empty:
                continue

            agg.n_rows += int(len(df))

            # Timestamp parse
            dt = to_utc_datetime(df["t"])
            bad_ts = int(dt.isna().sum())
            if bad_ts:
                agg.n_bad_ts += bad_ts
                issues.append(IntegrityIssue(path, symbol, year, "bad_timestamp_parse",
                                             f"bad_ts={bad_ts} in chunk={ci}"))

            good = dt.notna()
            df = df.loc[good].copy()
            dt = dt.loc[good]
            if df.empty:
                continue

            # row_id checks (primary key)
            rid = df["row_id"]

            # 1) NaN / empty row_id => issue (grave)
            rid_str = rid.astype("string")  # garde <NA>
            bad_rid = int(rid_str.isna().sum() + (rid_str.str.len().fillna(0) == 0).sum())
            if bad_rid:
                issues.append(IntegrityIssue(path, symbol, year, "bad_row_id",
                                            f"bad_row_id={bad_rid} in chunk={ci}"))

            # 2) duplicates inside chunk
            dup_rid_in_chunk = int(rid_str.duplicated().sum())
            if dup_rid_in_chunk:
                agg.n_dup_row_id += dup_rid_in_chunk
                issues.append(IntegrityIssue(path, symbol, year, "duplicate_row_id_in_chunk",
                                            f"count={dup_rid_in_chunk} chunk={ci}"))

            # 3) duplicates across chunks (always check, don't gate on a flag)
            #    count duplicates of already-seen ids
            for vv in rid_str.dropna().astype(str).values:
                if vv in seen_rid:
                    agg.n_dup_row_id += 1
                else:
                    seen_rid.add(vv)

            # row_id -> t should be 1-1 (detect collision across chunks)
            t_str = df["t"].astype("string")

            # chunk-level quick detection (optional but cheap)
            tmp = pd.DataFrame({"row_id": rid_str, "t": t_str})
            bad_map = tmp.dropna().groupby("row_id")["t"].nunique()
            bad_map = bad_map[bad_map > 1]
            if not bad_map.empty:
                issues.append(IntegrityIssue(path, symbol, year, "row_id_maps_to_multiple_t",
                                            f"n={len(bad_map)} in chunk={ci}"))

            # cross-chunk detection (source of truth)
            for ridv, tv in zip(rid_str.to_list(), t_str.to_list()):
                if ridv is None or pd.isna(ridv) or tv is None or pd.isna(tv):
                    continue
                rkey = str(ridv)
                tkey = str(tv)
                prev = rid_to_t.get(rkey)
                if prev is None:
                    rid_to_t[rkey] = tkey
                elif prev != tkey:
                    issues.append(IntegrityIssue(path, symbol, year, "row_id_maps_to_multiple_t_cross_chunk",
                                                f"row_id={rkey} prev_t={prev} new_t={tkey}"))
                                
            # dup/monotonic checks
            dup_in_chunk = int(dt.duplicated().sum())
            if dup_in_chunk:
                agg.n_dup_ts += dup_in_chunk
                use_global_dup_set = True

            if use_global_dup_set:
                dts = dt.astype("int64").to_numpy()
                for v in dts:
                    vv = int(v)
                    if vv in seen_ts:
                        agg.n_dup_ts += 1
                    else:
                        seen_ts.add(vv)

            if not dt.is_monotonic_increasing:
                agg.n_nonmonotonic_chunks += 1
                dt_sorted = dt.sort_values()
            else:
                dt_sorted = dt

            # step samples
            diffs = dt_sorted.diff().dropna().dt.total_seconds().to_numpy(dtype=np.float64)
            if diffs.size:
                diffs = diffs[(diffs > 0) & (diffs <= 3600)]
                if diffs.size:
                    agg.step_seconds_samples.add_array(diffs, rng)

            # Base numeric sanity
            if has_bidask:
                bid = pd.to_numeric(df["bid_entry"], errors="coerce")
                ask = pd.to_numeric(df["ask_entry"], errors="coerce")
                bad_px = int((bid <= 0).sum(skipna=True) + (ask <= 0).sum(skipna=True))
                agg.n_bad_entry_px_le0 += bad_px
            else:
                mid = pd.to_numeric(df["mid_entry"], errors="coerce")
                agg.n_bad_entry_px_le0 += int((mid <= 0).sum(skipna=True))

            spr = pd.to_numeric(df["spread_bps_entry"], errors="coerce")
            atr = pd.to_numeric(df["atr_bps"], errors="coerce")
            agg.n_bad_spread_lt0 += int((spr < 0).sum(skipna=True))
            agg.n_bad_atr_lt0 += int((atr < 0).sum(skipna=True))

            # --- spread_bps_entry: min + sanity check ---
            sprv = spr.to_numpy(dtype=np.float64, copy=False)
            ok_spr = np.isfinite(sprv)
            if ok_spr.any():
                smin = float(np.min(sprv[ok_spr]))
                if smin < agg.spread_min:
                    agg.spread_min = smin
                    # approx count in this chunk for that min
                    agg.spread_min_count = int(np.sum(sprv[ok_spr] == smin))
                elif smin == agg.spread_min:
                    agg.spread_min_count += int(np.sum(sprv[ok_spr] == smin))

            # --- validate spread_bps_entry against bid/ask entry (best validation of your Stage1 fix)
            if has_bidask:
                bid = pd.to_numeric(df["bid_entry"], errors="coerce").to_numpy(dtype=np.float64)
                ask = pd.to_numeric(df["ask_entry"], errors="coerce").to_numpy(dtype=np.float64)
                mid = 0.5 * (bid + ask)

                spr_recalc = np.where(np.isfinite(mid) & (mid > 0),
                                      1e4 * (ask - bid) / mid,
                                      np.nan)

                ok = np.isfinite(spr_recalc) & np.isfinite(sprv)
                if ok.any():
                    agg.n_checked_spread_recalc += int(ok.sum())

                    # tol permissive pour float32 + arrondis + CSV
                    diff = np.abs(sprv[ok] - spr_recalc[ok])
                    tol = 1e-3 + 1e-3 * np.abs(spr_recalc[ok])

                    agg.n_bad_spread_recalc_mismatch += int((diff > tol).sum())

            # ------------------------------------------------------------
            # Cross-consistency checks (if rr exists)
            # ------------------------------------------------------------
            for h, rrcol in rr_col_by_h.items():
                if rrcol not in df.columns:
                    continue
                y_rr = pd.to_numeric(df[rrcol], errors="coerce").fillna(0).astype("int8").to_numpy()

                gocol = go_col_by_h.get(h)
                rsn_col_h = f"exit_reason_{h}"
                if gocol and gocol in df.columns:
                    y_go = pd.to_numeric(df[gocol], errors="coerce").fillna(0).astype("int8").to_numpy()

                    if rsn_col_h in df.columns:
                        rsn = df[rsn_col_h].map(_norm_reason).to_numpy(dtype=object)
                        rs = np.asarray(rsn, dtype=str)
                        want = (np.char.find(rs, "TP_") >= 0).astype(np.int8)
                    else:
                        want = (y_rr != 0).astype(np.int8)

                    agg.n_checked_go_vs_rr += int(y_go.size)
                    agg.n_bad_go_vs_rr += int((y_go != want).sum())

                dcol = dir_col_by_h.get(h)
                if dcol and dcol in df.columns:
                    y_dir = pd.to_numeric(df[dcol], errors="coerce").fillna(0).astype("int8").to_numpy()
                    want = np.where(y_rr != 0, y_rr, 0).astype(np.int8)
                    agg.n_checked_dir_vs_rr += int(y_dir.size)
                    agg.n_bad_dir_vs_rr += int((y_dir != want).sum())

            # ------------------------------------------------------------
            # Per-label stats + monthly counts
            # ------------------------------------------------------------
            mkey = month_key(dt)

            for s in specs:
                if s.col not in df.columns:
                    continue

                y = pd.to_numeric(df[s.col], errors="coerce").fillna(0).astype("int8").to_numpy()
                al = agg.by_label[s.key]
                al.n += int(y.size)

                if s.family == "go":
                    al.n_pos += int((y == 1).sum())
                    al.n_zero += int((y == 0).sum())
                else:
                    al.n_pos += int((y == 1).sum())
                    al.n_neg += int((y == -1).sum())
                    al.n_zero += int((y == 0).sum())

                # Monthly counts per label
                g = pd.DataFrame({"m": mkey, "y": y})
                grp = g.groupby(["m", "y"]).size()
                for (mm, yy), cnt in grp.items():
                    k = (s.key, str(mm), int(yy))
                    agg.monthly[k] = agg.monthly.get(k, 0) + int(cnt)

                # RR-only extras
                if s.family == "rr":
                    rsn_col = f"exit_reason_{s.horizon}"
                    pnl_col = f"pnl_net_bps_{s.horizon}"

                    # exit_reason consistency + reason counts
                    if rsn_col in df.columns:
                        rsn = df[rsn_col].map(_norm_reason).to_numpy(dtype=object)
                        bad = 0
                        for yi, ri in zip(y, rsn):
                            if not _reason_ok(int(yi), str(ri)):
                                bad += 1
                        al.n_bad_y_exit_reason += int(bad)

                        rs = np.asarray(rsn, dtype=str)
                        al.n_tp_long  += int((np.char.find(rs, "TP_LONG")  >= 0).sum())
                        al.n_tp_short += int((np.char.find(rs, "TP_SHORT") >= 0).sum())
                        al.n_sl_long  += int((np.char.find(rs, "SL_LONG")  >= 0).sum())
                        al.n_sl_short += int((np.char.find(rs, "SL_SHORT") >= 0).sum())
                        al.n_time     += int((np.char.find(rs, "TIME")     >= 0).sum())
                        al.n_nofill   += int((np.char.find(rs, "NOFILL")   >= 0).sum())
                        al.n_none     += int(((rs == "") | (rs == "NONE") | (rs == "NA") | (rs == "NAN")).sum())

                    # entry_spread_half_bps ~ 0.5 * spread_bps_entry
                    if "entry_spread_half_bps" in df.columns:
                        half = pd.to_numeric(df["entry_spread_half_bps"], errors="coerce").to_numpy(dtype=np.float64)
                        sprv = pd.to_numeric(df["spread_bps_entry"], errors="coerce").to_numpy(dtype=np.float64)
                        ok = np.isfinite(half) & np.isfinite(sprv)
                        if ok.any():
                            diff = np.abs(half[ok] - 0.5 * sprv[ok])
                            tol = 1e-3 + 5e-4 * np.abs(sprv[ok])  # permissive
                            al.n_bad_entry_spread_half += int((diff > tol).sum())

                    # RR formula checks: tp_bps ≈ rr_min * risk_r_bps ; sl_bps ≈ risk_r_bps
                    tp_col = f"tp_bps_{s.horizon}"
                    sl_col = f"sl_bps_{s.horizon}"
                    rr_col = f"rr_min_{s.horizon}"
                    r_col  = f"risk_r_bps_{s.horizon}"
                    if all(c in df.columns for c in (tp_col, sl_col, rr_col, r_col)):
                        tp = pd.to_numeric(df[tp_col], errors="coerce").to_numpy(dtype=np.float64)
                        sl = pd.to_numeric(df[sl_col], errors="coerce").to_numpy(dtype=np.float64)
                        rr = pd.to_numeric(df[rr_col], errors="coerce").to_numpy(dtype=np.float64)
                        rb = pd.to_numeric(df[r_col],  errors="coerce").to_numpy(dtype=np.float64)
                        ok = np.isfinite(tp) & np.isfinite(sl) & np.isfinite(rr) & np.isfinite(rb)
                        if ok.any():
                            want_tp = rr[ok] * rb[ok]
                            want_sl = rb[ok]
                            tp_bad = np.abs(tp[ok] - want_tp) > (1e-2 + 2e-3 * np.abs(want_tp))
                            sl_bad = np.abs(sl[ok] - want_sl) > (1e-2 + 2e-3 * np.abs(want_sl))
                            al.n_bad_rr_formula += int(tp_bad.sum() + sl_bad.sum())
                    
                    # ---- p_thr_ev0 check (EV=0 threshold) ----
                    pcol = f"p_thr_ev0_{s.horizon}"
                    if pcol in df.columns:
                        p = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=np.float64)
                        okp = np.isfinite(p)

                        al.n_bad_p_thr_ev0_nan += int((~okp).sum())

                        # range [0,1] with tiny tolerance (CSV/float32)
                        if okp.any():
                            bad_range = (p[okp] < -1e-6) | (p[okp] > 1.0 + 1e-6)
                            al.n_bad_p_thr_ev0_range += int(bad_range.sum())

                            # sample for quantiles
                            al.pnl_trade  # keep existing
                            al.p_thr_ev0_samples.add_array(p[okp], rng)

                        # coherence check vs formula if inputs exist:
                        # p_thr_ev0 = (SL + cost) / (TP + SL)
                        tp_col = f"tp_bps_{s.horizon}"
                        sl_col = f"sl_bps_{s.horizon}"
                        if ("entry_spread_half_bps" in df.columns) and (tp_col in df.columns) and (sl_col in df.columns):
                            tp = pd.to_numeric(df[tp_col], errors="coerce").to_numpy(dtype=np.float64)
                            sl = pd.to_numeric(df[sl_col], errors="coerce").to_numpy(dtype=np.float64)
                            half = pd.to_numeric(df["entry_spread_half_bps"], errors="coerce").to_numpy(dtype=np.float64)

                            # fee_exit_bps is constant, but stage1 writes it; use it if present
                            if "fee_exit_bps" in df.columns:
                                fee = pd.to_numeric(df["fee_exit_bps"], errors="coerce").to_numpy(dtype=np.float64)
                            else:
                                fee = np.full(len(df), 0.0, dtype=np.float64)

                            cost = half + fee
                            denom = np.maximum(tp + sl, 1e-12)

                            want = (sl + cost) / denom

                            ok = np.isfinite(p) & np.isfinite(want)
                            if ok.any():
                                diff = np.abs(p[ok] - want[ok])
                                # permissif: float32 + csv
                                tol = 5e-4 + 5e-4 * np.abs(want[ok])
                                # on recycle n_bad_rr_formula ou on crée un compteur dédié (mieux)
                                # => ici: on compte comme "bad_p_thr_ev0_range" ? non.
                                # Ajoute plutôt un compteur dédié si tu veux.
                                al.n_bad_p_thr_ev0_formula += int((diff > tol).sum())

                    # pnl checks (new convention):
                    # - TP_* => pnl > 0
                    # - SL_* => pnl < 0
                    # - TIME/NOFILL/NONE => pnl ~= 0 (optional)
                    if pnl_col in df.columns:
                        pnl = pd.to_numeric(df[pnl_col], errors="coerce").to_numpy(dtype=np.float64)
                        okp = np.isfinite(pnl)

                        if rsn_col in df.columns and okp.any():
                            rsn = df[rsn_col].map(_norm_reason).to_numpy(dtype=object)
                            rs = np.asarray(rsn, dtype=str)

                            is_tp = (np.char.find(rs, "TP_") >= 0)
                            is_sl = (np.char.find(rs, "SL_") >= 0)
                            is_flat = (
                                (np.char.find(rs, "TIME") >= 0)
                                | (np.char.find(rs, "NOFILL") >= 0)
                                | (rs == "NONE")
                                | (rs == "")
                            )

                            # Sign checks
                            al.n_bad_pnl_sl_sign += int(((is_sl & okp) & (pnl >= -1e-6)).sum())
                            al.n_bad_pnl_tp_sign += int(((is_tp & okp) & (pnl <=  1e-6)).sum())

                            # Optional flat check (not stored)
                            # flat_bad = int(((is_flat & okp) & (np.abs(pnl) > 1e-6)).sum())

                            # Sample pnl on real trades (TP or SL)
                            mask_trade = (is_tp | is_sl) & okp
                            if mask_trade.any():
                                al.pnl_trade.add_array(pnl[mask_trade], rng)

                        else:
                            # No exit_reason => can't validate sign; still sample non-zero pnl
                            mask_trade = (okp) & (np.abs(pnl) > 1e-12)
                            if mask_trade.any():
                                al.pnl_trade.add_array(pnl[mask_trade], rng)

                # ------------------------------------------------------------
                # Stage1 v3 AUDIT checks (optional) — only if columns exist
                # ------------------------------------------------------------
                # We validate, per horizon, both sides L/S:
                # - audit*_mfe_R >= 0 ; audit*_mae_R >= 0
                # - audit*_first_touch in allowed set; step logic:
                #       first_touch == 0  => step == -1
                #       first_touch != 0  => step >= 0
                # - hit flags are 0/1
                # - monotonicity: hit_p3 => hit_p2 => hit_p1 ; hit_m3 => hit_m2 => hit_m1
                # - coherence with first_touch:
                #       first_touch == +k => hit_p{k} == 1
                #       first_touch == -k => hit_m{k} == 1
                #
                # If some cols are missing, we just skip that check.

                # horizons present among RR labels are the natural loop driver
                for h in rr_col_by_h.keys():
                    for side_prefix in ("auditL", "auditS"):
                        mfe_col  = _audit_col(side_prefix, "mfe_R", h)
                        mae_col  = _audit_col(side_prefix, "mae_R", h)
                        ft_col   = _audit_col(side_prefix, "first_touch", h)
                        fts_col  = _audit_col(side_prefix, "first_touch_step", h)

                        # If none of the core audit cols exist, skip (fast)
                        if (mfe_col not in df.columns) and (mae_col not in df.columns) and (ft_col not in df.columns):
                            continue

                        A = agg.audit  # shorthand

                        # --- MFE/MAE range checks
                        if mfe_col in df.columns:
                            mfe = pd.to_numeric(df[mfe_col], errors="coerce").to_numpy(dtype=np.float64)
                            ok = np.isfinite(mfe)
                            if ok.any():
                                A.n_checked += int(ok.sum())
                                bad = (mfe[ok] < -1e-9)
                                A.n_bad_mfe_mae += int(bad.sum())
                                A.n_bad += int(bad.sum())

                        if mae_col in df.columns:
                            mae = pd.to_numeric(df[mae_col], errors="coerce").to_numpy(dtype=np.float64)
                            ok = np.isfinite(mae)
                            if ok.any():
                                A.n_checked += int(ok.sum())
                                bad = (mae[ok] < -1e-9)
                                A.n_bad_mfe_mae += int(bad.sum())
                                A.n_bad += int(bad.sum())

                        # --- first_touch domain + step rules
                        if ft_col in df.columns:
                            ft = pd.to_numeric(df[ft_col], errors="coerce").to_numpy(dtype=np.float64)
                            ok = np.isfinite(ft)
                            if ok.any():
                                ft_i = ft[ok].astype(np.int16)
                                A.n_checked += int(ok.sum())

                                bad_dom = np.array([int(v) not in AUDIT_FIRST_TOUCH_ALLOWED for v in ft_i], dtype=bool)
                                if bad_dom.any():
                                    A.n_bad_first_touch += int(bad_dom.sum())
                                    A.n_bad += int(bad_dom.sum())

                                if fts_col in df.columns:
                                    fts = pd.to_numeric(df[fts_col], errors="coerce").to_numpy(dtype=np.float64)
                                    ok2 = ok & np.isfinite(fts)
                                    if ok2.any():
                                        ft2 = ft[ok2].astype(np.int16)
                                        st2 = fts[ok2].astype(np.int64)

                                        # ft==0 => step must be -1
                                        bad0 = (ft2 == 0) & (st2 != -1)
                                        # ft!=0 => step must be >=0
                                        bad1 = (ft2 != 0) & (st2 < 0)

                                        b = int(bad0.sum() + bad1.sum())
                                        if b:
                                            A.n_bad_first_touch_step += b
                                            A.n_bad += b

                        # --- hit flags checks (0/1) + monotonicity
                        # Collect present hit cols
                        hit_p_cols = []
                        hit_m_cols = []
                        for k in AUDIT_POS_LEVELS:
                            c = _audit_hit_col(side_prefix, "p", k, h)
                            if c in df.columns:
                                hit_p_cols.append((k, c))
                        for k in AUDIT_NEG_LEVELS:
                            c = _audit_hit_col(side_prefix, "m", k, h)
                            if c in df.columns:
                                hit_m_cols.append((k, c))

                        # 0/1 check on present columns
                        for k, c in hit_p_cols + hit_m_cols:
                            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
                            ok = np.isfinite(v)
                            if ok.any():
                                A.n_checked += int(ok.sum())
                                bad = ~((v[ok] == 0.0) | (v[ok] == 1.0))
                                if bad.any():
                                    b = int(bad.sum())
                                    A.n_bad_range += b
                                    A.n_bad += b

                        # Monotonicity: hit_p5=>...=>hit_p1 (when all involved exist)
                        # We check only consecutive pairs that exist
                        for (k2, c2), (k1, c1) in zip(hit_p_cols[1:], hit_p_cols[:-1]):
                            v2 = pd.to_numeric(df[c2], errors="coerce").to_numpy(dtype=np.float64)
                            v1 = pd.to_numeric(df[c1], errors="coerce").to_numpy(dtype=np.float64)
                            ok = np.isfinite(v2) & np.isfinite(v1)
                            if ok.any():
                                # if higher barrier hit then lower must be hit
                                bad = (v2[ok] == 1.0) & (v1[ok] != 1.0)
                                if bad.any():
                                    b = int(bad.sum())
                                    A.n_bad_monotonic += b
                                    A.n_bad += b

                        for (k2, c2), (k1, c1) in zip(hit_m_cols[1:], hit_m_cols[:-1]):
                            v2 = pd.to_numeric(df[c2], errors="coerce").to_numpy(dtype=np.float64)
                            v1 = pd.to_numeric(df[c1], errors="coerce").to_numpy(dtype=np.float64)
                            ok = np.isfinite(v2) & np.isfinite(v1)
                            if ok.any():
                                bad = (v2[ok] == 1.0) & (v1[ok] != 1.0)
                                if bad.any():
                                    b = int(bad.sum())
                                    A.n_bad_monotonic += b
                                    A.n_bad += b

                        # Coherence with first_touch if both are available:
                        if (ft_col in df.columns):
                            ft = pd.to_numeric(df[ft_col], errors="coerce").to_numpy(dtype=np.float64)
                            okft = np.isfinite(ft)
                            if okft.any():
                                ft_i = ft[okft].astype(np.int16)

                                # Build arrays for specific hit cols if present
                                # first_touch=+k -> hit_p{k}=1; first_touch=-k -> hit_m{k}=1
                                for k in AUDIT_POS_LEVELS:
                                    c = _audit_hit_col(side_prefix, "p", k, h)
                                    if c not in df.columns:
                                        continue
                                    hp = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)[okft]
                                    ok = np.isfinite(hp)
                                    if ok.any():
                                        bad = (ft_i[ok] == k) & (hp[ok] != 1.0)
                                        if bad.any():
                                            b = int(bad.sum())
                                            A.n_bad_first_touch += b
                                            A.n_bad += b

                                for k in AUDIT_NEG_LEVELS:
                                    c = _audit_hit_col(side_prefix, "m", k, h)
                                    if c not in df.columns:
                                        continue
                                    hm = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)[okft]
                                    ok = np.isfinite(hm)
                                    if ok.any():
                                        bad = (ft_i[ok] == -k) & (hm[ok] != 1.0)
                                        if bad.any():
                                            b = int(bad.sum())
                                            A.n_bad_first_touch += b
                                            A.n_bad += b
            if (ci % 20) == 0:
                logger.info(f"  chunk {ci}: rows_so_far={agg.n_rows:,}")
    finally:
        # ferme le handle local si utilisé
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    # Post-file issues
    if agg.n_dup_row_id:
        issues.append(IntegrityIssue(path, symbol, year, "duplicate_row_id_total", f"count={agg.n_dup_row_id}"))
    if agg.n_dup_ts:
        issues.append(IntegrityIssue(path, symbol, year, "duplicate_t", f"count={agg.n_dup_ts}"))
    if agg.n_nonmonotonic_chunks:
        issues.append(IntegrityIssue(path, symbol, year, "non_monotonic_chunks", f"count={agg.n_nonmonotonic_chunks}"))
    if agg.n_bad_ts:
        issues.append(IntegrityIssue(path, symbol, year, "bad_timestamp_parse_total", f"count={agg.n_bad_ts}"))

    if agg.n_bad_spread_lt0:
        issues.append(IntegrityIssue(path, symbol, year, "bad_spread_lt0", f"count={agg.n_bad_spread_lt0}"))
    if agg.n_bad_atr_lt0:
        issues.append(IntegrityIssue(path, symbol, year, "bad_atr_lt0", f"count={agg.n_bad_atr_lt0}"))
    if agg.n_bad_entry_px_le0:
        issues.append(IntegrityIssue(path, symbol, year, "bad_entry_px_le0", f"count={agg.n_bad_entry_px_le0}"))

    if agg.n_checked_go_vs_rr and agg.n_bad_go_vs_rr:
        issues.append(IntegrityIssue(path, symbol, year, "go_vs_rr_mismatch",
                                     f"bad={agg.n_bad_go_vs_rr} / checked={agg.n_checked_go_vs_rr}"))
    if agg.n_checked_dir_vs_rr and agg.n_bad_dir_vs_rr:
        issues.append(IntegrityIssue(path, symbol, year, "dir_vs_rr_mismatch",
                                     f"bad={agg.n_bad_dir_vs_rr} / checked={agg.n_checked_dir_vs_rr}"))
    
    if math.isfinite(agg.spread_min) and agg.spread_min < 0:
        issues.append(IntegrityIssue(
            path, symbol, year, "spread_bps_entry_min_lt0",
            f"min={agg.spread_min:.6f} (approx count={agg.spread_min_count})"
        ))

    if agg.n_checked_spread_recalc and agg.n_bad_spread_recalc_mismatch:
        issues.append(IntegrityIssue(
            path, symbol, year, "spread_bps_entry_recalc_mismatch",
            f"bad={agg.n_bad_spread_recalc_mismatch} / checked={agg.n_checked_spread_recalc}"
        ))
    
    if agg.audit.n_bad:
        issues.append(IntegrityIssue(
            path, symbol, year, "audit_stage1_v3_inconsistencies",
            f"bad={agg.audit.n_bad} checked≈{agg.audit.n_checked} "
            f"(range={agg.audit.n_bad_range}, mono={agg.audit.n_bad_monotonic}, "
            f"first_touch={agg.audit.n_bad_first_touch}, step={agg.audit.n_bad_first_touch_step}, "
            f"mfe_mae={agg.audit.n_bad_mfe_mae})"
        ))

    # RR-only per-label issues
    for key, al in agg.by_label.items():
        if not key.startswith("rr:"):
            continue
        h = key.split(":", 1)[1]
        if al.n_bad_y_exit_reason:
            issues.append(IntegrityIssue(path, symbol, year, "y_exit_reason_inconsistent",
                                         f"h={h} count={al.n_bad_y_exit_reason}"))
        if al.n_bad_rr_formula:
            issues.append(IntegrityIssue(path, symbol, year, "bad_rr_formula",
                                         f"h={h} count={al.n_bad_rr_formula}"))
        if al.n_bad_entry_spread_half:
            issues.append(IntegrityIssue(path, symbol, year, "bad_entry_spread_half_bps",
                                         f"h={h} count={al.n_bad_entry_spread_half}"))
        if al.n_bad_pnl_sl_sign:
            issues.append(IntegrityIssue(path, symbol, year, "bad_pnl_sl_sign",
                                         f"h={h} count={al.n_bad_pnl_sl_sign}"))
        if al.n_bad_pnl_tp_sign:
            issues.append(IntegrityIssue(path, symbol, year, "bad_pnl_tp_sign",
                                         f"h={h} count={al.n_bad_pnl_tp_sign}"))

    return symbol, year, agg, issues


# ============================
# Summaries
# ============================

def summarize_file_metrics(symbol: str, year: int, agg: AggFile) -> Dict[str, object]:
    step_q = agg.step_seconds_samples.quantiles((0.5,))
    return {
        "symbol": symbol,
        "year": year,
        "rows": agg.n_rows,
        "dup_row_id": agg.n_dup_row_id,
        "bad_ts": agg.n_bad_ts,
        "dup_ts": agg.n_dup_ts,
        "non_monotonic_chunks": agg.n_nonmonotonic_chunks,
        "bad_entry_px_le0": agg.n_bad_entry_px_le0,
        "bad_spread_lt0": agg.n_bad_spread_lt0,
        "bad_atr_lt0": agg.n_bad_atr_lt0,
        "base_step_seconds_median": step_q.get("q50", float("nan")),
        "go_vs_rr_bad": agg.n_bad_go_vs_rr,
        "go_vs_rr_checked": agg.n_checked_go_vs_rr,
        "dir_vs_rr_bad": agg.n_bad_dir_vs_rr,
        "dir_vs_rr_checked": agg.n_checked_dir_vs_rr,
        "spread_bps_entry_min": (agg.spread_min if math.isfinite(agg.spread_min) else float("nan")),
        "spread_bps_entry_min_count": agg.spread_min_count,
        "spread_recalc_bad": agg.n_bad_spread_recalc_mismatch,
        "spread_recalc_checked": agg.n_checked_spread_recalc,
        "audit_checked": agg.audit.n_checked,
        "audit_bad": agg.audit.n_bad,
        "audit_bad_range": agg.audit.n_bad_range,
        "audit_bad_monotonic": agg.audit.n_bad_monotonic,
        "audit_bad_first_touch": agg.audit.n_bad_first_touch,
        "audit_bad_first_touch_step": agg.audit.n_bad_first_touch_step,
        "audit_bad_mfe_mae": agg.audit.n_bad_mfe_mae,
    }

def summarize_label_metrics(symbol: str, year: int, key: str, al: AggLabel) -> Dict[str, object]:
    family, horizon = key.split(":", 1)
    n = al.n
    pos = al.n_pos
    neg = al.n_neg
    z = al.n_zero

    if family == "go":
        trade_n = pos
        trade_rate = safe_rate(pos, n)
        long_share = float("nan")
        short_share = float("nan")
    else:
        trade_n = pos + neg
        trade_rate = safe_rate(trade_n, n)
        long_share = safe_rate(pos, trade_n)
        short_share = safe_rate(neg, trade_n)

    pnl_q = al.pnl_trade.quantiles()

    return {
        "symbol": symbol,
        "year": year,
        "family": family,
        "horizon": horizon,
        "key": key,
        "n": n,
        "n_pos": pos,
        "n_neg": neg,
        "n_zero": z,
        "trade_n": trade_n,
        "trade_rate": trade_rate,
        "long_share_on_trades": long_share,
        "short_share_on_trades": short_share,
        "pnl_trade_q50": pnl_q.get("q50", float("nan")),
        "pnl_trade_q75": pnl_q.get("q75", float("nan")),
        "pnl_trade_q90": pnl_q.get("q90", float("nan")),
        # RR-only diagnostics
        "rr_bad_y_exit_reason": al.n_bad_y_exit_reason,
        "rr_bad_rr_formula": al.n_bad_rr_formula,
        "rr_bad_entry_spread_half": al.n_bad_entry_spread_half,
        "rr_bad_pnl_sl_sign": al.n_bad_pnl_sl_sign,
        "rr_bad_pnl_tp_sign": al.n_bad_pnl_tp_sign,
        # RR-only exit stats
        "rr_exit_tp_long": al.n_tp_long,
        "rr_exit_tp_short": al.n_tp_short,
        "rr_exit_sl_long": al.n_sl_long,
        "rr_exit_sl_short": al.n_sl_short,
        "rr_exit_time": al.n_time,
        "rr_exit_nofill": al.n_nofill,
        "rr_exit_none": al.n_none,
        "p_thr_ev0_q50": al.p_thr_ev0_samples.quantiles((0.5,)).get("q50", float("nan")),
        "p_thr_ev0_bad_nan": al.n_bad_p_thr_ev0_nan,
        "p_thr_ev0_bad_range": al.n_bad_p_thr_ev0_range,
        "p_thr_ev0_formula":al.n_bad_p_thr_ev0_formula,
    }

def build_monthly_df(all_aggs: List[Tuple[str, int, AggFile]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sym, year, agg in all_aggs:
        for (key, month, yv), cnt in agg.monthly.items():
            family, horizon = key.split(":", 1)
            rows.append({
                "symbol": sym,
                "year": year,
                "family": family,
                "horizon": horizon,
                "key": key,
                "month": month,
                "y": int(yv),
                "count": int(cnt),
            })
    if not rows:
        return pd.DataFrame(columns=["symbol","year","family","horizon","key","month","y","count"])
    return pd.DataFrame(rows).sort_values(["symbol","year","family","horizon","month","y"])


# ============================
# Recommendations (GO + DIR)
# ============================

def build_recommendations(df_label: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("=== Recommandations Stage1 (GO + DIR + RR) ===")
    lines.append("")
    if df_label.empty:
        lines.append("Aucune métrique label disponible.")
        return "\n".join(lines)

    # GO: want enough positives
    df_go = df_label[df_label["family"] == "go"].copy()
    if not df_go.empty:
        lines.append("[GO] Qualité / Gate (y_go_<tf>)")
        g = df_go.groupby(["symbol", "horizon"], as_index=False).agg(
            n=("n", "sum"),
            pos=("n_pos", "sum"),
        )
        for _, r in g.sort_values(["symbol", "horizon"]).iterrows():
            n = int(r["n"]); pos = int(r["pos"])
            rate = (pos / n) if n else float("nan")
            verdict = "✅" if (pos >= 1500 and rate >= 0.01) else ("⚠️" if pos >= 300 else "❌")
            lines.append(f"- {r['symbol']} {r['horizon']}: pos={pos} / {n} ({rate:.2%}) {verdict}")
        lines.append("")

    # DIR: want balance long/short on trades
    df_dir = df_label[df_label["family"] == "dir"].copy()
    if not df_dir.empty:
        lines.append("[DIR] Direction (y_dir_<tf>)")
        g = df_dir.groupby(["symbol", "horizon"], as_index=False).agg(
            trade_n=("trade_n", "sum"),
            long_share=("long_share_on_trades", "mean"),
            short_share=("short_share_on_trades", "mean"),
        )
        for _, r in g.sort_values(["symbol", "horizon"]).iterrows():
            tn = float(r["trade_n"])
            ls = float(r["long_share"]) if pd.notna(r["long_share"]) else float("nan")
            ss = float(r["short_share"]) if pd.notna(r["short_share"]) else float("nan")
            if tn < 500:
                verdict = "❌ pas assez de trades"
            elif not (math.isfinite(ls) and math.isfinite(ss)):
                verdict = "⚠️"
            elif ss < 0.10 or ls < 0.10:
                verdict = "❌ déséquilibré (une direction quasi absente)"
            else:
                verdict = "✅"
            lines.append(f"- {r['symbol']} {r['horizon']}: trades≈{int(tn)} long≈{ls:.1%} short≈{ss:.1%} {verdict}")
        lines.append("")

    # RR notes
    df_rr = df_label[df_label["family"] == "rr"].copy()
    if not df_rr.empty:
        lines.append("[RR] Notes sur RR (y_<tf>)")
        lines.append("- RR est la source de vérité: GO doit matcher (TP_*), DIR doit matcher y.")
        lines.append("- Vérifie que les SL ont pnl négatif et les TP pnl positif (checks intégrés).")
        lines.append("- p_thr_ev0_<tf> doit être dans [0,1] et cohérent avec (SL+cost)/(TP+SL).")
        lines.append("")

    return "\n".join(lines)


# ============================
# Main
# ============================

def main():
    ap = argparse.ArgumentParser("Validate Stage1 labels (RR + GO + DIR)")
    ap.add_argument("--root", required=True, help="Root containing <SYMBOL>/<YEAR>_signals.csv (local or s3://...)")
    ap.add_argument("--outdir", required=True, help="Output directory (local path or s3://...)")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args()

    logger = setup_logger(args.loglevel)
    so = s3_storage_options(args.aws_region)

    root = normalize_root(args.root)
    outdir = normalize_root(args.outdir)

    paths = list_signal_files(root, so)
    logger.info(f"FOUND files={len(paths)} root={root}")

    if args.years:
        years_set = set(int(y) for y in args.years)
        paths = [p for p in paths
                 if os.path.basename(p).split("_", 1)[0].isdigit()
                 and int(os.path.basename(p).split("_", 1)[0]) in years_set]

    if args.symbols:
        symset = set(s.upper() for s in args.symbols)
        paths = [p for p in paths if os.path.basename(os.path.dirname(p.rstrip("/"))).upper() in symset]

    paths = sorted(paths)
    if not paths:
        logger.error("No signals found after filters.")
        sys.exit(2)

    rng = np.random.default_rng(args.seed)

    all_overall: List[Dict[str, object]] = []
    all_label: List[Dict[str, object]] = []
    all_issues: List[IntegrityIssue] = []

    # keep aggs for monthly build
    kept_aggs: List[Tuple[str, int, AggFile]] = []

    t0 = time.perf_counter()

    for i, p in enumerate(paths, 1):
        logger.info(f"[{i}/{len(paths)}] {p}")
        sym, year, agg, issues = process_signals_file(
            p, so=so, chunksize=args.chunksize, rng=rng, logger=logger
        )
        all_issues.extend(issues)
        all_overall.append(summarize_file_metrics(sym, year, agg))
        kept_aggs.append((sym, year, agg))

        for key, al in agg.by_label.items():
            all_label.append(summarize_label_metrics(sym, year, key, al))

    df_overall = pd.DataFrame(all_overall).sort_values(["symbol", "year"])
    df_label = pd.DataFrame(all_label).sort_values(["symbol", "year", "family", "horizon"])
    df_monthly = build_monthly_df(kept_aggs)

    if all_issues:
        df_issues = pd.DataFrame([{
            "path": x.path,
            "symbol": x.symbol,
            "year": x.year,
            "issue": x.issue,
            "details": x.details
        } for x in all_issues]).sort_values(["symbol", "year", "issue"])
    else:
        df_issues = pd.DataFrame(columns=["path", "symbol", "year", "issue", "details"])

    # write outputs
    write_df_csv(join_path(outdir, "metrics_overall.csv"), df_overall, so)
    write_df_csv(join_path(outdir, "metrics_by_label.csv"), df_label, so)
    write_df_csv(join_path(outdir, "monthly_by_label.csv"), df_monthly, so)
    write_df_csv(join_path(outdir, "integrity_issues.csv"), df_issues, so)

    rec = build_recommendations(df_label)
    write_text(join_path(outdir, "recommendations.txt"), rec, so)

    logger.info(f"WROTE {join_path(outdir,'metrics_overall.csv')}")
    logger.info(f"WROTE {join_path(outdir,'metrics_by_label.csv')}")
    logger.info(f"WROTE {join_path(outdir,'monthly_by_label.csv')}")
    logger.info(f"WROTE {join_path(outdir,'integrity_issues.csv')}")
    logger.info(f"WROTE {join_path(outdir,'recommendations.txt')}")

    elapsed = time.perf_counter() - t0
    logger.info(f"DONE in {elapsed:.1f}s | files={len(paths)}")


if __name__ == "__main__":
    main()