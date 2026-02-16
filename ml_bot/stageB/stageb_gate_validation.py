#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stageb_gate_validation.py

Validate StageB gate quality vs Label_A (non-directional opportunity label).

Inputs:
- StageB dataset parquet:
  s3://.../data/stageB/dataset=v1/split={train|val|test}/parquet/*.parquet
  (files are BTCUSDT_YYYY-MM_partX.parquet)
  Must contain: id_t, label_A, audit_p_thr_ev0 (recommended), label_A_pnl_net_bps (optional)

- StageB scored parquet:
  s3://.../data/stageB/dataset=v1/scored/symbol=BTCUSDT/split={train|val|test}/parquet/*.parquet
  Must contain: id_t, stgB_allow, stgB_s, stgB_thr_s
  Optional: stgB_p, audit_p_thr_ev0 (sometimes copied), stgB_model_stamp/uri

What it reports:
- Match rate dataset vs scored by id_t
- allow rate
- Opportunity rate uplift: P(label_A=1 | allow=1) - P(label_A=1 | allow=0)
- Lift curve by quantiles of stgB_s
- Optional calibration curve (bins of stgB_p)
- Optional AUC if stgB_p exists

Outputs:
- summary CSV (local or s3)
- lift curve CSV
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import s3fs
import pyarrow.parquet as pq


# -----------------------
# S3 helpers
# -----------------------
def list_s3(fs: s3fs.S3FileSystem, pattern_s3: str) -> List[str]:
    pat = pattern_s3.replace("s3://", "")
    return sorted([f"s3://{x}" for x in fs.glob(pat)])

def read_parquet_cols(path: str, cols: List[str]) -> pd.DataFrame:
    # pandas can read s3 with s3fs installed; keep columns small
    return pd.read_parquet(path, engine="pyarrow", columns=cols)

def utc_dt(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")

def to_epoch_ns(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    # int64 ns
    return t.astype("int64").to_numpy(np.int64, copy=False)

def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")

def safe_rate_bool(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


# -----------------------
# Optional AUC (no sklearn dependency)
# -----------------------
def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes ROC AUC using rank method.
    y_true: 0/1
    """
    y_true = y_true.astype(np.int8)
    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]
    if y_true.size == 0:
        return float("nan")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # handle ties by average rank
    ys = y_score[order]
    i = 0
    while i < len(ys):
        j = i + 1
        while j < len(ys) and ys[j] == ys[i]:
            j += 1
        if j - i > 1:
            r = ranks[order[i:j]].mean()
            ranks[order[i:j]] = r
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# -----------------------
# Core aggregation
# -----------------------
@dataclass
class SplitAgg:
    split: str
    rows_ds: int = 0
    rows_scored: int = 0
    rows_join: int = 0

    allow0: int = 0
    allow1: int = 0

    # label stats
    pos_allow0: int = 0
    pos_allow1: int = 0

    # store for lift/calibration (reservoir sampling)
    # keep only a sample to avoid RAM blow-up
    sample_cap: int = 2_000_000
    _s_list: List[np.ndarray] = None
    _p_list: List[np.ndarray] = None
    _y_list: List[np.ndarray] = None
    _a_list: List[np.ndarray] = None

    def __post_init__(self):
        self._s_list = []
        self._p_list = []
        self._y_list = []
        self._a_list = []

    def add_chunk(self, s: np.ndarray, p: Optional[np.ndarray], y: np.ndarray, allow: np.ndarray):
        self.rows_join += int(len(y))

        a1 = (allow == 1)
        a0 = ~a1
        self.allow1 += int(a1.sum())
        self.allow0 += int(a0.sum())

        self.pos_allow1 += int(((y == 1) & a1).sum())
        self.pos_allow0 += int(((y == 1) & a0).sum())

        # sample storage (uniform-ish): keep all until cap, then thin by random
        cur = sum(arr.size for arr in self._y_list)
        if cur >= self.sample_cap:
            return

        room = self.sample_cap - cur
        if len(y) > room:
            idx = np.random.default_rng(123).choice(len(y), size=room, replace=False)
            s = s[idx]
            allow = allow[idx]
            y = y[idx]
            if p is not None:
                p = p[idx]

        self._s_list.append(s.astype(np.float32, copy=False))
        self._a_list.append(allow.astype(np.int8, copy=False))
        self._y_list.append(y.astype(np.int8, copy=False))
        if p is not None:
            self._p_list.append(p.astype(np.float32, copy=False))

    def finalize_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        s = np.concatenate(self._s_list) if self._s_list else np.array([], dtype=np.float32)
        a = np.concatenate(self._a_list) if self._a_list else np.array([], dtype=np.int8)
        y = np.concatenate(self._y_list) if self._y_list else np.array([], dtype=np.int8)
        p = np.concatenate(self._p_list) if self._p_list else None
        return s, a, y, p


def load_and_join_split(
    fs: s3fs.S3FileSystem,
    split: str,
    ds_glob: str,
    scored_glob: str,
    symbol: str,
) -> Tuple[SplitAgg, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - agg
      - lift_curve_df
      - calib_curve_df (may be empty)
    """
    # dataset files are BTCUSDT_YYYY-MM_partX.parquet
    ds_files = list_s3(fs, ds_glob)
    sc_files = list_s3(fs, scored_glob)

    print(f"[{split}] dataset_files={len(ds_files)} scored_files={len(sc_files)}")

    if not ds_files:
        raise FileNotFoundError(f"No dataset files for split={split} under {ds_glob}")
    if not sc_files:
        raise FileNotFoundError(f"No scored files for split={split} under {scored_glob}")

    agg = SplitAgg(split=split)

    # Read file by file (assume same partitioning count; but we join by id_t anyway)
    # We'll build a dict for scored per file only if filenames align; otherwise use two-pass map.
    # For robustness, we create an index map from (month token) -> list of scored files.
    def month_key(path: str) -> str:
        # try extract "YYYY-MM" from filename or folder
        base = path.split("/")[-1]
        for token in base.replace(".", "_").split("_"):
            if len(token) == 7 and token[4] == "-":
                return token
        # fallback: unknown
        return "UNKNOWN"

    sc_by_m = {}
    for p in sc_files:
        sc_by_m.setdefault(month_key(p), []).append(p)

    # columns
    ds_cols = ["id_t", "label_A"]
    # if exists, keep for optional sanity
    ds_cols_opt = ["audit_p_thr_ev0", "label_A_pnl_net_bps"]
    # scored required
    sc_cols = ["id_t", "stgB_s", "stgB_allow", "stgB_thr_s"]
    # optional
    sc_cols_opt = ["stgB_p", "audit_p_thr_ev0"]

    # iterate dataset files and match scored files by month
    for ds_path in ds_files:
        m = month_key(ds_path)
        sc_candidates = sc_by_m.get(m, sc_files)  # fallback all if no key

        # load dataset
        ds_cols_here = ds_cols.copy()
        # detect optional columns quickly via parquet schema
        with fs.open(ds_path, "rb") as f:
            sch = pq.ParquetFile(f).schema.names
        for c in ds_cols_opt:
            if c in sch:
                ds_cols_here.append(c)

        ds = read_parquet_cols(ds_path, ds_cols_here)
        agg.rows_ds += int(len(ds))

        # load scored candidates and merge (may be 1 or multiple parts)
        # We keep only columns that exist
        merged = None
        for sc_path in sc_candidates:
            with fs.open(sc_path, "rb") as f:
                sch2 = pq.ParquetFile(f).schema.names
            sc_cols_here = [c for c in sc_cols if c in sch2]
            for c in sc_cols_opt:
                if c in sch2:
                    sc_cols_here.append(c)

            sc = read_parquet_cols(sc_path, sc_cols_here)
            agg.rows_scored += int(len(sc))

            # join on epoch-ns id_t for robustness
            ds_key = to_epoch_ns(ds["id_t"])
            sc_key = to_epoch_ns(sc["id_t"])
            ds2 = ds.copy()
            sc2 = sc.copy()
            ds2["_k"] = ds_key
            sc2["_k"] = sc_key

            j = ds2.merge(sc2, on="_k", how="inner", suffixes=("", "_sc"))
            j = j.drop(columns=["_k"], errors="ignore")
            if merged is None:
                merged = j
            else:
                merged = pd.concat([merged, j], ignore_index=True)

        if merged is None or merged.empty:
            continue

        # count join
        agg.rows_join += 0  # we add via add_chunk only; keep this here unused

        # required fields (post-join)
        y = pd.to_numeric(merged["label_A"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        allow = pd.to_numeric(merged["stgB_allow"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        s = pd.to_numeric(merged["stgB_s"], errors="coerce").to_numpy(np.float32, copy=False)

        p = None
        if "stgB_p" in merged.columns:
            p = pd.to_numeric(merged["stgB_p"], errors="coerce").to_numpy(np.float32, copy=False)

        # add chunk to agg (with sampling)
        agg.add_chunk(s=s, p=p, y=y, allow=allow)

    # finalize sample arrays
    s_all, a_all, y_all, p_all = agg.finalize_arrays()

    # build lift curve by quantiles of s (using sample)
    lift_df = pd.DataFrame()
    if s_all.size:
        # quantile bins on finite s
        m = np.isfinite(s_all)
        s_f = s_all[m]
        y_f = y_all[m]
        a_f = a_all[m]
        if s_f.size:
            q = np.quantile(s_f, np.linspace(0, 1, 11))
            # make unique edges
            q = np.unique(q)
            if q.size >= 3:
                bin_id = np.digitize(s_f, q[1:-1], right=True)  # 0..nbin-1
                rows = []
                for b in range(bin_id.min(), bin_id.max() + 1):
                    mb = (bin_id == b)
                    if not mb.any():
                        continue
                    rows.append({
                        "split": split,
                        "bin": int(b),
                        "s_lo": float(np.nanmin(s_f[mb])),
                        "s_hi": float(np.nanmax(s_f[mb])),
                        "rows": int(mb.sum()),
                        "pos_rate": float((y_f[mb] == 1).mean()),
                        "allow_rate": float((a_f[mb] == 1).mean()),
                    })
                lift_df = pd.DataFrame(rows).sort_values(["split", "bin"]).reset_index(drop=True)

    # calibration curve by bins of p (if p exists)
    calib_df = pd.DataFrame()
    if p_all is not None and p_all.size:
        m = np.isfinite(p_all)
        p_f = p_all[m]
        y_f = y_all[m]
        if p_f.size:
            edges = np.linspace(0, 1, 11)
            b = np.clip(np.digitize(p_f, edges[1:-1], right=True), 0, 9)
            rows = []
            for k in range(10):
                mk = (b == k)
                if not mk.any():
                    continue
                rows.append({
                    "split": split,
                    "bin": int(k),
                    "p_lo": float(edges[k]),
                    "p_hi": float(edges[k+1]),
                    "rows": int(mk.sum()),
                    "p_mean": float(np.mean(p_f[mk])),
                    "pos_rate": float(np.mean(y_f[mk] == 1)),
                })
            calib_df = pd.DataFrame(rows).sort_values(["split", "bin"]).reset_index(drop=True)

    return agg, lift_df, calib_df


def summarize_split(agg: SplitAgg) -> Dict[str, float]:
    # allow rates + uplift on label_A
    allow1 = agg.allow1
    allow0 = agg.allow0
    pos1 = agg.pos_allow1
    pos0 = agg.pos_allow0

    r1 = (pos1 / allow1) if allow1 else float("nan")
    r0 = (pos0 / allow0) if allow0 else float("nan")
    uplift = (r1 - r0) if (np.isfinite(r1) and np.isfinite(r0)) else float("nan")

    # compute AUC on sample if p exists
    s_all, a_all, y_all, p_all = agg.finalize_arrays()
    auc = float("nan")
    if p_all is not None and p_all.size:
        auc = fast_auc(y_all, p_all)

    # also compute mean(s) by allow groups (good sanity)
    ms1 = float(np.nanmean(s_all[a_all == 1])) if s_all.size and (a_all == 1).any() else float("nan")
    ms0 = float(np.nanmean(s_all[a_all == 0])) if s_all.size and (a_all == 0).any() else float("nan")

    return {
        "split": agg.split,
        "ds_rows": agg.rows_ds,
        "scored_rows": agg.rows_scored,
        "joined_rows_sampled": int(s_all.size),
        "allow1_rows": allow1,
        "allow0_rows": allow0,
        "allow_rate": (allow1 / (allow1 + allow0)) if (allow1 + allow0) else float("nan"),
        "pos_rate_allow1": r1,
        "pos_rate_allow0": r0,
        "uplift_pos_rate": uplift,
        "mean_s_allow1": ms1,
        "mean_s_allow0": ms0,
        "auc_if_p_exists": auc,
    }


def write_csv(fs: s3fs.S3FileSystem, path: str, df: pd.DataFrame):
    if path.startswith("s3://"):
        with fs.open(path, "wb") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])

    ap.add_argument("--stageb-ds-root", 
                    default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/split={split}/parquet/*.parquet")
    ap.add_argument("--stageb-scored-root", 
                    default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/scored/symbol=BTCUSDT/split={split}/parquet/*.parquet")

    ap.add_argument("--out-summary-csv", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/stageb_gate_summary.csv")
    ap.add_argument("--out-lift-csv", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/stageb_gate_lift.csv")
    ap.add_argument("--out-calib-csv", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/stageb_gate_calib.csv")

    args = ap.parse_args()

    fs = s3fs.S3FileSystem()

    summaries = []
    lift_all = []
    calib_all = []

    for sp in args.splits:
        ds_glob = args.stageb_ds_root.format(split=sp, symbol=args.symbol)
        sc_glob = args.stageb_scored_root.format(split=sp, symbol=args.symbol)

        agg, lift_df, calib_df = load_and_join_split(
            fs=fs, split=sp, ds_glob=ds_glob, scored_glob=sc_glob, symbol=args.symbol
        )

        summ = summarize_split(agg)
        summaries.append(summ)

        if not lift_df.empty:
            lift_all.append(lift_df)
        if not calib_df.empty:
            calib_all.append(calib_df)

        print(f"=== {sp} ===")
        print(json.dumps(summ, indent=2))

    summary_df = pd.DataFrame(summaries)
    lift_df = pd.concat(lift_all, ignore_index=True) if lift_all else pd.DataFrame()
    calib_df = pd.concat(calib_all, ignore_index=True) if calib_all else pd.DataFrame()

    write_csv(fs, args.out_summary_csv, summary_df)
    write_csv(fs, args.out_lift_csv, lift_df)

    if args.out_calib_csv and not calib_df.empty:
        write_csv(fs, args.out_calib_csv, calib_df)

    print(f"Saved summary: {args.out_summary_csv}")
    print(f"Saved lift:    {args.out_lift_csv}")
    if args.out_calib_csv:
        print(f"Saved calib:   {args.out_calib_csv}")


if __name__ == "__main__":
    main()