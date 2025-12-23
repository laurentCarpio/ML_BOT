#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_stage2_dataset.py — Stage2 v3.3 (perf+logs fixed)

Fixes / Improvements vs v3.2:
- FIX: miss logs only printed on actual miss (no spam, no undefined vars).
- PERF: default bucket = 1h (CLI --bucket), drastically fewer S3 reads.
- PERF: book ret_stdev computed tick-based with rolling("10s") (no resample("1s")).
- LOGS: month start/end + progress every N events + per-month ETA + timing breakdown.
- Keeps same features and semantics as v3.1/v3.2 (go/dir + audit_* + task).

v3.3.1 PATCH (this file):
- FIX: trades parquet -> to_pandas() (avoid ArrowDtype timestamps issues).
- FIX: staleness dt calc uses int64 ns (tz-safe) instead of datetime64[ns] casts.
- CLEANUP: removed redundant book_df reset_index() line.
"""

from __future__ import annotations

import argparse
import re
import gc
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs as pa_fs
from pyarrow import types as patypes
import s3fs


# ------------------------------
# Constants
# ------------------------------
BOOK_LEVELS = 15

def TF_SORT_KEY(tf: str):
    tf = (tf or "").lower().strip()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    return 10**9

MAX_STALE_BY_TF = {
    "5m":  pd.Timedelta(seconds=1),
    "15m": pd.Timedelta(seconds=2),
    "30m": pd.Timedelta(seconds=3),
    "1h":  pd.Timedelta(seconds=5),
    "2h":  pd.Timedelta(seconds=6),
    "4h":  pd.Timedelta(seconds=10),
}

# cache sizes (buckets)
MAX_BOOK_CACHE = 128
MAX_TRADES_CACHE = 256

BATCH_MAX = 10_000
LOG_EVERY = 5_000


# ------------------------------
# Logging
# ------------------------------
def log(msg: str):
    print(f"{pd.Timestamp.utcnow().isoformat()}Z | {msg}", flush=True)


# ------------------------------
# S3 / Arrow helpers
# ------------------------------
def so(region: str, anon: bool) -> dict:
    out = {"client_kwargs": {"region_name": region}}
    if anon:
        out["anon"] = True
    return out

def s3fs_from_so(storage_options: dict) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(**(storage_options or {}))

def pa_filesystem(storage_options: dict) -> pa_fs.FileSystem:
    s3 = s3fs_from_so(storage_options or {})
    return pa_fs.PyFileSystem(pa_fs.FSSpecHandler(s3))

def norm(url: str) -> str:
    return re.sub(r"^([A-Za-z0-9]+)://", lambda m: m.group(1).lower() + "://", str(url).strip())

def ensure_s3(url: str) -> str:
    u = norm(url)
    if not u.startswith("s3://"):
        raise ValueError(f"S3 only: got {url}")
    return u

def s3_to_bucket_key(s3_url: str) -> str:
    u = ensure_s3(s3_url)
    return u[len("s3://"):]

def read_csv_s3(path: str, storage_options: dict) -> pd.DataFrame:
    path = ensure_s3(path)
    return pd.read_csv(path, storage_options=storage_options)

def write_parquet_s3(path: str, df: pd.DataFrame, storage_options: dict) -> None:
    path = ensure_s3(path)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False, storage_options=storage_options)

def mkdirs_s3(_prefix: str, _storage_options: dict) -> None:
    return

def arrow_time_filter(dset: ds.Dataset, ts_field: str, t0: pd.Timestamp, t1: pd.Timestamp):
    f = dset.schema.field(ts_field)
    if not patypes.is_timestamp(f.type):
        raise ValueError(f"{ts_field} is not timestamp: {f.type}")

    t0 = pd.to_datetime(t0, utc=True)
    t1 = pd.to_datetime(t1, utc=True)

    if getattr(f.type, "tz", None) is None:
        s0 = pa.scalar(t0.to_pydatetime().replace(tzinfo=None), type=f.type)
        s1 = pa.scalar(t1.to_pydatetime().replace(tzinfo=None), type=f.type)
    else:
        s0 = pa.scalar(t0.to_pydatetime(), type=f.type)
        s1 = pa.scalar(t1.to_pydatetime(), type=f.type)

    return (ds.field(ts_field) >= s0) & (ds.field(ts_field) <= s1)

def dataset_from_month_parquet(month_path_s3: str, pafs: pa_fs.FileSystem) -> ds.Dataset:
    p = s3_to_bucket_key(month_path_s3)
    return ds.dataset([p], format="parquet", filesystem=pafs)

def read_book_window(dset: ds.Dataset, t0: pd.Timestamp, t1: pd.Timestamp, cols: List[str]) -> pd.DataFrame:
    ts_field = "timestamp"
    names = set(dset.schema.names)
    if ts_field not in names:
        raise ValueError("Book parquet missing 'timestamp'")

    cols2 = list(dict.fromkeys([ts_field] + [c for c in cols if c in names]))
    flt = arrow_time_filter(dset, ts_field, t0, t1)
    tbl = dset.to_table(columns=cols2, filter=flt)
    if tbl.num_rows == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))

    df = tbl.to_pandas()  # IMPORTANT: avoid ArrowDtype timestamps that break merge_asof
    df[ts_field] = pd.to_datetime(df[ts_field], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_field]).sort_values(ts_field).set_index(ts_field)
    df.index.name = "timestamp"
    return df

def read_trades_window(dset: ds.Dataset, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    ts_field = "timestamp"
    names = set(dset.schema.names)
    if ts_field not in names:
        raise ValueError("Trades parquet missing 'timestamp'")

    want = [ts_field, "price", "qty", "is_aggr_buy", "side"]
    cols = [c for c in want if c in names]

    flt = arrow_time_filter(dset, ts_field, t0, t1)
    tbl = dset.to_table(columns=cols, filter=flt)
    if tbl.num_rows == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))

    # FIX v3.3.1: to_pandas() to avoid ArrowDtype timestamp quirks
    df = tbl.to_pandas()
    df[ts_field] = pd.to_datetime(df[ts_field], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_field]).sort_values(ts_field).set_index(ts_field)
    df.index.name = "timestamp"

    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce")
    if "is_aggr_buy" in df.columns:
        df["is_aggr_buy"] = df["is_aggr_buy"].astype("boolean")

    return df.dropna(subset=["price", "qty"]).sort_index()


# ------------------------------
# Feature helpers (book)
# ------------------------------
def obi_row(row: pd.Series, K: int) -> float:
    sb = 0.0
    sa = 0.0
    for i in range(K):
        sb += float(row.get(f"bid_{i}_size", 0.0) or 0.0)
        sa += float(row.get(f"ask_{i}_size", 0.0) or 0.0)
    tot = sb + sa
    return float((sb - sa) / tot) if tot > 0 else np.nan

def slope_from_levels(row: pd.Series, side: str, K: int) -> float:
    prices, sizes = [], []
    for i in range(K):
        prices.append(float(row.get(f"{side}_{i}_price", np.nan)))
        sizes.append(float(row.get(f"{side}_{i}_size", np.nan)))
    p = np.asarray(prices, float)
    q = np.asarray(sizes, float)
    ok = np.isfinite(p) & np.isfinite(q) & (q > 0)
    p = p[ok]; q = q[ok]
    if p.size < 2:
        return np.nan
    x = np.cumsum(q)
    xm, pm = x.mean(), p.mean()
    den = ((x - xm) ** 2).sum()
    if den <= 0:
        return np.nan
    return float(((x - xm) * (p - pm)).sum() / den)

def wall_opp_share(row: pd.Series, side: str, K: int) -> float:
    if side not in ("buy", "sell"):
        return np.nan
    opp = "ask" if side == "buy" else "bid"
    sizes = [float(row.get(f"{opp}_{i}_size", 0.0) or 0.0) for i in range(K)]
    tot = float(np.sum(sizes))
    return float(np.max(sizes) / tot) if tot > 0 else np.nan

def cum_depth_within_bps(row: pd.Series, side: str, mid: float, bps: float) -> float:
    if side not in ("buy", "sell"):
        return np.nan
    if not np.isfinite(mid) or mid <= 0:
        return np.nan
    tol = (bps / 1e4) * mid
    depth = 0.0
    if side == "buy":
        for i in range(BOOK_LEVELS):
            ap = float(row.get(f"ask_{i}_price", np.nan))
            aq = float(row.get(f"ask_{i}_size", 0.0) or 0.0)
            if np.isfinite(ap) and abs(ap - mid) <= tol:
                depth += aq
    else:
        for i in range(BOOK_LEVELS):
            bp = float(row.get(f"bid_{i}_price", np.nan))
            bq = float(row.get(f"bid_{i}_size", 0.0) or 0.0)
            if np.isfinite(bp) and abs(bp - mid) <= tol:
                depth += bq
    return float(depth)

def microprice_bias_from_top(row: pd.Series) -> float:
    bid = float(row.get("bid_0_price", np.nan))
    ask = float(row.get("ask_0_price", np.nan))
    bsz = float(row.get("bid_0_size", np.nan))
    asz = float(row.get("ask_0_size", np.nan))
    if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(bsz) and np.isfinite(asz)):
        return np.nan
    mid = (bid + ask) / 2.0
    den = bsz + asz
    if mid <= 0 or den <= 0:
        return np.nan
    w = asz / den
    micro = w * ask + (1.0 - w) * bid
    return float((micro - mid) / mid)

def side_adjust(x: float, side_num: int) -> float:
    return float(side_num * x) if np.isfinite(x) else np.nan


# ------------------------------
# Fast asof helpers
# ------------------------------
def _asof_get(series: pd.Series, t: pd.Timestamp) -> float:
    if series is None or series.empty:
        return np.nan
    idx = series.index
    j = idx.searchsorted(t, side="right") - 1
    if j < 0:
        return np.nan
    v = series.iloc[j]
    try:
        return float(v)
    except Exception:
        return np.nan

@dataclass
class BookPrepared:
    df: pd.DataFrame
    churn10: pd.Series           # rolling 10s quote churn
    retstd10_bps: pd.Series      # rolling std of mid returns over 10s (tick-based)

@dataclass
class TradesPrepared:
    df: pd.DataFrame
    aggr3: pd.Series
    aggr5: pd.Series
    aggr10: pd.Series
    aggr15: pd.Series


def prepare_book_features(book_full: pd.DataFrame) -> BookPrepared:
    if book_full is None or book_full.empty:
        empty = pd.Series(dtype=float)
        return BookPrepared(book_full, empty, empty)

    df = book_full
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    bsz = pd.to_numeric(df.get("bid_0_size"), errors="coerce").astype(float)
    asz = pd.to_numeric(df.get("ask_0_size"), errors="coerce").astype(float)
    churn_inst = bsz.diff().abs().fillna(0.0) + asz.diff().abs().fillna(0.0)
    churn10 = churn_inst.rolling("10s").sum()

    bid0 = pd.to_numeric(df.get("bid_0_price"), errors="coerce").astype(float)
    ask0 = pd.to_numeric(df.get("ask_0_price"), errors="coerce").astype(float)
    mid = (bid0 + ask0) / 2.0

    ret = mid.pct_change()
    retstd10_bps = (ret.rolling("10s").std() * 1e4).astype(float)

    return BookPrepared(df=df, churn10=churn10.astype(float), retstd10_bps=retstd10_bps)

def prepare_trades_features(trades_full: pd.DataFrame) -> TradesPrepared:
    if trades_full is None or trades_full.empty or "is_aggr_buy" not in trades_full.columns:
        empty = pd.Series(dtype=float)
        return TradesPrepared(trades_full, empty, empty, empty, empty)

    df = trades_full
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    qty = pd.to_numeric(df.get("qty"), errors="coerce").astype(float).fillna(0.0)
    is_buy = df["is_aggr_buy"].astype("boolean")
    buy_qty = qty.where(is_buy == True, 0.0)

    def r(window: str) -> pd.Series:
        buyw = buy_qty.rolling(window).sum()
        totw = qty.rolling(window).sum()
        out = (buyw / totw).replace([np.inf, -np.inf], np.nan)
        return out.astype(float)

    return TradesPrepared(df=df, aggr3=r("3s"), aggr5=r("5s"), aggr10=r("10s"), aggr15=r("15s"))

def _asof_values(series: pd.Series, t: pd.Series) -> np.ndarray:
    """
    Vectorized asof: for each t, return last series value where series.index <= t.
    Works with tz-aware (UTC) and avoids datetime64[ns] casts.
    """
    if series is None or series.empty or t is None or len(t) == 0:
        return np.full(len(t) if t is not None else 0, np.nan, dtype=float)

    s = series.dropna()
    if s.empty:
        return np.full(len(t), np.nan, dtype=float)

    if not s.index.is_monotonic_increasing:
        s = s.sort_index()

    tt = pd.to_datetime(t, utc=True, errors="coerce")
    tv = tt.astype("int64").to_numpy()
    ok_t = ~tt.isna().to_numpy()

    idx = s.index.asi8  # int64 ns
    pos = np.searchsorted(idx, tv, side="right") - 1

    out = np.full(len(t), np.nan, dtype=float)
    ok = ok_t & (pos >= 0)
    if ok.any():
        vals = s.to_numpy(dtype=float, copy=False)
        out[ok] = vals[pos[ok]]
    return out

def _slope_batch(prices: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    N, K = prices.shape
    out = np.full(N, np.nan, dtype=float)
    for i in range(N):
        p = prices[i]
        q = sizes[i]
        ok = np.isfinite(p) & np.isfinite(q) & (q > 0)
        if ok.sum() < 2:
            continue
        p2 = p[ok]
        q2 = q[ok]
        x = np.cumsum(q2)
        xm = x.mean()
        pm = p2.mean()
        den = np.sum((x - xm) ** 2)
        if den <= 0:
            continue
        out[i] = np.sum((x - xm) * (p2 - pm)) / den
    return out

def build_book_features_batch(df_snap: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_snap.index)

    bid0 = pd.to_numeric(df_snap.get("bid_0_price"), errors="coerce").astype(float)
    ask0 = pd.to_numeric(df_snap.get("ask_0_price"), errors="coerce").astype(float)
    bsz0 = pd.to_numeric(df_snap.get("bid_0_size"), errors="coerce").astype(float)
    asz0 = pd.to_numeric(df_snap.get("ask_0_size"), errors="coerce").astype(float)

    mid = (bid0 + ask0) / 2.0
    out["mid_t"] = mid

    out["spread_bps_top"] = np.where(mid > 0, 1e4 * (ask0 - bid0) / mid, np.nan)

    den = (bsz0 + asz0)
    w = np.where(den > 0, asz0 / den, np.nan)
    micro = w * ask0 + (1.0 - w) * bid0
    out["microprice_bias"] = np.where(mid > 0, (micro - mid) / mid, np.nan)

    for K in (5, 15):
        bid_sizes = []
        ask_sizes = []
        for i in range(K):
            bid_sizes.append(pd.to_numeric(df_snap.get(f"bid_{i}_size"), errors="coerce").astype(float).fillna(0.0).values)
            ask_sizes.append(pd.to_numeric(df_snap.get(f"ask_{i}_size"), errors="coerce").astype(float).fillna(0.0).values)
        sb = np.sum(np.vstack(bid_sizes), axis=0)
        sa = np.sum(np.vstack(ask_sizes), axis=0)
        tot = sb + sa
        out[f"obi_{K}"] = np.where(tot > 0, (sb - sa) / tot, np.nan)

    return out


# ------------------------------
# Output sink
# ------------------------------
@dataclass
class PartsSink:
    out_year_dir: str
    storage_options: dict
    k: int = 0

    def __post_init__(self):
        self.base = ensure_s3(self.out_year_dir).rstrip("/")
        self.parts = f"{self.base}/parts"
        mkdirs_s3(self.parts, self.storage_options)

    def write_df(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        if "Y" in df.columns:
            df["Y"] = pd.array(pd.to_numeric(df["Y"], errors="coerce"), dtype="Int8")
        part_path = f"{self.parts}/part-{self.k:05d}.parquet"
        write_parquet_s3(part_path, df, self.storage_options)
        self.k += 1


# ------------------------------
# Signals -> events
# ------------------------------
def parse_tfs_from_signals_columns(cols: List[str]) -> List[str]:
    tfs = []
    for c in cols:
        if not c.startswith("y_"):
            continue
        tail = c[2:]
        if tail.startswith("go_") or tail.startswith("dir_"):
            continue
        tfs.append(tail)
    return sorted(list(set(tfs)), key=TF_SORT_KEY)

def _col_or_default(df: pd.DataFrame, col: str, default, n: int) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * n, index=df.index)

def make_events_long_with_label_type(
    sig: pd.DataFrame,
    tfs: List[str],
    label_types: List[str],
    dir_include_neutral: bool,
    legacy_single_row: bool,
    include_zero_legacy: bool,
) -> pd.DataFrame:
    base_req = ["t", "symbol", "year", "bid_entry", "ask_entry", "spread_bps_entry", "atr_bps"]
    for c in base_req:
        if c not in sig.columns:
            raise ValueError(f"Signals missing required column: {c}")

    sig = sig.copy()
    sig["t"] = pd.to_datetime(sig["t"], utc=True, errors="coerce")
    sig = sig.dropna(subset=["t"])

    rows = []
    for tf in tfs:
        ycol = f"y_{tf}"
        if ycol not in sig.columns:
            continue

        tmp = sig[[
            "t","symbol","year",
            "bid_entry","ask_entry",
            "mid_entry" if "mid_entry" in sig.columns else "bid_entry",
            "spread_bps_entry","atr_bps"
        ]].copy()

        if "mid_entry" not in sig.columns:
            tmp = tmp.rename(columns={"bid_entry": "mid_entry"})

        tmp["tf"] = tf
        tmp["y_raw"] = pd.to_numeric(sig[ycol], errors="coerce").fillna(0).astype("int8")

        opt_fields = ["pnl_net_bps","exit_reason","tp_bps","sl_bps","rr_min","risk_r_bps","fill_mode","fill_window_sec"]
        for opt in opt_fields:
            copt = f"{opt}_{tf}"
            if copt in sig.columns:
                tmp[f"audit_{opt}"] = sig[copt]

        if legacy_single_row:
            tmp["label_type"] = "legacy"
            tmp["task"] = "legacy_" + tf
            tmp["Y"] = tmp["y_raw"].astype("int8")
            tmp["side"] = np.where(tmp["Y"] == 1, "buy", np.where(tmp["Y"] == -1, "sell", "none"))
            tmp["side_num"] = np.where(tmp["Y"] == 1, 1, np.where(tmp["Y"] == -1, -1, 0)).astype("int8")
            tmp["entry"] = np.where(tmp["side_num"] == 1,
                                    pd.to_numeric(tmp["bid_entry"], errors="coerce"),
                                    np.where(tmp["side_num"] == -1,
                                             pd.to_numeric(tmp["ask_entry"], errors="coerce"),
                                             np.nan)).astype(float)
            if not include_zero_legacy:
                tmp = tmp[tmp["Y"] != 0]
            rows.append(tmp)
            continue

        for lt in label_types:
            if lt not in ("go", "dir"):
                continue
            if lt == "dir" and (not dir_include_neutral):
                sub = tmp[tmp["y_raw"] != 0].copy()
            else:
                sub = tmp.copy()
            if sub.empty:
                continue

            sub["label_type"] = lt
            sub["task"] = sub["label_type"] + "_" + sub["tf"]

            buy = sub.copy()
            buy["side"] = "buy"
            buy["side_num"] = np.int8(1)
            buy["entry"] = pd.to_numeric(buy["bid_entry"], errors="coerce").astype(float)

            sell = sub.copy()
            sell["side"] = "sell"
            sell["side_num"] = np.int8(-1)
            sell["entry"] = pd.to_numeric(sell["ask_entry"], errors="coerce").astype(float)

            buy["Y"]  = (buy["y_raw"] == 1).astype("int8")
            sell["Y"] = (sell["y_raw"] == -1).astype("int8")

            if "audit_pnl_net_bps" in sub.columns:
                buy.loc[buy["Y"] != 1, "audit_pnl_net_bps"] = np.nan
                sell.loc[sell["Y"] != 1, "audit_pnl_net_bps"] = np.nan
            if "audit_exit_reason" in sub.columns:
                buy.loc[buy["Y"] != 1, "audit_exit_reason"] = "NA"
                sell.loc[sell["Y"] != 1, "audit_exit_reason"] = "NA"

            rows.append(buy)
            rows.append(sell)

    if not rows:
        return pd.DataFrame()

    ev = pd.concat(rows, ignore_index=True)
    ev["symbol"] = ev["symbol"].astype(str)
    ev["year"] = pd.to_numeric(ev["year"], errors="coerce").fillna(-1).astype(int)
    ev["ym"] = ev["t"].dt.strftime("%Y-%m")
    return ev.sort_values(["t","tf","label_type","side"]).reset_index(drop=True)


# ------------------------------
# Core processing (ultra perf bucket)
# ------------------------------
def month_dir_from_root_tpl(root_tpl: str, symbol: str, year: int) -> str:
    p = root_tpl.replace("<SYMBOL>", symbol).replace("<YEAR>", str(year)).rstrip("/")
    return p.rsplit("/", 1)[0]

def process_symbol_year(
    symbol: str,
    year: int,
    signals_root: str,
    book_root_tpl: str,
    trades_root_tpl: str,
    out_root: str,
    split: str,
    tfs_keep: List[str],
    storage_options: dict,
    label_types: List[str],
    dir_include_neutral: bool,
    legacy_single_row: bool,
    include_zero_legacy: bool,
    bucket_freq: str,
    debug: bool,
) -> int:
    t_global0 = time.time()

    sig_path = f"{signals_root.rstrip('/')}/{symbol}/{year}_signals.csv"
    log(f"[{symbol} {year}] read signals: {sig_path}")
    sig = read_csv_s3(sig_path, storage_options)
    if sig is None or sig.empty:
        log(f"[{symbol} {year}] signals empty -> skip")
        return 0

    if "symbol" not in sig.columns:
        sig["symbol"] = symbol
    if "year" not in sig.columns:
        sig["year"] = year

    found_tfs = parse_tfs_from_signals_columns(list(sig.columns))
    if tfs_keep:
        keep = set([x.lower().strip() for x in tfs_keep])
        found_tfs = [x for x in found_tfs if x in keep]
    if not found_tfs:
        log(f"[{symbol} {year}] no TF found after filter -> skip")
        return 0

    ev = make_events_long_with_label_type(
        sig=sig,
        tfs=found_tfs,
        label_types=label_types,
        dir_include_neutral=dir_include_neutral,
        legacy_single_row=legacy_single_row,
        include_zero_legacy=include_zero_legacy,
    )
    if ev is None or ev.empty:
        log(f"[{symbol} {year}] events empty -> skip")
        return 0

    out_year_dir = f"{out_root.rstrip('/')}/{split}/{symbol}/{year}"
    sink = PartsSink(out_year_dir=out_year_dir, storage_options=storage_options)

    pafs = pa_filesystem(storage_options)
    book_dir = month_dir_from_root_tpl(book_root_tpl, symbol, year)
    trades_dir = month_dir_from_root_tpl(trades_root_tpl, symbol, year)

    try:
        bucket_delta = pd.Timedelta(bucket_freq)
    except Exception:
        log(f"[{symbol} {year}] invalid --bucket={bucket_freq}, fallback to 1h")
        bucket_freq = "1h"
        bucket_delta = pd.Timedelta(hours=1)

    LOOKBACK = pd.Timedelta(seconds=60)
    FORWARD = pd.Timedelta(seconds=180)

    total_written = 0
    df_batch_parts: List[pd.DataFrame] = []
    batch_rows_acc = 0

    t_read_book = t_read_trades = t_prep = t_feat = t_write = 0.0

    log(f"[{symbol} {year}] events={len(ev):,} tfs={found_tfs} label_types={label_types} out={out_year_dir} bucket={bucket_freq}")

    for ym, evm in ev.groupby("ym"):
        t_month0 = time.time()

        book_path = f"{book_dir}/{ym}.parquet"
        trades_path = f"{trades_dir}/{ym}.parquet"

        try:
            book_dset = dataset_from_month_parquet(book_path, pafs)
            trades_dset = dataset_from_month_parquet(trades_path, pafs)
        except Exception as e:
            if debug:
                log(f"[{symbol} {year} {ym}] skip month: {e}")
            continue

        book_cache: OrderedDict[pd.Timestamp, BookPrepared] = OrderedDict()
        trades_cache: OrderedDict[pd.Timestamp, TradesPrepared] = OrderedDict()

        book_hit = book_miss = 0
        tr_hit = tr_miss = 0
        skipped = 0

        book_cols = ["received_time", "seq", "spread_top"]
        for i in range(BOOK_LEVELS):
            book_cols += [f"bid_{i}_price", f"bid_{i}_size", f"ask_{i}_price", f"ask_{i}_size"]

        evm = evm.sort_values("t").copy()
        evm["t"] = pd.to_datetime(evm["t"], utc=True, errors="coerce")
        evm = evm.dropna(subset=["t"])
        evm["bucket"] = evm["t"].dt.floor(bucket_freq)

        n_month = len(evm)
        if n_month == 0:
            continue

        wrote_before_month = total_written
        log(f"[{symbol} {year} {ym}] START month events={n_month:,} book={book_path} trades={trades_path}")

        t_last_heartbeat = time.time()
        n_done = 0

        for bucket, evb in evm.groupby("bucket", sort=True):
            n_done += len(evb)

            # --- book prep ---
            if bucket in book_cache:
                book_prep = book_cache[bucket]
                book_cache.move_to_end(bucket)
                book_hit += 1
            else:
                t0 = bucket - LOOKBACK
                t1 = bucket + bucket_delta + FORWARD

                t_rb0 = time.time()
                try:
                    book_full = read_book_window(book_dset, t0, t1, book_cols)
                except Exception:
                    book_full = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
                t_rb = time.time() - t_rb0
                t_read_book += t_rb

                t_p0 = time.time()
                book_prep = prepare_book_features(book_full)
                t_prep += (time.time() - t_p0)

                if len(book_cache) >= MAX_BOOK_CACHE:
                    book_cache.popitem(last=False)
                book_cache[bucket] = book_prep
                book_miss += 1

                if book_miss <= 3:
                    log(f"[{symbol} {year} {ym}] book_miss#{book_miss} bucket={bucket} read_book={t_rb:.3f}s rows={len(book_full):,}")

            if book_prep.df is None or book_prep.df.empty:
                skipped += len(evb)
                continue

            # --- trades prep ---
            if bucket in trades_cache:
                trades_prep = trades_cache[bucket]
                trades_cache.move_to_end(bucket)
                tr_hit += 1
            else:
                t0_tr = bucket - LOOKBACK
                t1_tr = bucket + bucket_delta + FORWARD

                t_rt0 = time.time()
                try:
                    trades_full = read_trades_window(trades_dset, t0_tr, t1_tr)
                except Exception:
                    trades_full = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
                t_rt = time.time() - t_rt0
                t_read_trades += t_rt

                t_p0 = time.time()
                trades_prep = prepare_trades_features(trades_full)
                t_prep += (time.time() - t_p0)

                if len(trades_cache) >= MAX_TRADES_CACHE:
                    trades_cache.popitem(last=False)
                trades_cache[bucket] = trades_prep
                tr_miss += 1

                if tr_miss <= 3:
                    log(f"[{symbol} {year} {ym}] trades_miss#{tr_miss} bucket={bucket} read_trades={t_rt:.3f}s rows={len(trades_full):,}")

            # --- vectorized build for this bucket ---
            t_f0 = time.time()

            evb2 = evb.sort_values("t").copy()
            if evb2.empty:
                continue

            book_df = book_prep.df.reset_index().rename(columns={"timestamp": "book_ts"})
            book_df["book_ts"] = pd.to_datetime(book_df["book_ts"], utc=True, errors="coerce")
            book_df = book_df.dropna(subset=["book_ts"]).sort_values("book_ts")
            evb2 = evb2.sort_values("t")

            merged = pd.merge_asof(
                evb2,
                book_df,
                left_on="t",
                right_on="book_ts",
                direction="backward",
                allow_exact_matches=True,
            )

            # --- staleness per tf (FIX v3.3.1: int64 ns, tz-safe) ---
            tf = merged["tf"].astype(str).str.lower().str.strip()
            max_stale_ns = tf.map(lambda x: int(MAX_STALE_BY_TF.get(x, pd.Timedelta(seconds=5)).to_timedelta64())).to_numpy()

            t_ns = pd.to_datetime(merged["t"], utc=True, errors="coerce").astype("int64").to_numpy()
            b_ns = pd.to_datetime(merged["book_ts"], utc=True, errors="coerce").astype("int64").to_numpy()
            dt_ns = t_ns - b_ns

            ok_stale = dt_ns <= max_stale_ns

            bid0 = pd.to_numeric(merged.get("bid_0_price"), errors="coerce").astype(float)
            ask0 = pd.to_numeric(merged.get("ask_0_price"), errors="coerce").astype(float)
            ok_px = np.isfinite(bid0) & np.isfinite(ask0) & (bid0 > 0) & (ask0 > 0)

            merged = merged[ok_stale & ok_px].copy()
            if merged.empty:
                skipped += len(evb2)
                t_feat += (time.time() - t_f0)
                continue

            bf = build_book_features_batch(merged)

            bf["quote_churn_10s"] = _asof_values(book_prep.churn10, merged["t"])
            bf["ret_stdev_1s_10s_bps"] = _asof_values(book_prep.retstd10_bps, merged["t"])

            agr3  = _asof_values(trades_prep.aggr3, merged["t"])
            agr5  = _asof_values(trades_prep.aggr5, merged["t"])
            agr10 = _asof_values(trades_prep.aggr10, merged["t"])
            agr15 = _asof_values(trades_prep.aggr15, merged["t"])

            bf["aggr_ratio_10s"] = agr10
            bf["aggr_ratio_15s"] = agr15
            bf["bt_dom_3s"] = agr3
            bf["bt_dom_5s"] = agr5
            bf["bt_dom_10s"] = agr10

            side = merged["side"].astype(str).values
            mid = bf["mid_t"].values.astype(float)

            def mat(prefix, field, K):
                cols = [f"{prefix}_{i}_{field}" for i in range(K)]
                return np.vstack([
                    pd.to_numeric(merged.get(c), errors="coerce").astype(float).fillna(0.0).values
                    for c in cols
                ]).T

            ask_sz_5 = mat("ask", "size", 5)
            bid_sz_5 = mat("bid", "size", 5)
            ask_sz_15 = mat("ask", "size", 15)
            bid_sz_15 = mat("bid", "size", 15)

            ask_px_15 = np.vstack([pd.to_numeric(merged.get(f"ask_{i}_price"), errors="coerce").astype(float).values for i in range(15)]).T
            bid_px_15 = np.vstack([pd.to_numeric(merged.get(f"bid_{i}_price"), errors="coerce").astype(float).values for i in range(15)]).T

            def wall_share(sz_mat):
                tot = np.sum(sz_mat, axis=1)
                mx = np.max(sz_mat, axis=1)
                return np.where(tot > 0, mx / tot, np.nan)

            is_buy = (side == "buy")
            bf["wall_opp_share_5"] = np.where(is_buy, wall_share(ask_sz_5), wall_share(bid_sz_5))
            bf["wall_opp_share_15"] = np.where(is_buy, wall_share(ask_sz_15), wall_share(bid_sz_15))

            def depth_within_bps(px_mat, sz_mat, mid, bps):
                tol = (bps / 1e4) * mid
                ok = np.isfinite(px_mat) & np.isfinite(mid[:, None]) & (np.abs(px_mat - mid[:, None]) <= tol[:, None])
                return np.sum(np.where(ok, sz_mat, 0.0), axis=1)

            depth_ask_5bps = depth_within_bps(ask_px_15, ask_sz_15, mid, 5.0)
            depth_ask_10bps = depth_within_bps(ask_px_15, ask_sz_15, mid, 10.0)
            depth_bid_5bps = depth_within_bps(bid_px_15, bid_sz_15, mid, 5.0)
            depth_bid_10bps = depth_within_bps(bid_px_15, bid_sz_15, mid, 10.0)

            bf["cum_depth_within_5bps_opp"] = np.where(is_buy, depth_ask_5bps, depth_bid_5bps)
            bf["cum_depth_within_10bps_opp"] = np.where(is_buy, depth_ask_10bps, depth_bid_10bps)

            bid_px_5 = np.vstack([pd.to_numeric(merged.get(f"bid_{i}_price"), errors="coerce").astype(float).values for i in range(5)]).T
            ask_px_5 = np.vstack([pd.to_numeric(merged.get(f"ask_{i}_price"), errors="coerce").astype(float).values for i in range(5)]).T

            bf["slope_bid_5"] = _slope_batch(bid_px_5, bid_sz_5)
            bf["slope_ask_5"] = _slope_batch(ask_px_5, ask_sz_5)
            bf["slope_bid_15"] = _slope_batch(bid_px_15, bid_sz_15)
            bf["slope_ask_15"] = _slope_batch(ask_px_15, ask_sz_15)

            side_num = pd.to_numeric(merged.get("side_num"), errors="coerce").fillna(0).astype(int).values
            def SA(arr):
                arr = arr.astype(float)
                out = np.full_like(arr, np.nan, dtype=float)
                ok = (side_num == 1) | (side_num == -1)
                out[ok] = side_num[ok] * arr[ok]
                return out

            bf["obi_5_side"] = SA(bf["obi_5"].values.astype(float))
            bf["obi_15_side"] = SA(bf["obi_15"].values.astype(float))
            bf["microprice_bias_side"] = SA(bf["microprice_bias"].values.astype(float))
            bf["aggr_ratio_10s_side"] = SA(np.where(np.isfinite(agr10), 2.0 * agr10 - 1.0, np.nan))
            bf["aggr_ratio_15s_side"] = SA(np.where(np.isfinite(agr15), 2.0 * agr15 - 1.0, np.nan))
            bf["bt_dom_3s_side"] = SA(np.where(np.isfinite(agr3), 2.0 * agr3 - 1.0, np.nan))
            bf["bt_dom_10s_side"] = SA(np.where(np.isfinite(agr10), 2.0 * agr10 - 1.0, np.nan))

            n = len(merged)
            out_df = pd.DataFrame({
                "t": merged["t"].values,
                "symbol": merged["symbol"].astype(str).values,
                "year": pd.to_numeric(merged["year"], errors="coerce").fillna(-1).astype(int).values,
                "tf": merged["tf"].astype(str).values,
                "label_type": merged["label_type"].astype(str).values,
                "task": merged.get("task", "NA").astype(str).values,
                "y_raw": pd.to_numeric(merged["y_raw"], errors="coerce").fillna(0).astype(int).values,
                "side": merged["side"].astype(str).values,
                "side_num": side_num,
                "entry": pd.to_numeric(merged["entry"], errors="coerce").astype(float).values,
                "bid_entry": pd.to_numeric(merged["bid_entry"], errors="coerce").astype(float).values,
                "ask_entry": pd.to_numeric(merged["ask_entry"], errors="coerce").astype(float).values,
                "spread_bps_entry": pd.to_numeric(merged["spread_bps_entry"], errors="coerce").astype(float).values,
                "atr_bps": pd.to_numeric(merged["atr_bps"], errors="coerce").astype(float).values,
                "Y": pd.to_numeric(merged["Y"], errors="coerce").fillna(0).astype(int).values,
                "audit_pnl_net_bps": pd.to_numeric(_col_or_default(merged, "audit_pnl_net_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_exit_reason": _col_or_default(merged, "audit_exit_reason", "NA", n).astype(str).values,
                "audit_tp_bps": pd.to_numeric(_col_or_default(merged, "audit_tp_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_sl_bps": pd.to_numeric(_col_or_default(merged, "audit_sl_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_rr_min": pd.to_numeric(_col_or_default(merged, "audit_rr_min", np.nan, n), errors="coerce").astype(float).values,
                "audit_risk_r_bps": pd.to_numeric(_col_or_default(merged, "audit_risk_r_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_fill_mode": _col_or_default(merged, "audit_fill_mode", "NA", n).astype(str).values,
                "audit_fill_window_sec": pd.to_numeric(_col_or_default(merged, "audit_fill_window_sec", np.nan, n), errors="coerce").astype(float).values,
            })

            out_df = pd.concat([out_df, bf.reset_index(drop=True)], axis=1)

            df_batch_parts.append(out_df)
            batch_rows_acc += len(out_df)
            total_written += len(out_df)

            t_feat += (time.time() - t_f0)

            if batch_rows_acc >= BATCH_MAX:
                t_w0 = time.time()
                sink.write_df(pd.concat(df_batch_parts, ignore_index=True))
                t_write += (time.time() - t_w0)
                df_batch_parts.clear()
                batch_rows_acc = 0
                gc.collect()

            now = time.time()
            if now - t_last_heartbeat > 30:
                log(
                    f"[{symbol} {year} {ym}] heartbeat done={n_done:,}/{n_month:,} "
                    f"wrote_month={(total_written - wrote_before_month):,} wrote_total={total_written:,} skipped={skipped:,} | "
                    f"book hit/miss={book_hit}/{book_miss} trades hit/miss={tr_hit}/{tr_miss} | "
                    f"timing(s): read_book={t_read_book:.1f} read_trades={t_read_trades:.1f} prep={t_prep:.1f} feat={t_feat:.1f} write={t_write:.1f}"
                )
                t_last_heartbeat = now

        if df_batch_parts:
            t_w0 = time.time()
            sink.write_df(pd.concat(df_batch_parts, ignore_index=True))
            t_write += (time.time() - t_w0)
            df_batch_parts.clear()
            batch_rows_acc = 0

        dtm = time.time() - t_month0
        wrote_month = total_written - wrote_before_month
        rate = (n_month / dtm) if dtm > 0 else 0.0
        log(
            f"[{symbol} {year} {ym}] DONE month events={n_month:,} wrote_month={wrote_month:,} "
            f"wrote_total={total_written:,} skipped={skipped:,} time={dtm:.1f}s rate={rate:.1f} ev/s | "
            f"book hit/miss={book_hit}/{book_miss} trades hit/miss={tr_hit}/{tr_miss}"
        )

    if df_batch_parts:
        t_w0 = time.time()
        sink.write_df(pd.concat(df_batch_parts, ignore_index=True))
        t_write += (time.time() - t_w0)
        df_batch_parts.clear()

    dt = time.time() - t_global0
    log(
        f"[{symbol} {year}] DONE rows_written={total_written:,} total_time={dt:.1f}s (~{(total_written/dt if dt>0 else 0):.1f} rows/s) | "
        f"timing_total(s): read_book={t_read_book:.1f} read_trades={t_read_trades:.1f} prep={t_prep:.1f} feat={t_feat:.1f} write={t_write:.1f}"
    )
    return total_written


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser("Stage2 v3.3 — perf+logs fixed (v3.3.1 patched)")

    p.add_argument("--signals-root", default="s3://tradebot-config-tokyo/data/stage1")
    p.add_argument("--book-root", default="s3://tradebot-config-tokyo/data/book/<SYMBOL>/<YEAR>-*.parquet")
    p.add_argument("--trades-root", default="s3://tradebot-config-tokyo/data/trade/<SYMBOL>/<YEAR>-*.parquet")
    p.add_argument("--out-root", default="s3://tradebot-config-tokyo/data/stage2")

    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)

    p.add_argument("--tfs", nargs="+", default=None)
    p.add_argument("--split", choices=["train","val","test","all"], default="all")

    p.add_argument("--s3-region", default="ap-northeast-1")
    p.add_argument("--s3-anon", action="store_true")

    p.add_argument("--label-types", nargs="+", default=["go","dir"], choices=["go","dir"])
    p.add_argument("--dir-include-neutral", action="store_true")

    p.add_argument("--legacy-single-row", action="store_true")
    p.add_argument("--include-zero", action="store_true")

    p.add_argument("--bucket", default="1h", help="Cache bucket size, e.g. 1h (default), 30min, 15min, 5min")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    storage_options = so(args.s3_region, args.s3_anon)

    tfs_keep = []
    if args.tfs:
        tfs_keep = [str(x).strip().lower() for x in args.tfs if str(x).strip()]

    label_types = [str(x).strip().lower() for x in (args.label_types or [])]
    label_types = [x for x in label_types if x in ("go","dir")]
    if not label_types and not args.legacy_single_row:
        label_types = ["go","dir"]

    for sym in args.symbols:
        for year in args.years:
            n = process_symbol_year(
                symbol=sym,
                year=year,
                signals_root=args.signals_root,
                book_root_tpl=args.book_root,
                trades_root_tpl=args.trades_root,
                out_root=args.out_root,
                split=args.split,
                tfs_keep=tfs_keep,
                storage_options=storage_options,
                label_types=label_types,
                dir_include_neutral=bool(args.dir_include_neutral),
                legacy_single_row=bool(args.legacy_single_row),
                include_zero_legacy=bool(args.include_zero),
                bucket_freq=str(args.bucket),
                debug=args.debug,
            )
            log(f"[DONE] {sym} {year} rows_written={n} split={args.split}")

if __name__ == "__main__":
    main()