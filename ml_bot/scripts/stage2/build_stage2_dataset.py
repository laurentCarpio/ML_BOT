#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_stage2_dataset.py — Stage2 v4.1 (GO+DIR dual outputs, no Y) — GO wide + DIR wide

- Stage2 unique, 2 sorties:
  - out_root/.../go/<SYMBOL>/<YEAR>/parts/*.parquet
  - out_root/.../dir/<SYMBOL>/<YEAR>/parts/*.parquet
- PAS de colonne Y produite en stage2.
- Labels conservés:
  - y_go  (0/1)      : gate tradeable
  - y_dir (-1/0/+1)  : direction (0 = neutre / pas de direction)
- GO dataset = 1 ligne par (row_id,t,tf) avec features buy & sell en colonnes séparées.
- DIR dataset = même format MAIS filtré sur y_dir in {-1,+1}.
- NEW (bretelles):
  - task ∈ {"go","dir"}
  - event_id = f"{row_id}|{task}|{tf}"
- Stage3 (go/dir) fabriquera Y juste avant stage4 + normalisation.
"""

from __future__ import annotations

import argparse
import re
import gc
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List
import json
import fsspec

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

TF_CFG = {
  "15m": {"W": 0.18, "L": 0.60, "C": 0.80, "wA": 3.0, "lA": 0.70, "chopB": 0.85, "k_eps30": 1.0, "k_eps120": 1.5},
  "30m": {"W": 0.20, "L": 0.60, "C": 0.80, "wA": 3.0, "lA": 0.70, "chopB": 0.85, "k_eps30": 1.0, "k_eps120": 1.5},
  "1h":  {"W": 0.22, "L": 0.55, "C": 0.82, "wA": 3.0, "lA": 0.70, "chopB": 0.86, "k_eps30": 1.0, "k_eps120": 1.5},
  "2h":  {"W": 0.25, "L": 0.55, "C": 0.84, "wA": 3.0, "lA": 0.70, "chopB": 0.87, "k_eps30": 1.0, "k_eps120": 1.6},
  "4h":  {"W": 0.30, "L": 0.50, "C": 0.86, "wA": 3.0, "lA": 0.70, "chopB": 0.88, "k_eps30": 1.0, "k_eps120": 1.6},
}

MAX_STALE_BY_TF = {
    "5m":  pd.Timedelta(seconds=1),
    "15m": pd.Timedelta(seconds=2),
    "30m": pd.Timedelta(seconds=3),
    "1h":  pd.Timedelta(seconds=5),
    "2h":  pd.Timedelta(seconds=6),
    "4h":  pd.Timedelta(seconds=10),
}

CANDLE_LOOKBACK = pd.Timedelta(days=8)
CANDLE_FORWARD  = pd.Timedelta(minutes=5)

BB_N = 20
BB_K = 2.0
ATR_N = 14
ADX_N = 14

MAX_BOOK_CACHE = 128
MAX_TRADES_CACHE = 256
MAX_CANDLE_CACHE = 32

BATCH_MAX = 10_000

LOG_NEG_SPREAD_MAX_EX = 5
DEDUP_BOOK_TS = True
RECALC_SPREAD_ENTRY = False

_HIT_STEP_RE = re.compile(r"^audit_hit_step_[pm]\d+R_[LS]$")
_HIT_FLAG_RE = re.compile(r"^audit_hit_[pm]\d+R_[LS]$") 

def _cast_audits_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    df contient des colonnes audit_* déjà sans suffixe tf.
    IMPORTANT: on force hit_step en int32 avec -1 par défaut.
    """
    for c in list(df.columns):
        if c in ("audit_exit_reason", "audit_fill_mode"):
            df[c] = df[c].astype(str)
            continue

        if _HIT_STEP_RE.match(c):
            # must be int with -1 when missing
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.fillna(-1).astype(np.int32)
            df[c] = s
            continue

        if _HIT_FLAG_RE.match(c):
            s = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)
            df[c] = s
            continue

        if c.startswith("audit_first_touch_step_") and c.endswith(("_L", "_S")):
            s = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int32)
            df[c] = s
            continue

        if c.startswith("audit_first_touch_") and c.endswith(("_L", "_S")):
            s = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)
            df[c] = s
            continue

        # le reste: float (mfe_R/mae_R/p_thr/etc)
        if c.startswith("audit_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    return df

# ------------------------------
# Column grouping helpers
# ------------------------------
META_COLS_FIXED = [
    "event_id", "task", "row_id", "t", "symbol", "year", "tf",
    "y_go", "y_dir",
    "bid_entry", "ask_entry", "spread_bps_entry", "atr_bps",
]

AUDIT_PREFIXES = ("audit_",)  # Stage1 audits + Stage2 flags
SUP_PREFIXES = ("sup_",)      # oracle post-entry supervision (NEVER in X)


def reorder_stage2_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order: meta | audits | sup | features (everything else)
    Makes it trivial for Stage3 to keep audits/sup but exclude from X.
    """
    cols = list(df.columns)

    meta = [c for c in META_COLS_FIXED if c in cols]

    audits = [c for c in cols if c.startswith(AUDIT_PREFIXES)]
    sups   = [c for c in cols if c.startswith(SUP_PREFIXES)]

    # remove duplicates while preserving order
    seen = set()
    def uniq(xs):
        out = []
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    meta_u = uniq(meta)
    audits_u = uniq(audits)
    sups_u = uniq(sups)

    rest = [c for c in cols if c not in seen]  # features

    return df[meta_u + audits_u + sups_u + rest]

# ------------------------------
# Logging
# ------------------------------
def log(msg: str):
    print(f"{pd.Timestamp.utcnow().isoformat()}Z | {msg}", flush=True)

def assert_has_cols(df: pd.DataFrame, cols: list[str], *, name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss[:20]} (total {len(miss)})")

def assert_schema_has(dataset: ds.Dataset, cols: list[str], *, name: str):
    names = set(dataset.schema.names)
    miss = [c for c in cols if c not in names]
    if miss:
        raise ValueError(f"{name} parquet schema missing: {miss[:20]} (total {len(miss)})")
    
# ------------------------------
# S3 / Arrow helpers
# ------------------------------
def read_json_s3(path: str, storage_options: dict) -> dict:
    with fsspec.open(path, "rb", **(storage_options or {})) as f:
        return json.load(f)

def tfs_from_stage1_columns(stage1_cols: dict) -> list[str]:
    # tf_cols contient "horizon_sec_<tf>" etc.
    tfs = set()
    for c in stage1_cols.get("tf_cols", []):
        c = str(c)
        if c.startswith("horizon_sec_"):
            tfs.add(c[len("horizon_sec_"):])
    return sorted(tfs, key=TF_SORT_KEY)

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

def candle_path_from_tpl(root_tpl: str, symbol: str, year: int) -> str:
    return ensure_s3(root_tpl.replace("<SYMBOL>", symbol).replace("<YEAR>", str(year)))

def read_candles_window(candle_path_s3: str, pafs: pa_fs.FileSystem,
                        t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    dset = ds.dataset([s3_to_bucket_key(candle_path_s3)], format="parquet", filesystem=pafs)

    ts_field = "timestamp"
    names = set(dset.schema.names)
    if ts_field not in names:
        raise ValueError("Candles parquet missing 'timestamp'")

    cols = [c for c in ["timestamp","open","high","low","close","volume"] if c in names]
    flt = arrow_time_filter(dset, ts_field, t0, t1)
    tbl = dset.to_table(columns=cols, filter=flt)
    if tbl.num_rows == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))

    df = tbl.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    df.index.name = "timestamp"

    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    need = ["open","high","low","close"]
    df = df.dropna(subset=[c for c in need if c in df.columns])
    return df

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

    df = tbl.to_pandas()
    df[ts_field] = pd.to_datetime(df[ts_field], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_field])

    sort_cols = [ts_field]
    if "seq" in df.columns:
        sort_cols.append("seq")
    if "received_time" in df.columns:
        try:
            df["received_time"] = pd.to_datetime(df["received_time"], utc=True, errors="coerce")
        except Exception:
            pass
        sort_cols.append("received_time")

    df = df.sort_values(sort_cols, kind="mergesort")

    if DEDUP_BOOK_TS:
        df = df.drop_duplicates(subset=[ts_field], keep="last")

    df = df.set_index(ts_field)
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
# Candle feature helpers
# ------------------------------
def _tf_to_pandas_freq(tf: str) -> str:
    tf = (tf or "").lower().strip()
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{int(tf[:-1])}min"
    if tf.endswith("h") and tf[:-1].isdigit():
        return f"{int(tf[:-1])}h"
    return tf

def _wilder_ewm(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()

def compute_bb_width(close: pd.Series, n: int = BB_N, k: float = BB_K) -> pd.Series:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return (upper - lower) / ma.replace(0, np.nan)

def compute_tr_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = ATR_N):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = _wilder_ewm(tr, n)
    return tr, atr

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = ADX_N) -> pd.Series:
    up = high.diff()
    dn = -low.diff()

    plus_dm  = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)

    tr, _ = compute_tr_atr(high, low, close, n)
    tr_s = _wilder_ewm(tr, n).replace(0, np.nan)

    plus_di  = 100.0 * (_wilder_ewm(plus_dm, n)  / tr_s)
    minus_di = 100.0 * (_wilder_ewm(minus_dm, n) / tr_s)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_ewm(dx, n).clip(0.0, 100.0)
    return adx

def rolling_percentile_of_last(x: np.ndarray) -> float:
    if x.size == 0 or not np.isfinite(x[-1]):
        return np.nan
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.nan
    last = x[-1]
    return float((v <= last).mean())

def compute_candle_features_for_tfs(candles_1m: pd.DataFrame, tfs: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if candles_1m is None or candles_1m.empty:
        return out

    tfs_norm = sorted(set([str(x).lower().strip() for x in tfs if x]))

    for tf in tfs_norm:
        tf_freq = _tf_to_pandas_freq(tf)

        ohlc = candles_1m[["open", "high", "low", "close", "volume"]].resample(
            tf_freq, label="right", closed="right"
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open", "high", "low", "close"])
        if ohlc.empty:
            continue

        close = ohlc["close"]
        high = ohlc["high"]
        low = ohlc["low"]

        bb_width = compute_bb_width(close, BB_N, BB_K)
        _, atr = compute_tr_atr(high, low, close, ATR_N)
        adx = compute_adx(high, low, close, ADX_N)

        try:
            tf_delta = pd.Timedelta(tf_freq)
        except Exception:
            tf_delta = pd.Timedelta(hours=1)

        win = int(pd.Timedelta(days=7) / tf_delta) if tf_delta > pd.Timedelta(0) else 168
        win = max(win, 50)

        bb_pctl = bb_width.rolling(win, min_periods=max(10, win // 10)).apply(rolling_percentile_of_last, raw=True)
        atr_pctl = atr.rolling(win, min_periods=max(10, win // 10)).apply(rolling_percentile_of_last, raw=True)

        df_feat = pd.DataFrame(index=ohlc.index)
        df_feat["bb_width"] = bb_width
        df_feat["bb_width_pctl"] = bb_pctl
        df_feat["lf_bb_width_pct"] = bb_pctl
        df_feat["atr_percentile"] = atr_pctl

        if tf_freq == "30min":
            df_feat["lf_atr_rank_30m"] = atr_pctl
            df_feat["atr_pct_rank_30m"] = atr_pctl
        else:
            df_feat["lf_atr_rank_30m"] = np.nan
            df_feat["atr_pct_rank_30m"] = np.nan

        df_feat["adx"] = adx

        key = "30m" if tf_freq == "30min" else tf
        out[key] = df_feat

    return out

def attach_candle_features(merged_ev: pd.DataFrame, candles_1m: pd.DataFrame) -> pd.DataFrame:
    if merged_ev is None or merged_ev.empty:
        return merged_ev

    merged_ev = merged_ev.copy()
    merged_ev["t"] = pd.to_datetime(merged_ev["t"], utc=True, errors="coerce")
    merged_ev = merged_ev.dropna(subset=["t"])
    if merged_ev.empty:
        return merged_ev

    merged_ev["tf"] = merged_ev["tf"].astype(str).str.lower().str.strip()
    tfs = merged_ev["tf"].dropna().unique().tolist()
    feats_by_tf = compute_candle_features_for_tfs(candles_1m, tfs + ["30m"])
    feat_30m = feats_by_tf.get("30m")

    parts = []
    for tf, g in merged_ev.groupby("tf", sort=False):
        g = g.sort_values("t")
        fdf = feats_by_tf.get(tf)

        if fdf is None or fdf.empty:
            for c in ["bb_width", "bb_width_pctl", "lf_bb_width_pct",
                      "atr_percentile", "lf_atr_rank_30m", "atr_pct_rank_30m", "adx"]:
                g[c] = np.nan
            parts.append(g)
            continue

        f = fdf.reset_index().rename(columns={"timestamp": "c_ts"}).sort_values("c_ts")
        g2 = pd.merge_asof(g, f, left_on="t", right_on="c_ts",
                           direction="backward", allow_exact_matches=True).drop(columns=["c_ts"], errors="ignore")

        if tf not in ("30m", "30min") and feat_30m is not None and not feat_30m.empty:
            f30 = feat_30m[["lf_atr_rank_30m", "atr_pct_rank_30m"]].reset_index() \
                .rename(columns={"timestamp": "c30_ts"}).sort_values("c30_ts")

            g2 = pd.merge_asof(
                g2, f30,
                left_on="t", right_on="c30_ts",
                direction="backward", allow_exact_matches=True,
                suffixes=("", "_30m")
            ).drop(columns=["c30_ts"], errors="ignore")

            for c in ["lf_atr_rank_30m", "atr_pct_rank_30m"]:
                c30 = c + "_30m"
                if c30 in g2.columns:
                    g2[c] = g2[c].where(g2[c].notna(), g2[c30])
                    g2.drop(columns=[c30], inplace=True, errors="ignore")

        parts.append(g2)

    return pd.concat(parts, ignore_index=True)

def _check_schema(df: pd.DataFrame, schema_cols, name: str):
    cols = list(df.columns)
    if schema_cols is None:
        return cols  # first time becomes reference
    if cols != schema_cols:
        # diff readable
        s = set(schema_cols); c = set(cols)
        added = sorted(list(c - s))[:30]
        removed = sorted(list(s - c))[:30]
        raise ValueError(f"{name} schema drift: added={added} removed={removed}")
    return schema_cols
# ------------------------------
# Prepared caches
# ------------------------------
@dataclass
class BookPrepared:
    df: pd.DataFrame
    churn10: pd.Series
    retstd10_bps: pd.Series

@dataclass
class TradesPrepared:
    df: pd.DataFrame
    aggr3: pd.Series
    aggr5: pd.Series
    aggr10: pd.Series
    aggr15: pd.Series
    ntr_30s: pd.Series
    vol_30s: pd.Series
    ntr_2m: pd.Series
    vol_2m: pd.Series
    ntr_5m: pd.Series
    vol_5m: pd.Series
    last_trade_age_s: pd.Series

def prepare_book_features(book_full: pd.DataFrame) -> BookPrepared:
    if book_full is None or book_full.empty:
        empty = pd.Series(dtype=float)
        return BookPrepared(book_full, empty, empty)

    df = book_full.sort_index() if not book_full.index.is_monotonic_increasing else book_full

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

def safe_div(num, den, default=np.nan):
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")

    if isinstance(n, pd.Series) or isinstance(d, pd.Series):
        # conserve index
        n = pd.Series(n, index=getattr(num, "index", None))
        d = pd.Series(d, index=getattr(den, "index", None))
        out = n / d
        bad = (~np.isfinite(out)) | (~np.isfinite(d)) | (d <= 0)
        out = out.astype(float)
        out[bad] = default
        return out

    out = np.asarray(n, dtype=float) / np.asarray(d, dtype=float)
    d2 = np.asarray(d, dtype=float)
    bad = (~np.isfinite(out)) | (~np.isfinite(d2)) | (d2 <= 0)
    out = out.astype(float)
    out[bad] = default
    return out

def prepare_trades_features(trades_full: pd.DataFrame) -> TradesPrepared:
    if trades_full is None or trades_full.empty or "is_aggr_buy" not in trades_full.columns:
        empty = pd.Series(dtype=float)
        return TradesPrepared(trades_full, empty, empty, empty, empty,
                              empty, empty, empty, empty, empty, empty, empty)

    df = trades_full.sort_index() if not trades_full.index.is_monotonic_increasing else trades_full

    qty = pd.to_numeric(df.get("qty"), errors="coerce").astype(float).fillna(0.0)
    is_buy = df["is_aggr_buy"].astype("boolean")
    buy_qty = qty.where(is_buy == True, 0.0)

    def r(window: str) -> pd.Series:
        buyw = buy_qty.rolling(window).sum()
        totw = qty.rolling(window).sum()
        out = safe_div(buyw, totw, default=0.5).astype(float)
        out = np.clip(out, 0.0, 1.0)
        return out

    # --- intensity features ---
    ones = pd.Series(1.0, index=df.index)
    ntr_30s = ones.rolling("30s").sum()
    vol_30s = qty.rolling("30s").sum()
    ntr_2m  = ones.rolling("2min").sum()
    vol_2m  = qty.rolling("2min").sum()
    ntr_5m  = ones.rolling("5min").sum()
    vol_5m  = qty.rolling("5min").sum()

    # last_trade_age_s: age du dernier trade connu au timestamp t
    # on construit une série "timestamp en ns" et on forward-fill
    ts_ns = pd.Series(df.index.asi8, index=df.index).astype("int64")
    # age au temps courant = (t - last_trade_ts)
    # => on laisse Stage2 faire le asof sur ts_ns, puis convertir en secondes
    # mais on peut pré-calculer une série age=0 sur chaque trade, puis rolling min n'a pas de sens
    # donc on expose plutôt last_trade_ts_ns:
    last_trade_ts_ns = ts_ns

    return TradesPrepared(
        df=df,
        aggr3=r("3s"), aggr5=r("5s"), aggr10=r("10s"), aggr15=r("15s"),
        ntr_30s=ntr_30s.astype(float), vol_30s=vol_30s.astype(float),
        ntr_2m=ntr_2m.astype(float),   vol_2m=vol_2m.astype(float),
        ntr_5m=ntr_5m.astype(float),   vol_5m=vol_5m.astype(float),
        last_trade_age_s=last_trade_ts_ns.astype(float),  # temporaire: ts_ns
    )

def _asof_values(series: pd.Series, t: pd.Series) -> np.ndarray:
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

    idx = s.index.asi8
    pos = np.searchsorted(idx, tv, side="right") - 1

    out = np.full(len(t), np.nan, dtype=float)
    ok = ok_t & (pos >= 0)
    if ok.any():
        vals = s.to_numpy(dtype=float, copy=False)
        out[ok] = vals[pos[ok]]
    return out

# ------------------------------
# Supervision (POST-entry) — oracle-safe columns sup_*
# ------------------------------

SUP_HORIZONS_SEC = [5, 30, 120]  # horizons de supervision, ajuste si besoin
SUP_EPS = 1e-12

def _asof_forward_values(series: pd.Series, t: pd.Series) -> np.ndarray:
    """
    Lookup forward (first value at or after t).
    series: index timestamp ascending.
    t: event timestamps
    Returns array aligned with t (NaN if no future point).
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
    pos = np.searchsorted(idx, tv, side="left")  # forward

    out = np.full(len(t), np.nan, dtype=float)
    ok = ok_t & (pos >= 0) & (pos < len(idx))
    if ok.any():
        vals = s.to_numpy(dtype=float, copy=False)
        out[ok] = vals[pos[ok]]
    return out

def compute_supervision_features_post_entry(
    merged: pd.DataFrame,
    book_prep: "BookPrepared",
    horizons_sec: list[int] = None,
) -> pd.DataFrame:
    if horizons_sec is None:
        horizons_sec = SUP_HORIZONS_SEC

    out = pd.DataFrame(index=merged.index)
    t = pd.to_datetime(merged["t"], utc=True, errors="coerce")

    bdf = book_prep.df
    if bdf is None or bdf.empty:
        out["sup_mid_now"] = np.nan
        out["sup_spread_bps_now"] = np.nan
        out["sup_liq_top_sum_now"] = np.nan

        for h in horizons_sec:
            out[f"sup_mid_fwd_{h}s"] = np.nan
            out[f"sup_spread_bps_fwd_{h}s"] = np.nan
            out[f"sup_liq_top_sum_fwd_{h}s"] = np.nan
            out[f"sup_ret_bps_fwd_{h}s"] = np.nan

        out["sup_liq_drop_ratio_5s"] = np.nan
        out["sup_spread_widen_bps_5s"] = np.nan
        out["sup_chop_score_120s"] = np.nan
        return out

    # ---- book series (future-aware for forward lookups) ----
    bid0 = pd.to_numeric(bdf.get("bid_0_price"), errors="coerce").astype("float64")
    ask0 = pd.to_numeric(bdf.get("ask_0_price"), errors="coerce").astype("float64")
    bsz0 = pd.to_numeric(bdf.get("bid_0_size"), errors="coerce").astype("float64")
    asz0 = pd.to_numeric(bdf.get("ask_0_size"), errors="coerce").astype("float64")

    mid_s = ((bid0 + ask0) / 2.0).replace([np.inf, -np.inf], np.nan)
    spread_bps_s = (np.where(mid_s > 0, 1e4 * (ask0 - bid0) / mid_s, np.nan)).astype("float64")
    liq_top_s = (bsz0.fillna(0.0) + asz0.fillna(0.0)).astype("float64")

    # ---- NOW reference: prefer merged snapshot (same as model features) ----
    have_merged_book = all(c in merged.columns for c in ["bid_0_price", "ask_0_price", "bid_0_size", "ask_0_size"])

    if have_merged_book:
        bid_now_s = pd.to_numeric(merged["bid_0_price"], errors="coerce").astype("float64")
        ask_now_s = pd.to_numeric(merged["ask_0_price"], errors="coerce").astype("float64")
        bsz_now_s = pd.to_numeric(merged["bid_0_size"], errors="coerce").astype("float64")
        asz_now_s = pd.to_numeric(merged["ask_0_size"], errors="coerce").astype("float64")

        mid_now = ((bid_now_s + ask_now_s) / 2.0).to_numpy(dtype="float64", copy=False)

        spr_now = np.where(
            np.isfinite(mid_now) & (mid_now > 0)
            & np.isfinite(bid_now_s.to_numpy(dtype="float64", copy=False))
            & np.isfinite(ask_now_s.to_numpy(dtype="float64", copy=False)),
            1e4 * (ask_now_s.to_numpy(dtype="float64", copy=False) - bid_now_s.to_numpy(dtype="float64", copy=False)) / mid_now,
            np.nan
        ).astype("float64")

        liq_now = (bsz_now_s.fillna(0.0) + asz_now_s.fillna(0.0)).to_numpy(dtype="float64", copy=False)
    else:
        # fallback: use backward-asof from book series at event time
        mid_now = _asof_values(mid_s, t)
        spr_now = _asof_values(pd.Series(spread_bps_s, index=bdf.index), t)
        liq_now = _asof_values(liq_top_s, t)

    out["sup_mid_now"] = mid_now
    out["sup_spread_bps_now"] = spr_now
    out["sup_liq_top_sum_now"] = liq_now

    # ---- forward metrics ----
    for h in horizons_sec:
        th = t + pd.to_timedelta(int(h), unit="s")
        mid_f = _asof_forward_values(mid_s, th)
        spr_f = _asof_forward_values(pd.Series(spread_bps_s, index=bdf.index), th)
        liq_f = _asof_forward_values(liq_top_s, th)

        out[f"sup_mid_fwd_{h}s"] = mid_f
        out[f"sup_spread_bps_fwd_{h}s"] = spr_f
        out[f"sup_liq_top_sum_fwd_{h}s"] = liq_f

        ret = 1e4 * (mid_f / np.maximum(mid_now, SUP_EPS) - 1.0)
        ret[~np.isfinite(ret)] = np.nan
        ret = np.clip(ret, -500, 500)  # garde-fou
        out[f"sup_ret_bps_fwd_{h}s"] = ret

    # ---- derived heuristics ----
    if 5 in horizons_sec:
        liq5 = out["sup_liq_top_sum_fwd_5s"].to_numpy(dtype="float64")
        out["sup_liq_drop_ratio_5s"] = np.where(
            np.isfinite(liq_now) & (liq_now > 0) & np.isfinite(liq5),
            liq5 / np.maximum(liq_now, SUP_EPS),
            np.nan
        )
        spr5 = out["sup_spread_bps_fwd_5s"].to_numpy(dtype="float64")
        out["sup_spread_widen_bps_5s"] = np.where(
            np.isfinite(spr_now) & np.isfinite(spr5),
            spr5 - spr_now,
            np.nan
        )
    else:
        out["sup_liq_drop_ratio_5s"] = np.nan
        out["sup_spread_widen_bps_5s"] = np.nan

    if 30 in horizons_sec and 120 in horizons_sec:
        r30 = out["sup_ret_bps_fwd_30s"].to_numpy(dtype=float)
        r120 = out["sup_ret_bps_fwd_120s"].to_numpy(dtype=float)

        chop = np.full(len(out), 1.0, dtype="float64")  # défaut: choppy => fail gate
        ok = np.isfinite(r30) & np.isfinite(r120) & (np.abs(r30) > 0.0)
        chop[ok] = (1.0 - (np.abs(r120[ok]) / np.maximum(np.abs(r30[ok]), SUP_EPS)))
        out["sup_chop_score_120s"] = np.clip(chop, 0.0, 1.0)
    else:
        out["sup_chop_score_120s"] = 1.0

    return out

def _slope_batch(prices: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    N, K = prices.shape
    out = np.full(N, np.nan, dtype=float)
    for i in range(N):
        p = prices[i]; q = sizes[i]
        ok = np.isfinite(p) & np.isfinite(q) & (q > 0)
        if ok.sum() < 2:
            continue
        p2 = p[ok]; q2 = q[ok]
        x = np.cumsum(q2)
        xm = x.mean(); pm = p2.mean()
        den = np.sum((x - xm) ** 2)
        if den <= 0:
            continue
        out[i] = np.sum((x - xm) * (p2 - pm)) / den
    return out

# ------------------------------
# Feature builder (sym + buy/sell columns)
# ------------------------------
def build_book_features_sym(df_snap: pd.DataFrame) -> pd.DataFrame:
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

def _mat(merged: pd.DataFrame, prefix: str, field: str, K: int) -> np.ndarray:
    cols = [f"{prefix}_{i}_{field}" for i in range(K)]
    return np.vstack([
        pd.to_numeric(merged.get(c), errors="coerce").astype(float).fillna(0.0).values
        for c in cols
    ]).T

def _depth_within_bps(px_mat: np.ndarray, sz_mat: np.ndarray, mid: np.ndarray, bps: float) -> np.ndarray:
    tol = (bps / 1e4) * mid
    ok = np.isfinite(px_mat) & np.isfinite(mid[:, None]) & (np.abs(px_mat - mid[:, None]) <= tol[:, None])
    return np.sum(np.where(ok, sz_mat, 0.0), axis=1)

def enrich_side_columns(merged: pd.DataFrame, bf: pd.DataFrame) -> pd.DataFrame:
    """
    Convention wide (Point 5):
      - Colonnes "sym" sans suffixe: mid_t, spread_bps_top, obi_*, microprice_bias, slopes bid/ask.
      - Colonnes side-specific suffixées:
          *_buy / *_sell  (valeurs définies sur "côté action")
      - Colonnes signées side:
          *_side_buy / *_side_sell (ex: +obi pour buy, -obi pour sell)
    """
    out = bf.copy()
    mid = out["mid_t"].values.astype(float)

    ask_sz_5  = _mat(merged, "ask", "size", 5)
    bid_sz_5  = _mat(merged, "bid", "size", 5)
    ask_sz_15 = _mat(merged, "ask", "size", 15)
    bid_sz_15 = _mat(merged, "bid", "size", 15)

    ask_px_15 = np.vstack([pd.to_numeric(merged.get(f"ask_{i}_price"), errors="coerce").astype(float).values for i in range(15)]).T
    bid_px_15 = np.vstack([pd.to_numeric(merged.get(f"bid_{i}_price"), errors="coerce").astype(float).values for i in range(15)]).T

    def wall_share(sz_mat: np.ndarray) -> np.ndarray:
        tot = np.sum(sz_mat, axis=1)
        mx = np.max(sz_mat, axis=1)
        return np.where(tot > 0, mx / tot, np.nan)

    out["wall_opp_share_5_buy"]   = wall_share(ask_sz_5)
    out["wall_opp_share_5_sell"]  = wall_share(bid_sz_5)
    out["wall_opp_share_15_buy"]  = wall_share(ask_sz_15)
    out["wall_opp_share_15_sell"] = wall_share(bid_sz_15)

    depth_ask_5bps  = _depth_within_bps(ask_px_15, ask_sz_15, mid, 5.0)
    depth_ask_10bps = _depth_within_bps(ask_px_15, ask_sz_15, mid, 10.0)
    depth_bid_5bps  = _depth_within_bps(bid_px_15, bid_sz_15, mid, 5.0)
    depth_bid_10bps = _depth_within_bps(bid_px_15, bid_sz_15, mid, 10.0)

    out["cum_depth_within_5bps_opp_buy"]   = depth_ask_5bps
    out["cum_depth_within_10bps_opp_buy"]  = depth_ask_10bps
    out["cum_depth_within_5bps_opp_sell"]  = depth_bid_5bps
    out["cum_depth_within_10bps_opp_sell"] = depth_bid_10bps

    bid_px_5 = np.vstack([pd.to_numeric(merged.get(f"bid_{i}_price"), errors="coerce").astype(float).values for i in range(5)]).T
    ask_px_5 = np.vstack([pd.to_numeric(merged.get(f"ask_{i}_price"), errors="coerce").astype(float).values for i in range(5)]).T

    slope_bid_5  = _slope_batch(bid_px_5, bid_sz_5)
    slope_ask_5  = _slope_batch(ask_px_5, ask_sz_5)
    slope_bid_15 = _slope_batch(bid_px_15, bid_sz_15)
    slope_ask_15 = _slope_batch(ask_px_15, ask_sz_15)

    out["slope_bid_5"] = slope_bid_5
    out["slope_ask_5"] = slope_ask_5
    out["slope_bid_15"] = slope_bid_15
    out["slope_ask_15"] = slope_ask_15

    out["slope_opp_5_buy"]   = slope_ask_5
    out["slope_opp_5_sell"]  = slope_bid_5
    out["slope_opp_15_buy"]  = slope_ask_15
    out["slope_opp_15_sell"] = slope_bid_15

    out["obi_5_side_buy"] = out["obi_5"]
    out["obi_5_side_sell"] = -out["obi_5"]
    out["obi_15_side_buy"] = out["obi_15"]
    out["obi_15_side_sell"] = -out["obi_15"]

    out["microprice_bias_side_buy"] = out["microprice_bias"]
    out["microprice_bias_side_sell"] = -out["microprice_bias"]

    return out

# ------------------------------
# Output sinks (GO / DIR)
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
        part_path = f"{self.parts}/part-{self.k:05d}.parquet"
        write_parquet_s3(part_path, df, self.storage_options)
        self.k += 1

# ------------------------------
# Signals -> base events (NO side duplication)
# ------------------------------

def _col_or_default(df: pd.DataFrame, col: str, default, n: int) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * n, index=df.index)

def make_events_base_v2(sig: pd.DataFrame, tfs: list[str], stage1_cols: dict) -> pd.DataFrame:
    # requis stage1
    need = ["row_id", "t", "symbol", "year", "bid_entry", "ask_entry", "spread_bps_entry", "atr_bps"]
    for c in need:
        if c not in sig.columns:
            raise ValueError(f"Signals missing required column: {c}")

    sig = sig.copy()
    sig["t"] = pd.to_datetime(sig["t"], utc=True, errors="coerce")
    sig = sig.dropna(subset=["t"])
    if sig.empty:
        return pd.DataFrame()

    audit_cols_all = stage1_cols.get("audit_cols", [])
    tf_cols_all    = stage1_cols.get("tf_cols", [])
    base_cols_all  = stage1_cols.get("base_cols", [])

    rows = []
    for tf in tfs:
        tf = str(tf).lower().strip()
        suf = "_" + tf

        # base + entries
        cols_keep = [c for c in base_cols_all if c in sig.columns]
        # safety: assure t/row_id/symbol/year
        for c in ["row_id","t","symbol","year"]:
            if c not in cols_keep and c in sig.columns:
                cols_keep.append(c)

        tmp = sig[cols_keep].copy()
        # Keep rr_min_{tf} as config parameter in stage2
        rrc = f"rr_min{suf}"
        if rrc in sig.columns:
            tmp[rrc] = pd.to_numeric(sig[rrc], errors="coerce").astype(np.float32)

        tmp["tf"] = tf

        # labels
        ydir = f"y_dir_{tf}"
        yleg = f"y_{tf}"
        ygo  = f"y_go_{tf}"
        if ydir in sig.columns:
            tmp["y_dir"] = pd.to_numeric(sig[ydir], errors="coerce").fillna(0).astype(np.int8)
        elif yleg in sig.columns:
            tmp["y_dir"] = pd.to_numeric(sig[yleg], errors="coerce").fillna(0).astype(np.int8)
        else:
            tmp["y_dir"] = np.int8(0)

        if ygo in sig.columns:
            tmp["y_go"] = pd.to_numeric(sig[ygo], errors="coerce").fillna(0).astype(np.int8)
        else:
            tmp["y_go"] = (tmp["y_dir"] != 0).astype(np.int8)

        # lift tf_cols -> audit_* canonical (tu peux en garder moins si tu veux)
        # mapping simple “tf suffix -> audit_”
        map_tf_to_audit = {
            f"p_thr_ev0{suf}": "audit_p_thr_ev0",
            f"pnl_net_bps{suf}": "audit_pnl_net_bps",
            f"exit_reason{suf}": "audit_exit_reason",
            f"tp_bps{suf}": "audit_tp_bps",
            f"sl_bps{suf}": "audit_sl_bps",
            f"risk_r_bps{suf}": "audit_risk_r_bps",
            f"fill_mode{suf}": "audit_fill_mode",
        }
        for src, dst in map_tf_to_audit.items():
            if src in sig.columns and dst not in tmp.columns:
                tmp[dst] = sig[src]

        # lift ALL audit cols for this tf: audit_*_{tf} -> audit_*
        for c in audit_cols_all:
            c = str(c)
            if c.endswith(suf) and c in sig.columns:
                tmp[c[:-len(suf)]] = sig[c]  # strip suffix

        # Guardrail: make sure we did not accidentally create audit_rr_min
        if "audit_rr_min" in tmp.columns:
            tmp.drop(columns=["audit_rr_min"], inplace=True)

        # ensure mid_entry if missing
        if "mid_entry" not in tmp.columns:
            tmp["mid_entry"] = (pd.to_numeric(tmp["bid_entry"], errors="coerce") +
                                pd.to_numeric(tmp["ask_entry"], errors="coerce")) / 2.0

        # cast audits safely (CRITICAL for hit_step)
        tmp = _cast_audits_canonical(tmp)
        rows.append(tmp)

    ev = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if ev.empty:
        return ev

    ev["symbol"] = ev["symbol"].astype(str)
    ev["year"] = pd.to_numeric(ev["year"], errors="coerce").fillna(-1).astype(int)
    ev["ym"] = ev["t"].dt.strftime("%Y-%m")
    ev = ev.sort_values(["t","tf"]).reset_index(drop=True)
    return ev
# ------------------------------
# Core processing
# ------------------------------
def _cfg_for_tf(tf: str) -> dict:
    """Return TF config dict with safe default."""
    tf = str(tf).lower().strip()
    return TF_CFG.get(tf, TF_CFG.get("30m"))

def month_dir_from_root_tpl(root_tpl: str, symbol: str, year: int) -> str:
    p = root_tpl.replace("<SYMBOL>", symbol).replace("<YEAR>", str(year)).rstrip("/")
    return p.rsplit("/", 1)[0]

def build_audit_flags_from_sup(
    merged: pd.DataFrame,
    sup_df: pd.DataFrame,
    *,
    noise_override: np.ndarray | pd.Series | None = None,
    debug_cols: bool = False,
) -> pd.DataFrame:
    """
    Construit A/B/C à partir de sup_* + colonnes context (atr_bps, spread_bps_entry, ret_stdev_1s_10s_bps).
    Retourne un DF avec audit_early_abort, audit_timeout, audit_market_toxic (+ debug cols).

    - noise_override: permet de fournir directement la série/array de noise (ret_stdev_1s_10s_bps)
      sans écrire dans `merged` (évite la fragmentation / copies).
    """
    out = pd.DataFrame(index=sup_df.index)

    # --- TF config (vectorisé) ---
    tf = merged["tf"].astype(str).str.lower().str.strip()
    cfg_list = tf.map(_cfg_for_tf).tolist()

    W     = np.array([d["W"]       for d in cfg_list], dtype="float64")
    L     = np.array([d["L"]       for d in cfg_list], dtype="float64")
    C     = np.array([d["C"]       for d in cfg_list], dtype="float64")
    wA    = np.array([d["wA"]      for d in cfg_list], dtype="float64")
    lA    = np.array([d["lA"]      for d in cfg_list], dtype="float64")
    chopB = np.array([d["chopB"]   for d in cfg_list], dtype="float64")
    k30   = np.array([d["k_eps30"] for d in cfg_list], dtype="float64")
    k120  = np.array([d["k_eps120"]for d in cfg_list], dtype="float64")

    # --- helpers ---
    def _to_f64(arr_like, n: int, default=np.nan) -> np.ndarray:
        if arr_like is None:
            return np.full(n, default, dtype="float64")
        if isinstance(arr_like, pd.Series):
            a = pd.to_numeric(arr_like, errors="coerce").to_numpy(dtype="float64", copy=False)
        else:
            a = pd.to_numeric(np.asarray(arr_like), errors="coerce").astype("float64", copy=False)
        if a.shape[0] != n:
            raise ValueError(f"build_audit_flags_from_sup: len mismatch (got {a.shape[0]}, expected {n})")
        return a

    n = len(merged)

    # --- base arrays ---
    spread_entry = pd.to_numeric(merged.get("spread_bps_entry"), errors="coerce").astype("float64").to_numpy()
    atr_bps      = pd.to_numeric(merged.get("atr_bps"), errors="coerce").astype("float64").to_numpy()

    widen_5s = pd.to_numeric(sup_df.get("sup_spread_widen_bps_5s"), errors="coerce").astype("float64").to_numpy()
    liq_drop = pd.to_numeric(sup_df.get("sup_liq_drop_ratio_5s"), errors="coerce").astype("float64").to_numpy()
    chop     = pd.to_numeric(sup_df.get("sup_chop_score_120s"), errors="coerce").astype("float64").to_numpy()

    ret5   = pd.to_numeric(sup_df.get("sup_ret_bps_fwd_5s"), errors="coerce").astype("float64").to_numpy()
    ret30  = pd.to_numeric(sup_df.get("sup_ret_bps_fwd_30s"), errors="coerce").astype("float64").to_numpy()
    ret120 = pd.to_numeric(sup_df.get("sup_ret_bps_fwd_120s"), errors="coerce").astype("float64").to_numpy()

    # --- noise (prefer override to avoid writing into merged) ---
    if noise_override is not None:
        noise = _to_f64(noise_override, n, default=0.0)
    elif "ret_stdev_1s_10s_bps" in merged.columns:
        noise = pd.to_numeric(merged["ret_stdev_1s_10s_bps"], errors="coerce").astype("float64").to_numpy()
    else:
        noise = np.zeros(n, dtype="float64")

    # NaN/inf -> 0 (safe), eps floors will handle min=2.0
    noise = np.where(np.isfinite(noise), noise, 0.0)

    # ---- C: market toxic ----
    widen_norm = widen_5s / np.maximum(spread_entry, 1e-6)
    toxic = (
        (~np.isfinite(widen_norm)) | (~np.isfinite(liq_drop)) | (~np.isfinite(chop)) |
        (widen_norm >= W) | (liq_drop <= L) | (chop >= C)
    )

    # ---- A: early abort ----
    X = np.clip(0.3 * atr_bps, 4.0, 10.0)
    early_abort = (
        (np.isfinite(ret5) & (ret5 <= -X)) |
        ((np.isfinite(widen_5s) & (widen_5s >= wA)) & (np.isfinite(liq_drop) & (liq_drop <= lA)))
    )

    # ---- B: timeout ----
    eps30  = np.maximum(2.0, k30  * noise)
    eps120 = np.maximum(2.0, k120 * noise)

    timeout = (
        (np.isfinite(ret30) & (np.abs(ret30) < eps30)) &
        (np.isfinite(ret120) & (np.abs(ret120) < eps120))
    ) | (
        (np.isfinite(ret30) & (np.abs(ret30) < eps30)) &
        (np.isfinite(chop) & (chop >= chopB))
    )

    out["audit_market_toxic"] = toxic.astype("int8")
    out["audit_early_abort"]  = early_abort.astype("int8")
    out["audit_timeout"]      = timeout.astype("int8")

    if debug_cols:
        out["audit_widen_norm_5s"] = widen_norm.astype("float64")
        out["audit_eps30"]         = eps30.astype("float64")
        out["audit_eps120"]        = eps120.astype("float64")
        out["audit_X_abort"]       = X.astype("float64")

    return out
def process_symbol_year(
    symbol: str,
    year: int,
    signals_root: str,
    book_root_tpl: str,
    trades_root_tpl: str,
    candle_root_tpl: str,
    out_root: str,
    split: str,
    tfs_keep: List[str],
    storage_options: dict,
    bucket_freq: str,
    debug: bool,
) -> int:
    t_global0 = time.time()

    go_schema_cols = None
    dir_schema_cols = None
    
    sig_path = f"{signals_root.rstrip('/')}/{symbol}/{year}_signals.csv"
    log(f"[{symbol} {year}] read signals: {sig_path}")

    sig = read_csv_s3(sig_path, storage_options)
    if sig is None or sig.empty:
        log(f"[{symbol} {year}] signals empty -> skip")
        return 0
    assert_has_cols(sig, ["row_id","t","bid_entry","ask_entry","spread_bps_entry","atr_bps"], name="stage1 signals csv")

    if "symbol" not in sig.columns:
        sig["symbol"] = symbol
    if "year" not in sig.columns:
        sig["year"] = year

    cols_json_path = f"{signals_root.rstrip('/')}/stage1_columns.json"
    log(f"[{symbol} {year}] read stage1_columns.json: {cols_json_path}")
    stage1_cols = read_json_s3(cols_json_path, storage_options)

    found_tfs = tfs_from_stage1_columns(stage1_cols)
    if tfs_keep:
        keep = set([x.lower().strip() for x in tfs_keep])
        found_tfs = [x for x in found_tfs if x in keep]
    if not found_tfs:
        log(f"[{symbol} {year}] no TF after filter -> skip")
        return 0

    ev = make_events_base_v2(sig=sig, tfs=found_tfs, stage1_cols=stage1_cols)
    if ev is None or ev.empty:
        log(f"[{symbol} {year}] events empty -> skip")
        return 0

    out_go_year_dir  = f"{out_root.rstrip('/')}/{split}/go/{symbol}/{year}"
    out_dir_year_dir = f"{out_root.rstrip('/')}/{split}/dir/{symbol}/{year}"
    
    sink_go  = PartsSink(out_year_dir=out_go_year_dir,  storage_options=storage_options)
    sink_dir = PartsSink(out_year_dir=out_dir_year_dir, storage_options=storage_options)

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
    FORWARD  = pd.Timedelta(seconds=180)

    total_go_written = 0
    total_dir_written = 0

    df_go_parts: List[pd.DataFrame] = []
    df_dir_parts: List[pd.DataFrame] = []
    go_rows_acc = 0
    dir_rows_acc = 0

    t_read_book = t_read_trades = t_prep = t_feat = t_write = 0.0

    log(f"[{symbol} {year}] base_events={len(ev):,} tfs={found_tfs} out_go={out_go_year_dir} out_dir={out_dir_year_dir} bucket={bucket_freq}")

    for ym, evm in ev.groupby("ym"):
        t_month0 = time.time()

        book_path = f"{book_dir}/{ym}.parquet"
        trades_path = f"{trades_dir}/{ym}.parquet"

        try:
            book_dset = dataset_from_month_parquet(book_path, pafs)
            trades_dset = dataset_from_month_parquet(trades_path, pafs)
            # book needs at least level0
            assert_schema_has(book_dset, ["timestamp","bid_0_price","ask_0_price","bid_0_size","ask_0_size"], name=f"book {ym}")
            # trades needs at least timestamp/qty and aggr flag if you rely on it
            assert_schema_has(trades_dset, ["timestamp","qty"], name=f"trades {ym}")

        except Exception as e:
            if debug:
                log(f"[{symbol} {year} {ym}] skip month: {e}")
            continue

        book_cache: OrderedDict[pd.Timestamp, BookPrepared] = OrderedDict()
        trades_cache: OrderedDict[pd.Timestamp, TradesPrepared] = OrderedDict()
        candle_cache: OrderedDict[pd.Timestamp, pd.DataFrame] = OrderedDict()

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

        log(f"[{symbol} {year} {ym}] START month events={n_month:,} book={book_path} trades={trades_path}")

        for bucket, evb in evm.groupby("bucket", sort=True):
            # --- book prep ---
            if bucket in book_cache:
                book_prep = book_cache[bucket]
                book_cache.move_to_end(bucket)
            else:
                t0 = bucket - LOOKBACK
                t1 = bucket + bucket_delta + FORWARD

                t_rb0 = time.time()
                try:
                    book_full = read_book_window(book_dset, t0, t1, book_cols)
                except Exception:
                    book_full = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
                t_read_book += (time.time() - t_rb0)

                # Point (3): timing prep correct
                t_p0 = time.time()
                book_prep = prepare_book_features(book_full)
                t_prep += (time.time() - t_p0)

                if len(book_cache) >= MAX_BOOK_CACHE:
                    book_cache.popitem(last=False)
                book_cache[bucket] = book_prep

            if book_prep.df is None or book_prep.df.empty:
                continue

            # --- trades prep ---
            if bucket in trades_cache:
                trades_prep = trades_cache[bucket]
                trades_cache.move_to_end(bucket)
            else:
                t0_tr = bucket - LOOKBACK
                t1_tr = bucket + bucket_delta + FORWARD

                t_rt0 = time.time()
                try:
                    trades_full = read_trades_window(trades_dset, t0_tr, t1_tr)
                except Exception:
                    trades_full = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
                t_read_trades += (time.time() - t_rt0)

                # Point (3): timing prep correct
                t_p0 = time.time()
                trades_prep = prepare_trades_features(trades_full)
                t_prep += (time.time() - t_p0)

                if len(trades_cache) >= MAX_TRADES_CACHE:
                    trades_cache.popitem(last=False)
                trades_cache[bucket] = trades_prep

            # --- vectorized build for this bucket ---
            t_f0 = time.time()

            evb2 = evb.sort_values("t").copy()
            if evb2.empty:
                continue

            book_df = book_prep.df.reset_index().rename(columns={"timestamp": "book_ts"})
            book_df["book_ts"] = pd.to_datetime(book_df["book_ts"], utc=True, errors="coerce")
            book_df = book_df.dropna(subset=["book_ts"])

            sort_cols = ["book_ts"]
            if "seq" in book_df.columns:
                sort_cols.append("seq")
            if "received_time" in book_df.columns:
                sort_cols.append("received_time")
            book_df = book_df.sort_values(sort_cols, kind="mergesort")

            if DEDUP_BOOK_TS:
                book_df = book_df.drop_duplicates(subset=["book_ts"], keep="last")

            merged = pd.merge_asof(
                evb2.sort_values("t"),
                book_df,
                left_on="t",
                right_on="book_ts",
                direction="backward",
                allow_exact_matches=True,
            )

            # --- sanity merge ---
            nan_rate = merged["book_ts"].isna().mean()
            if nan_rate > 0.05:
                log(f"[WARN] {symbol} {year} {ym} bucket={bucket} merge_asof book_ts NaN rate={nan_rate:.2%}")

            # drop rows with no book match (typically month boundary)
            merged = merged.dropna(subset=["book_ts"]).copy()
            merged = merged.copy()
            if merged.empty:
                t_feat += (time.time() - t_f0)
                continue

            # now safe to compute dt
            t_ns = pd.to_datetime(merged["t"], utc=True, errors="coerce").astype("int64").to_numpy()
            b_ns = pd.to_datetime(merged["book_ts"], utc=True, errors="coerce").astype("int64").to_numpy()
            dt_ns = t_ns - b_ns

            # dt négatif = très mauvais (asof inversé / data corrupt)
            neg_dt = np.isfinite(dt_ns) & (dt_ns < 0)
            if neg_dt.any():
                ex = merged.loc[neg_dt, ["t","book_ts","tf"]].head(5)
                raise ValueError(f"NEGATIVE dt_ns detected (asof bug / unsorted input). Examples:\n{ex}")

            # staleness per tf
            tf = merged["tf"].astype(str).str.lower().str.strip()

            max_stale_ns = tf.map(lambda x: int(MAX_STALE_BY_TF.get(x, pd.Timedelta(seconds=5)).to_timedelta64())).to_numpy()

            t_ns = pd.to_datetime(merged["t"], utc=True, errors="coerce").astype("int64").to_numpy()
            b_ns = pd.to_datetime(merged["book_ts"], utc=True, errors="coerce").astype("int64").to_numpy()
            
            ok_stale = dt_ns <= max_stale_ns

            bid0 = pd.to_numeric(merged.get("bid_0_price"), errors="coerce").astype(float)
            ask0 = pd.to_numeric(merged.get("ask_0_price"), errors="coerce").astype(float)
            ok_px = np.isfinite(bid0) & np.isfinite(ask0) & (bid0 > 0) & (ask0 > 0) & (ask0 >= bid0)

            # --- sanity merge ---
            if merged["book_ts"].isna().mean() > 0.05:
                log(f"[WARN] {symbol} {year} {ym} bucket={bucket} merge_asof book_ts NaN rate={merged['book_ts'].isna().mean():.2%}")

            t_ns = pd.to_datetime(merged["t"], utc=True, errors="coerce").astype("int64")
            b_ns = pd.to_datetime(merged["book_ts"], utc=True, errors="coerce").astype("int64")
            dt_ns = (t_ns - b_ns).to_numpy()

            # staleness check : si tu drop > 95% des lignes, tu veux le savoir vite
            pre_n = len(merged)
            
            merged = merged[ok_stale & ok_px].copy()
            post_n = len(merged)
            if pre_n > 0 and post_n / pre_n < 0.05:
                log(f"[WARN] {symbol} {year} {ym} bucket={bucket} kept={post_n}/{pre_n} ({post_n/pre_n:.2%}) after stale+px filters")

            if merged.empty:
                t_feat += (time.time() - t_f0)
                continue

            merged = _cast_audits_canonical(merged)

            # ==========================
            # NEW oracle-safe "anti-audit" proxies (pure backward)
            # ==========================
            t_evt = pd.to_datetime(merged["t"], utc=True, errors="coerce")
            t_book = pd.to_datetime(merged["book_ts"], utc=True, errors="coerce")

            # 1) book_age_ms (snapshot freshness actually used)
            age_ms = (t_evt.astype("int64") - t_book.astype("int64")) / 1e6
            age_ms = age_ms.to_numpy(dtype="float64", copy=False)
            age_ms[~np.isfinite(age_ms)] = np.nan
            age_ms = np.clip(age_ms, 0.0, 60_000.0)  # cap 60s
            # We'll store it later into bf as a model feature.

            # Build book time-series (from book_prep.df) for backward-asof at shifted times
            bdf = book_prep.df
            bid0_s = pd.to_numeric(bdf.get("bid_0_price"), errors="coerce").astype("float64")
            ask0_s = pd.to_numeric(bdf.get("ask_0_price"), errors="coerce").astype("float64")
            bsz0_s = pd.to_numeric(bdf.get("bid_0_size"), errors="coerce").astype("float64").fillna(0.0)
            asz0_s = pd.to_numeric(bdf.get("ask_0_size"), errors="coerce").astype("float64").fillna(0.0)

            mid_s = ((bid0_s + ask0_s) / 2.0).replace([np.inf, -np.inf], np.nan)
            spr_s = pd.Series(np.where(mid_s > 0, 1e4 * (ask0_s - bid0_s) / mid_s, np.nan), index=bdf.index)
            liq_s = (bsz0_s + asz0_s).astype("float64")

            # NOW values from merged snapshot (consistent with other features)
            bid_now = pd.to_numeric(merged.get("bid_0_price"), errors="coerce").astype("float64")
            ask_now = pd.to_numeric(merged.get("ask_0_price"), errors="coerce").astype("float64")
            mid_now = ((bid_now + ask_now) / 2.0).to_numpy(dtype="float64", copy=False)

            spr_now = np.where(
                np.isfinite(mid_now) & (mid_now > 0),
                1e4 * (ask_now.to_numpy(dtype="float64", copy=False) - bid_now.to_numpy(dtype="float64", copy=False)) / mid_now,
                np.nan
            ).astype("float64")

            liq_now = (
                pd.to_numeric(merged.get("bid_0_size"), errors="coerce").astype("float64").fillna(0.0)
                + pd.to_numeric(merged.get("ask_0_size"), errors="coerce").astype("float64").fillna(0.0)
            ).to_numpy(dtype="float64", copy=False)

            # 2) spread_trend_10s = spread_now - spread(t-10s)
            t_m10 = t_evt - pd.Timedelta(seconds=10)
            spr_m10 = _asof_values(spr_s, t_m10)
            spread_trend_10s = spr_now - spr_m10
            spread_trend_10s[~np.isfinite(spread_trend_10s)] = np.nan
            spread_trend_10s = np.clip(spread_trend_10s, -500.0, 500.0)

            # 3) liq_trend_10s = liq_now / liq(t-10s)
            liq_m10 = _asof_values(liq_s, t_m10)
            liq_trend_10s = np.where(
                np.isfinite(liq_now) & (liq_now > 0) & np.isfinite(liq_m10) & (liq_m10 > 0),
                liq_now / liq_m10,
                np.nan
            ).astype("float64")
            liq_trend_10s = np.clip(liq_trend_10s, 0.0, 100.0)

            # 4) ret_bps_5s_back = 1e4*(mid_now/mid(t-5s) - 1)
            t_m5 = t_evt - pd.Timedelta(seconds=5)
            mid_m5 = _asof_values(mid_s, t_m5)
            ret_bps_5s_back = 1e4 * (mid_now / np.maximum(mid_m5, 1e-12) - 1.0)
            ret_bps_5s_back[~np.isfinite(ret_bps_5s_back)] = np.nan
            ret_bps_5s_back = np.clip(ret_bps_5s_back, -500.0, 500.0)

            # entry spread diagnostics
            be = pd.to_numeric(merged.get("bid_entry"), errors="coerce")
            ae = pd.to_numeric(merged.get("ask_entry"), errors="coerce")
            mid_e = (be + ae) / 2.0
            spread_entry_recalc = np.where(mid_e > 0, 1e4 * (ae - be) / mid_e, np.nan)

            if RECALC_SPREAD_ENTRY:
                merged["spread_bps_entry"] = spread_entry_recalc

            # Point (4): debug checks (neg spread + mismatch)
            if debug:
                neg = np.isfinite(spread_entry_recalc) & (spread_entry_recalc < -1e-9)
                if neg.any():
                    ex = merged.loc[neg, ["t","tf","bid_entry","ask_entry","spread_bps_entry"]].head(LOG_NEG_SPREAD_MAX_EX)
                    log(f"[{symbol} {year} {ym}] NEG_SPREAD_ENTRY n={int(neg.sum())} examples=\n{ex}")

                sbe = pd.to_numeric(merged.get("spread_bps_entry"), errors="coerce")
                mis = np.isfinite(sbe) & np.isfinite(spread_entry_recalc) & (np.abs(sbe - spread_entry_recalc) > 1e-6)
                if mis.any():
                    ex = merged.loc[mis, ["t","tf","bid_entry","ask_entry","spread_bps_entry"]].head(LOG_NEG_SPREAD_MAX_EX)
                    log(f"[{symbol} {year} {ym}] SPREAD_ENTRY_MISMATCH n={int(mis.sum())} examples=\n{ex}")

            # candles
            candle_path = candle_path_from_tpl(candle_root_tpl, symbol, year)
            if bucket in candle_cache:
                candles_1m = candle_cache[bucket]
                candle_cache.move_to_end(bucket)
            else:
                t0_c = bucket - CANDLE_LOOKBACK
                t1_c = bucket + bucket_delta + CANDLE_FORWARD
                try:
                    candles_1m = read_candles_window(candle_path, pafs, t0_c, t1_c)
                except Exception:
                    candles_1m = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
                if len(candle_cache) >= MAX_CANDLE_CACHE:
                    candle_cache.popitem(last=False)
                candle_cache[bucket] = candles_1m

            merged = attach_candle_features(merged, candles_1m)

            # === features ===
            bf = build_book_features_sym(merged)

            bf["book_age_ms"] = age_ms
            bf["spread_trend_10s"] = spread_trend_10s
            bf["liq_trend_10s"] = liq_trend_10s
            bf["ret_bps_5s_back"] = ret_bps_5s_back

            bf["quote_churn_10s"] = _asof_values(book_prep.churn10, merged["t"])
            bf["ret_stdev_1s_10s_bps"] = _asof_values(book_prep.retstd10_bps, merged["t"])
            # Needed by build_audit_flags_from_sup (it reads from merged)

            agr3  = _asof_values(trades_prep.aggr3,  merged["t"])
            agr5  = _asof_values(trades_prep.aggr5,  merged["t"])
            agr10 = _asof_values(trades_prep.aggr10, merged["t"])
            agr15 = _asof_values(trades_prep.aggr15, merged["t"])

            bf["aggr_ratio_10s"] = agr10
            bf["aggr_ratio_15s"] = agr15
            bf["bt_dom_3s"] = agr3
            bf["bt_dom_5s"] = agr5
            bf["bt_dom_10s"] = agr10

            # --- intensity asof ---
            ntr_30s = _asof_values(trades_prep.ntr_30s, merged["t"])
            vol_30s = _asof_values(trades_prep.vol_30s, merged["t"])
            ntr_2m  = _asof_values(trades_prep.ntr_2m,  merged["t"])
            vol_2m  = _asof_values(trades_prep.vol_2m,  merged["t"])
            ntr_5m  = _asof_values(trades_prep.ntr_5m,  merged["t"])
            vol_5m  = _asof_values(trades_prep.vol_5m,  merged["t"])

            bf["n_trades_30s"] = ntr_30s
            bf["vol_traded_30s"] = vol_30s
            bf["n_trades_2m"] = ntr_2m
            bf["vol_traded_2m"] = vol_2m
            bf["n_trades_5m"] = ntr_5m
            bf["vol_traded_5m"] = vol_5m

            # last_trade_age_s: on récupère ts_ns du dernier trade (asof), puis t - ts
            last_ts_ns = _asof_values(trades_prep.last_trade_age_s, merged["t"])  # float64
            t_ns = pd.to_datetime(merged["t"], utc=True, errors="coerce").astype("int64").to_numpy(dtype="int64", copy=False)

            age_s = np.full(len(t_ns), np.nan, dtype="float64")
            ok = np.isfinite(last_ts_ns)
            age_s[ok] = (t_ns[ok] - last_ts_ns[ok].astype("int64")) / 1e9
            age_s = np.clip(age_s, 0.0, 3600.0)
            bf["last_trade_age_s"] = age_s

            bf = enrich_side_columns(merged, bf)

            eps = 1e-12
            bf["thinness_5bps_buy"]  = bf["spread_bps_top"] / (bf["cum_depth_within_5bps_opp_buy"]  + eps)
            bf["thinness_5bps_sell"] = bf["spread_bps_top"] / (bf["cum_depth_within_5bps_opp_sell"] + eps)

            bf["thinness_10bps_buy"]  = bf["spread_bps_top"] / (bf["cum_depth_within_10bps_opp_buy"]  + eps)
            bf["thinness_10bps_sell"] = bf["spread_bps_top"] / (bf["cum_depth_within_10bps_opp_sell"] + eps)

            THIN_MAX = 1e6  # safe cap, oracle-safe

            for c in [
                "thinness_5bps_buy", "thinness_5bps_sell",
                "thinness_10bps_buy", "thinness_10bps_sell",
            ]:
                bf[c] = np.clip(bf[c], 0.0, THIN_MAX)

            # signed aggr for buy/sell (neutre => 0)
            bf["aggr_ratio_10s_side_buy"]  = 2.0 * agr10 - 1.0
            bf["aggr_ratio_10s_side_sell"] = -bf["aggr_ratio_10s_side_buy"]

            bf["aggr_ratio_15s_side_buy"]  = 2.0 * agr15 - 1.0
            bf["aggr_ratio_15s_side_sell"] = -bf["aggr_ratio_15s_side_buy"]

            bf["bt_dom_3s_side_buy"]  = 2.0 * agr3 - 1.0
            bf["bt_dom_3s_side_sell"] = -bf["bt_dom_3s_side_buy"]

            # ✅ AJOUT manquant (5s)
            bf["bt_dom_5s_side_buy"]  = 2.0 * agr5 - 1.0
            bf["bt_dom_5s_side_sell"] = -bf["bt_dom_5s_side_buy"]

            bf["bt_dom_10s_side_buy"]  = 2.0 * agr10 - 1.0
            bf["bt_dom_10s_side_sell"] = -bf["bt_dom_10s_side_buy"]

            # candles cols
            CANDLE_COLS = ["bb_width","bb_width_pctl","lf_bb_width_pct","atr_percentile","lf_atr_rank_30m","atr_pct_rank_30m","adx"]
            for c in CANDLE_COLS:
                bf[c] = pd.to_numeric(merged.get(c), errors="coerce").astype(float).values

            n = len(merged)

            # Point (1-2): NO event_id from upstream; we build later with task/tf
            base_out = pd.DataFrame({
                "t": merged["t"].values,
                "symbol": merged["symbol"].astype(str).values,
                "row_id": merged["row_id"].astype(str).values,
                "year": pd.to_numeric(merged["year"], errors="coerce").fillna(-1).astype(int).values,
                "tf": merged["tf"].astype(str).str.lower().str.strip().values,

                # labels bruts (stage3 fera Y)
                "y_go": pd.to_numeric(merged["y_go"], errors="coerce").fillna(0).astype(int).values,
                "y_dir": pd.to_numeric(merged["y_dir"], errors="coerce").fillna(0).astype(int).values,

                # core entry/metaev0
                "bid_entry": pd.to_numeric(merged["bid_entry"], errors="coerce").astype(float).values,
                "ask_entry": pd.to_numeric(merged["ask_entry"], errors="coerce").astype(float).values,
                "spread_bps_entry": pd.to_numeric(merged["spread_bps_entry"], errors="coerce").astype(float).values,
                "atr_bps": pd.to_numeric(merged["atr_bps"], errors="coerce").fillna(0.0).astype(float).values,

                # audits (optional)
                "audit_pnl_net_bps": pd.to_numeric(_col_or_default(merged, "audit_pnl_net_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_exit_reason": _col_or_default(merged, "audit_exit_reason", "NONE", n),
                "audit_tp_bps": pd.to_numeric(_col_or_default(merged, "audit_tp_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_sl_bps": pd.to_numeric(_col_or_default(merged, "audit_sl_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_risk_r_bps": pd.to_numeric(_col_or_default(merged, "audit_risk_r_bps", np.nan, n), errors="coerce").astype(float).values,
                "audit_fill_mode": _col_or_default(merged, "audit_fill_mode", "NA", n).astype(str).values,
                "audit_p_thr_ev0": pd.to_numeric(_col_or_default(merged, "audit_p_thr_ev0", np.nan, n), errors="coerce").astype(float).values,
            })

            # ---- KEEP rr_min_<tf> config columns for Stage3 (NON-audit; do not map to audit_rr_min) ----
            # Ensure schema stability: always emit rr_min_<tf> for every TF in found_tfs
            for tfv in found_tfs:
                col = f"rr_min_{str(tfv).lower().strip()}"
                base_out[col] = pd.to_numeric(
                    _col_or_default(merged, col, np.nan, n),
                    errors="coerce"
                ).astype(np.float32).values
                
            # ---- NEW: keep Stage1 audit_*_L/_S (lifted) into base_out ----
            lifted = [c for c in merged.columns if c.startswith("audit_") and c not in base_out.columns]
            if lifted:
                base_out = pd.concat([base_out, merged[lifted].reset_index(drop=True)], axis=1)


            # ✅ NEW: oracle-safe supervision (post-entry) — NEVER as model features
            sup_df = compute_supervision_features_post_entry(
                merged=merged,
                book_prep=book_prep,
                horizons_sec=SUP_HORIZONS_SEC,
            )
            
            audit_flags = build_audit_flags_from_sup(
                merged=merged,
                sup_df=sup_df,
                noise_override=bf["ret_stdev_1s_10s_bps"].values,
                debug_cols=debug
            )
            
            out_all = pd.concat([
                base_out.reset_index(drop=True),
                bf.reset_index(drop=True),
                sup_df.reset_index(drop=True),          # ✅ NEW: keep sup_* columns
                audit_flags.reset_index(drop=True)
            ], axis=1)
            out_all = _cast_audits_canonical(out_all)

            # Guardrail: Stage2 must not output audit_rr_min anymore (was a legacy duplicate)
            if "audit_rr_min" in out_all.columns:
                out_all.drop(columns=["audit_rr_min"], inplace=True)
            # Fail-safe: never keep deprecated audit_fill_window_sec
            if "audit_fill_window_sec" in out_all.columns:
                out_all.drop(columns=["audit_fill_window_sec"], inplace=True)

            # Point (6): DIR drop neutral — keep only y_dir ∈ {-1,+1}
            out_go = out_all.copy()
            out_dir = out_all[out_all["y_dir"].isin([-1, 1])].copy()

            # Point (1-2): set task + event_id = row_id|task|tf (bretelles)
            if not out_go.empty:
                out_go["task"] = "go"
                out_go["event_id"] = out_go["row_id"].astype(str) + "|go|" + out_go["tf"].astype(str)

            if not out_dir.empty:
                out_dir["task"] = "dir"
                out_dir["event_id"] = out_dir["row_id"].astype(str) + "|dir|" + out_dir["tf"].astype(str)

            # ✅ Reorder for Stage3 convenience
            out_go = reorder_stage2_columns(out_go)
            out_dir = reorder_stage2_columns(out_dir)

            if debug and (not out_go.empty) and out_go["event_id"].duplicated().any():
                raise ValueError(
                    f"duplicate event_id in GO: "
                    f"{out_go.loc[out_go['event_id'].duplicated(), 'event_id'].head(5).tolist()}"
                )

            if debug and (not out_dir.empty) and out_dir["event_id"].duplicated().any():
                raise ValueError(
                    f"duplicate event_id in DIR: "
                    f"{out_dir.loc[out_dir['event_id'].duplicated(), 'event_id'].head(5).tolist()}"
                )

            # Point (5) guardrails: ensure no ambiguous columns like "..._side" without buy/sell
            if debug:
                bad_side_cols = [c for c in out_all.columns if c.endswith("_side")]
                if bad_side_cols:
                    log(f"[WARN] ambiguous *_side columns (expected *_side_buy/_side_sell): {bad_side_cols[:20]}")

            if debug:
                log(f"[{symbol} {year} {ym}] out_go rows={len(out_go):,} out_dir rows={len(out_dir):,} dir_rate={(len(out_dir)/max(1,len(out_go))):.2%}")

            if not out_go.empty:
                go_schema_cols = _check_schema(out_go, go_schema_cols, f"GO {symbol}/{year}")
            if not out_dir.empty:
                dir_schema_cols = _check_schema(out_dir, dir_schema_cols, f"DIR {symbol}/{year}")
                
            if not out_go.empty:
                df_go_parts.append(out_go)
                go_rows_acc += len(out_go)
                total_go_written += len(out_go)

            if not out_dir.empty:
                df_dir_parts.append(out_dir)
                dir_rows_acc += len(out_dir)
                total_dir_written += len(out_dir)

            t_feat += (time.time() - t_f0)

            # flush batches
            if go_rows_acc >= BATCH_MAX:
                t_w0 = time.time()
                sink_go.write_df(pd.concat(df_go_parts, ignore_index=True))
                t_write += (time.time() - t_w0)
                df_go_parts.clear()
                go_rows_acc = 0
                gc.collect()

            if dir_rows_acc >= BATCH_MAX:
                t_w0 = time.time()
                sink_dir.write_df(pd.concat(df_dir_parts, ignore_index=True))
                t_write += (time.time() - t_w0)
                df_dir_parts.clear()
                dir_rows_acc = 0
                gc.collect()

        # end month: flush remainders
        if df_go_parts:
            t_w0 = time.time()
            sink_go.write_df(pd.concat(df_go_parts, ignore_index=True))
            t_write += (time.time() - t_w0)
            df_go_parts.clear()
            go_rows_acc = 0

        if df_dir_parts:
            t_w0 = time.time()
            sink_dir.write_df(pd.concat(df_dir_parts, ignore_index=True))
            t_write += (time.time() - t_w0)
            df_dir_parts.clear()
            dir_rows_acc = 0

        dtm = time.time() - t_month0
        log(f"[{symbol} {year} {ym}] DONE month time={dtm:.1f}s wrote_go={total_go_written:,} wrote_dir={total_dir_written:,}")

    dt = time.time() - t_global0
    log(
        f"[{symbol} {year}] DONE wrote_go={total_go_written:,} wrote_dir={total_dir_written:,} total_time={dt:.1f}s | "
        f"timing_total(s): read_book={t_read_book:.1f} read_trades={t_read_trades:.1f} prep={t_prep:.1f} feat={t_feat:.1f} write={t_write:.1f}"
    )
    return int(total_go_written + total_dir_written)

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser("Stage2 v4.1 — dual outputs (go/dir), no Y — GO wide + DIR wide")

    p.add_argument("--signals-root", default="s3://tradebot-config-tokyo/data/stage1")
    p.add_argument("--candle-root", default="s3://tradebot-config-tokyo/data/bougie/<SYMBOL>/<SYMBOL>-1m-<YEAR>.parquet")
    p.add_argument("--book-root", default="s3://tradebot-config-tokyo/data/book/<SYMBOL>/<YEAR>-*.parquet")
    p.add_argument("--trades-root", default="s3://tradebot-config-tokyo/data/trade/<SYMBOL>/<YEAR>-*.parquet")
    p.add_argument("--out-root", default="s3://tradebot-config-tokyo/data/stage2")

    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)

    p.add_argument("--tfs", nargs="+", default=None)
    p.add_argument("--split", choices=["train","val","test","all"], default="all")

    p.add_argument("--s3-region", default="ap-northeast-1")
    p.add_argument("--s3-anon", action="store_true")

    p.add_argument("--bucket", default="1h", help="Cache bucket size, e.g. 1h (default), 30min, 15min, 5min")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    storage_options = so(args.s3_region, args.s3_anon)

    tfs_keep = []
    if args.tfs:
        tfs_keep = [str(x).strip().lower() for x in args.tfs if str(x).strip()]

    for sym in args.symbols:
        for year in args.years:
            n = process_symbol_year(
                symbol=sym,
                year=year,
                signals_root=args.signals_root,
                book_root_tpl=args.book_root,
                trades_root_tpl=args.trades_root,
                candle_root_tpl=args.candle_root,
                out_root=args.out_root,
                split=args.split,
                tfs_keep=tfs_keep,
                storage_options=storage_options,
                bucket_freq=str(args.bucket),
                debug=bool(args.debug),
            )
            log(f"[DONE] {sym} {year} rows_written(go+dir)={n} split={args.split}")

if __name__ == "__main__":
    main()