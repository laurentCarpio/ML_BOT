#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_bot/stageC/build_dataset_stageC.py

Build StageC dataset v1 (directional confirmation) from:
- StageB parquet dataset (t0 rows + features + audit_p_thr_ev0)
- StageB model (XGB) to compute p(t0) and gate allow via s = p - p_thr_ev0 >= thr_s
- prices_1s mid-price parquet (id_t epoch seconds + mid) to build:
    - features at t1 = t0 + Δ (Δ=30s)
    - label L2 first-touch (TP/SL) starting at t1 over horizon H (H=120s)

Output:
s3://.../data/stageC/dataset=v1/symbol=BTCUSDT/split={train|val|test}/date=YYYY-MM-DD/part-*.parquet

Notes:
- Dataset StageC est "allow-only" (on écrit uniquement les lignes passées par StageB gate).
- Features StageC = mid-only (robuste) ; tu pourras ajouter book/trades ensuite.

Assumptions:
- StageB parquet contains:
    - id_t (datetime64[ns, UTC] ou string parseable)
    - audit_p_thr_ev0 (float)
    - features: columns.json feature_cols_xgb (utilisées par le modèle StageB)
- prices_1s parquet contains:
    - id_t (int64 epoch seconds UTC)
    - mid (float)
"""

from __future__ import annotations

import argparse
import json
import math
import gc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_numeric_dtype,
)

# -----------------------------
# Utils
# -----------------------------
def utc_dt(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")

def to_epoch_sec(ts_utc: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    return (ts.astype("int64") // 1_000_000_000).to_numpy(np.int64, copy=False)

def date_str_from_epoch(sec: np.ndarray) -> np.ndarray:
    # sec int64 -> "YYYY-MM-DD"
    return pd.to_datetime(sec, unit="s", utc=True).strftime("%Y-%m-%d").to_numpy()

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def parse_json_s3(fs: s3fs.S3FileSystem, uri: str) -> dict:
    with fs.open(uri, "rb") as f:
        return json.load(f)

def write_parquet_s3(fs: s3fs.S3FileSystem, uri: str, df: pd.DataFrame, compression: str = "snappy"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    with fs.open(uri, "wb") as f:
        pq.write_table(table, f, compression=compression)

def list_stageb_files(fs: s3fs.S3FileSystem, stageb_root: str, symbol: str, split: str) -> List[str]:
    prefix = f"{stageb_root.rstrip('/')}/symbol={symbol}/split={split}/parquet"
    pattern = prefix.replace("s3://", "") + "/*.parquet"
    out = [f"s3://{p}" for p in fs.glob(pattern)]
    if not out:
        raise FileNotFoundError(f"No StageB SCORED parquet files found under {prefix}")
    return sorted(out)

def sample_paths(paths: List[str], n_files: int, seed: int) -> List[str]:
    if n_files <= 0 or n_files >= len(paths):
        return paths
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(paths), size=int(n_files), replace=False)
    return [paths[i] for i in idx]

def ensure_dir_s3(fs: s3fs.S3FileSystem, uri_dir: str):
    # s3fs doesn't need explicit mkdir, but we keep for clarity
    if not uri_dir.endswith("/"):
        uri_dir = uri_dir + "/"
    if not fs.exists(uri_dir):
        fs.mkdirs(uri_dir, exist_ok=True)

def col_as_float32(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32, copy=False)

def col_as_int32(df: pd.DataFrame, col: str, default: int = 0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.int32)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(np.int32, copy=False)

class BookMonthCache:
    def __init__(self, fs: s3fs.S3FileSystem, book_root: str, symbol: str, max_months: int = 2):
        self.fs = fs
        self.book_root = book_root.rstrip("/")
        self.symbol = symbol
        self._cache: Dict[str, pd.DataFrame] = {}
        self.max_months = max_months
        self._order: List[str] = []

    def _schema_cols(self, path: str) -> set[str]:
        with self.fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            return set(pf.schema.names)
    
    def _load_month(self, ym: str) -> pd.DataFrame:
        # ym="YYYY-MM"
        path = f"{self.book_root}/{self.symbol}/{ym}.parquet"
        df = pd.read_parquet(path, engine="pyarrow")
        ts = "timestamp" if "timestamp" in df.columns else None
        if ts is None:
            raise ValueError("Book parquet missing timestamp")
        df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")
        df = df.dropna(subset=[ts]).sort_values(ts).set_index(ts, drop=True)

        # detect bid0/ask0
        if "bid_0_price" in df.columns and "ask_0_price" in df.columns:
            df["bid0"] = pd.to_numeric(df["bid_0_price"], errors="coerce")
            df["ask0"] = pd.to_numeric(df["ask_0_price"], errors="coerce")
        elif "bid0" in df.columns and "ask0" in df.columns:
            df["bid0"] = pd.to_numeric(df["bid0"], errors="coerce")
            df["ask0"] = pd.to_numeric(df["ask0"], errors="coerce")
        elif "bid" in df.columns and "ask" in df.columns:
            df["bid0"] = pd.to_numeric(df["bid"], errors="coerce")
            df["ask0"] = pd.to_numeric(df["ask"], errors="coerce")
        else:
            raise ValueError("Book parquet missing bid/ask top columns")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["bid0", "ask0"])
        df = df[(df["bid0"] > 0) & (df["ask0"] > 0) & (df["ask0"] > df["bid0"])]

        mid = (df["bid0"].astype("float64") + df["ask0"].astype("float64")) / 2.0
        spread_bps = ((df["ask0"] - df["bid0"]) / mid) * 1e4

        # quote updates flag
        ch = ((df["bid0"].diff().ne(0)) | (df["ask0"].diff().ne(0))).astype("int32")

        # depth sums for imb L1/L5 if size cols exist
        def _sum_depth(side: str, k: int) -> Optional[pd.Series]:
            cols = []
            for i in range(k):
                c1 = f"{side}_{i}_size"
                c2 = f"{side}_{i}_qty"
                if c1 in df.columns: cols.append(c1)
                elif c2 in df.columns: cols.append(c2)
            if not cols:
                return None
            mat = df[cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
            return pd.Series(mat.sum(axis=1), index=df.index, dtype="float32")

        bid1 = _sum_depth("bid", 1)
        ask1 = _sum_depth("ask", 1)
        bid5 = _sum_depth("bid", 5)
        ask5 = _sum_depth("ask", 5)

        def _imb(b: Optional[pd.Series], a: Optional[pd.Series]) -> pd.Series:
            if b is None or a is None:
                return pd.Series(np.nan, index=df.index, dtype="float32")
            den = (b + a).replace(0, np.nan)
            return ((b - a) / den).astype("float32")

        imbL1 = _imb(bid1, ask1)
        imbL5 = _imb(bid5, ask5)

        out = pd.DataFrame({
            "spread_bps": spread_bps.astype("float32"),
            "imb_L1": imbL1.astype("float32"),
            "imb_L5": imbL5.astype("float32"),
            "quote_ch": ch.astype("int32"),
        }, index=df.index)

        # resample 1s
        out_1s = out.resample("1s").last().ffill()

        # quote_updates_20s
        out_1s["quote_updates_20s"] = out_1s["quote_ch"].rolling(20, min_periods=1).sum().astype("int32")

        return out_1s

    def get_month(self, ym: str) -> pd.DataFrame:
        if ym not in self._cache:
            self._cache[ym] = self._load_month(ym)
            self._order.append(ym)
            if len(self._order) > self.max_months:
                old = self._order.pop(0)
                self._cache.pop(old, None)
        return self._cache[ym]

    def _load_month_slice(self, ym: str, t_min: pd.Timestamp, t_max: pd.Timestamp) -> pd.DataFrame:
        path = f"{self.book_root}/{self.symbol}/{ym}.parquet"
        names = self._schema_cols(path)

        if "timestamp" not in names:
            raise ValueError("Book parquet missing timestamp")

        # prices
        if "bid_0_price" in names and "ask_0_price" in names:
            bid_px0, ask_px0 = "bid_0_price", "ask_0_price"
        elif "bid0" in names and "ask0" in names:
            bid_px0, ask_px0 = "bid0", "ask0"
        elif "bid" in names and "ask" in names:
            bid_px0, ask_px0 = "bid", "ask"
        else:
            raise ValueError("Book parquet missing bid/ask top columns")

        # sizes (0..14)
        def _size_col(side: str, i: int) -> str | None:
            c1 = f"{side}_{i}_size"
            c2 = f"{side}_{i}_qty"
            if c1 in names: return c1
            if c2 in names: return c2
            return None

        bid_sz_cols_15 = [c for i in range(15) if (c := _size_col("bid", i)) is not None]
        ask_sz_cols_15 = [c for i in range(15) if (c := _size_col("ask", i)) is not None]
        bid_sz_cols_5 = bid_sz_cols_15[:5]
        ask_sz_cols_5 = ask_sz_cols_15[:5]

        cols = ["timestamp", bid_px0, ask_px0] + bid_sz_cols_15 + ask_sz_cols_15
        df = pd.read_parquet(path, engine="pyarrow", columns=cols)

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.assign(timestamp=ts).dropna(subset=["timestamp"])

        # slice time window
        df = df[(df["timestamp"] >= t_min) & (df["timestamp"] <= t_max)]
        if df.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

        df = df.sort_values("timestamp").set_index("timestamp", drop=True)

        bid0 = pd.to_numeric(df[bid_px0], errors="coerce")
        ask0 = pd.to_numeric(df[ask_px0], errors="coerce")

        # clean
        df2 = pd.DataFrame({"bid0": bid0, "ask0": ask0}, index=df.index)
        df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
        df2 = df2[(df2["bid0"] > 0) & (df2["ask0"] > 0) & (df2["ask0"] > df2["bid0"])]

        mid = (df2["bid0"].astype("float64") + df2["ask0"].astype("float64")) / 2.0
        spread_bps = ((df2["ask0"] - df2["bid0"]) / mid) * 1e4
        ch = ((df2["bid0"].diff().ne(0)) | (df2["ask0"].diff().ne(0))).astype("int32")

        out = pd.DataFrame({
            "spread_bps": spread_bps.astype("float32"),
            "quote_ch": ch.astype("int32"),
        }, index=df2.index)

        # depth helpers
        def _sum_depth(cols_list: list[str]) -> pd.Series:
            if not cols_list:
                return pd.Series(np.nan, index=out.index, dtype="float32")
            m = df.loc[out.index, cols_list].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32, copy=False)
            return pd.Series(m.sum(axis=1), index=out.index, dtype="float32")

        bid1 = _sum_depth(bid_sz_cols_15[:1])
        ask1 = _sum_depth(ask_sz_cols_15[:1])
        bid5 = _sum_depth(bid_sz_cols_5)
        ask5 = _sum_depth(ask_sz_cols_5)
        bid15 = _sum_depth(bid_sz_cols_15)
        ask15 = _sum_depth(ask_sz_cols_15)

        def _imb(b: pd.Series, a: pd.Series) -> pd.Series:
            den = (b + a).replace(0, np.nan)
            return ((b - a) / den).astype("float32")

        out["bid_depth_L15"] = bid15
        out["ask_depth_L15"] = ask15
        out["imb_top5"] = _imb(bid5, ask5)
        out["imb_L15"] = _imb(bid15, ask15)

        # microprice + queue imbalance (L1)
        den = (bid1 + ask1).replace(0, np.nan)
        microprice = (df2.loc[out.index, "ask0"] * bid1 + df2.loc[out.index, "bid0"] * ask1) / den
        out["microprice_diff_bps"] = (((microprice - mid.loc[out.index]) / mid.loc[out.index]) * 1e4).astype("float32")
        out["queue_imb"] = _imb(bid1, ask1)

        # resample 1s + rolling quote updates
        out_1s = out.resample("1s").last().ffill()
        out_1s["quote_updates_20s"] = out_1s["quote_ch"].rolling(20, min_periods=1).sum().astype("int32")
        return out_1s

    def asof(self, t1_utc) -> pd.DataFrame:
        idx = pd.DatetimeIndex(pd.to_datetime(t1_utc, utc=True, errors="coerce")).dropna()
        if len(idx) == 0:
            return pd.DataFrame(index=idx)

        # marge pour rolling(20s) + ffill
        t_min = idx.min() - pd.Timedelta(seconds=60)
        t_max = idx.max() + pd.Timedelta(seconds=5)

        yms = sorted(set(idx.strftime("%Y-%m")))
        frames = [self._load_month_slice(ym, t_min, t_max) for ym in yms]
        b = pd.concat(frames).sort_index()

        return b.reindex(idx, method="ffill")
    
class TradesMonthCache:
    def __init__(self, fs, trade_root: str, symbol: str, max_months: int = 2):
        self.fs = fs
        self.trade_root = trade_root.rstrip("/")
        self.symbol = symbol
        self.max_months = max_months
        self.OUT_COLS = ["flow_signed_vol_20s", "flow_aggr_buy_ratio_20s"]

    def _schema_cols(self, path: str):
        with self.fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            return set(pf.schema.names)

    def _to_utc_ts(self, s: pd.Series) -> pd.Series:
        """
        Supporte:
        - datetime tz-aware (datetime64[ns, UTC]) -> keep
        - datetime naïf -> localise en UTC
        - string ISO
        - int/float epoch (s/ms/us/ns) auto-detect
        """
        # tz-aware datetime
        if is_datetime64tz_dtype(s):
            return pd.to_datetime(s, utc=True, errors="coerce")

        # datetime (naïf ou numpy datetime64)
        if is_datetime64_any_dtype(s):
            ts = pd.to_datetime(s, errors="coerce")
            # si naïf -> assume UTC
            try:
                return ts.dt.tz_localize("UTC")
            except Exception:
                return pd.to_datetime(ts, utc=True, errors="coerce")

        # numeric epoch
        if is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            med = x.dropna().astype("float64").median() if x.notna().any() else np.nan
            if not np.isfinite(med):
                return pd.Series([pd.NaT] * len(s), dtype="datetime64[ns, UTC]")

            if med >= 1e18:
                unit = "ns"
            elif med >= 1e15:
                unit = "us"
            elif med >= 1e12:
                unit = "ms"
            else:
                unit = "s"
            return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")

        # fallback string
        return pd.to_datetime(s.astype(str), utc=True, errors="coerce")

    def _load_month_slice(self, ym: str, t_min: pd.Timestamp, t_max: pd.Timestamp) -> pd.DataFrame:
        path = f"{self.trade_root}/{self.symbol}/{ym}.parquet"
        if not self.fs.exists(path):
            return pd.DataFrame({c: pd.Series(dtype="float32") for c in self.OUT_COLS},
                                index=pd.DatetimeIndex([], tz="UTC"))

        # lecture minimaliste
        df = pd.read_parquet(path, engine="pyarrow", columns=["timestamp", "qty", "is_aggr_buy", "side"])

        # timestamp -> UTC
        ts = self._to_utc_ts(df["timestamp"])
        df = df.assign(timestamp=ts).dropna(subset=["timestamp"])

        # slice window (+ marge pour rolling)
        df = df[(df["timestamp"] >= t_min) & (df["timestamp"] <= t_max)]
        if df.empty:
            return pd.DataFrame({c: pd.Series(dtype="float32") for c in self.OUT_COLS},
                                index=pd.DatetimeIndex([], tz="UTC"))

        df = df.sort_values("timestamp").set_index("timestamp", drop=True)

        qty = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0).astype("float64")

        # aggressor buy
        if "is_aggr_buy" in df.columns and df["is_aggr_buy"].notna().any():
            is_buy = df["is_aggr_buy"].astype(bool)
        else:
            s = df["side"].astype(str).str.lower().str.strip()
            is_buy = (s == "buy")

        # --- RESAMPLE 1s ---
        buy_1s  = qty.where(is_buy, 0.0).resample("1s").sum()
        sell_1s = qty.where(~is_buy, 0.0).resample("1s").sum()
        tot_1s = buy_1s + sell_1s
        signed_1s = buy_1s - sell_1s

        # --- ROLLING 20s ---
        signed_20 = signed_1s.rolling(20, min_periods=1).sum()
        buy_20 = buy_1s.rolling(20, min_periods=1).sum()
        tot_20 = tot_1s.rolling(20, min_periods=1).sum()

        out = pd.DataFrame(index=tot_1s.index)
        out["flow_signed_vol_20s"] = signed_20.astype("float32")

        denom = tot_20.replace(0.0, np.nan)
        ratio = (buy_20 / denom).astype("float32")

        # choix "sans NaN" pour ratio : neutre 0.5 quand pas de volume
        ratio = ratio.fillna(0.5)

        out["flow_aggr_buy_ratio_20s"] = ratio.astype("float32")

        return out

    def asof(self, t1_utc) -> pd.DataFrame:
        idx = pd.DatetimeIndex(pd.to_datetime(t1_utc, utc=True, errors="coerce")).dropna()
        if len(idx) == 0:
            return pd.DataFrame({c: pd.Series(dtype="float32") for c in self.OUT_COLS}, index=idx)

        t_min = idx.min() - pd.Timedelta(seconds=60)
        t_max = idx.max() + pd.Timedelta(seconds=5)

        yms = sorted(set(idx.strftime("%Y-%m")))
        frames = [self._load_month_slice(ym, t_min, t_max) for ym in yms]
        t = pd.concat(frames).sort_index() if frames else pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

        # reindex/ffill sur les timestamps t1 (pile à la seconde)
        out = t.reindex(idx, method="ffill")

        # garanties "no NaN"
        if "flow_signed_vol_20s" not in out.columns:
            out["flow_signed_vol_20s"] = 0.0
        out["flow_signed_vol_20s"] = pd.to_numeric(out["flow_signed_vol_20s"], errors="coerce").fillna(0.0).astype("float32")

        if "flow_aggr_buy_ratio_20s" not in out.columns:
            out["flow_aggr_buy_ratio_20s"] = 0.5
        out["flow_aggr_buy_ratio_20s"] = pd.to_numeric(out["flow_aggr_buy_ratio_20s"], errors="coerce").fillna(0.5).astype("float32")

        return out[self.OUT_COLS]

# -----------------------------
# prices_1s loader (per date)
# -----------------------------
class PricesCache:
    """
    Cache simple des jours prices_1s : date -> (sec_array, mid_array)
    On suppose coverage ~ 1s, id_t unique et trié.
    """
    def __init__(self, fs: s3fs.S3FileSystem, prices_root: str, symbol: str):
        self.fs = fs
        self.prices_root = prices_root.rstrip("/")
        self.symbol = symbol
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _load_date(self, date: str) -> Tuple[np.ndarray, np.ndarray]:
        prefix = f"{self.prices_root}/symbol={self.symbol}/date={date}"
        pattern = prefix.replace("s3://", "") + "/part-*.parquet"
        parts = [f"s3://{p}" for p in self.fs.glob(pattern)]
        if not parts:
            raise FileNotFoundError(f"Missing prices_1s for {self.symbol} date={date} under {prefix}")

        dfs = []
        for p in parts:
            d = pd.read_parquet(p, columns=["id_t", "mid"], engine="pyarrow")
            # downcast tout de suite
            d["id_t"] = pd.to_numeric(d["id_t"], errors="coerce").astype("int64")
            d["mid"] = pd.to_numeric(d["mid"], errors="coerce").astype("float32")
            d = d.dropna(subset=["id_t", "mid"])
            dfs.append(d)

        # concat -> ok mais on limite la RAM
        df = pd.concat(dfs, ignore_index=True, copy=False)

        # IMPORTANT: mid float32 (pas float64)
        df = df.sort_values("id_t", kind="mergesort")
        df = df.drop_duplicates(subset=["id_t"], keep="last")
        sec = df["id_t"].to_numpy(np.int64, copy=False)
        mid = df["mid"].to_numpy(np.float32, copy=False)
        return sec, mid

    def get_date(self, date: str) -> Tuple[np.ndarray, np.ndarray]:
        if date not in self._cache:
            self._cache[date] = self._load_date(date)
        return self._cache[date]

    def get_mid_at(self, sec: int) -> Optional[float]:
        date = pd.to_datetime(sec, unit="s", utc=True).strftime("%Y-%m-%d")
        sec_arr, mid_arr = self.get_date(date)
        # binary search exact
        i = np.searchsorted(sec_arr, sec)
        if i < len(sec_arr) and sec_arr[i] == sec:
            return float(mid_arr[i])
        return None

    def get_window(self, sec0: int, sec1: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns sec, mid for [sec0, sec1] inclusive.
        Window must be within same day OR spans two days (rare in our use; handle both).
        """
        if sec1 < sec0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        d0 = pd.to_datetime(sec0, unit="s", utc=True).strftime("%Y-%m-%d")
        d1 = pd.to_datetime(sec1, unit="s", utc=True).strftime("%Y-%m-%d")

        if d0 == d1:
            sec_arr, mid_arr = self.get_date(d0)
            i0 = np.searchsorted(sec_arr, sec0, side="left")
            i1 = np.searchsorted(sec_arr, sec1, side="right")
            return sec_arr[i0:i1], mid_arr[i0:i1]

        # span 2 dates
        sec_a, mid_a = self.get_window(sec0, int(pd.Timestamp(d0, tz="UTC").timestamp()) + 86399)
        sec_b, mid_b = self.get_window(int(pd.Timestamp(d1, tz="UTC").timestamp()), sec1)
        if sec_a.size == 0:
            return sec_b, mid_b
        if sec_b.size == 0:
            return sec_a, mid_a
        return np.concatenate([sec_a, sec_b]), np.concatenate([mid_a, mid_b])


# -----------------------------
# StageC features & label
# -----------------------------
def compute_mid_features_at_t1(pr: PricesCache, t1_sec: np.ndarray, window_sec: int) -> Dict[str, np.ndarray]:
    """
    Mid-only features at decision t1:
    - ret_1s_bps, ret_5s_bps, ret_20s_bps
    - trend_20s_bps (mid(t1)-mid(t1-20))/mid(t1)*1e4
    - vol_20s_bps: std of 1s returns over [t1-20, t1] (20 steps)
    """
    n = len(t1_sec)
    ret_1 = np.full(n, np.nan, dtype=np.float32)
    ret_5 = np.full(n, np.nan, dtype=np.float32)
    ret_w = np.full(n, np.nan, dtype=np.float32)
    trend_w = np.full(n, np.nan, dtype=np.float32)
    vol_w = np.full(n, np.nan, dtype=np.float32)

    w = int(window_sec)

    for i in range(n):
        s1 = int(t1_sec[i])

        m1 = pr.get_mid_at(s1)
        if m1 is None or not np.isfinite(m1) or m1 <= 0:
            continue

        def _ret(k: int) -> Optional[float]:
            mk = pr.get_mid_at(s1 - k)
            if mk is None or (not np.isfinite(mk)) or mk <= 0:
                return None
            return (m1 / mk - 1.0) * 1e4

        r1 = _ret(1)
        r5 = _ret(5)
        rw = _ret(w)

        if r1 is not None: ret_1[i] = np.float32(r1)
        if r5 is not None: ret_5[i] = np.float32(r5)
        if rw is not None:
            ret_w[i] = np.float32(rw)
            trend_w[i] = np.float32(rw)

        # vol: std of 1s returns over last w seconds
        # needs mid for [t1-w, t1]
        sec_win, mid_win = pr.get_window(s1 - w, s1)
        # expect length w+1 (but tolerate missing)
        if mid_win.size >= max(6, w // 2):
            # compute 1s returns on consecutive points (assume sec increments mostly 1)
            # use pct change on values in order
            r = (mid_win[1:] / mid_win[:-1] - 1.0) * 1e4
            r = r[np.isfinite(r)]
            if r.size >= max(5, w // 3):
                vol_w[i] = np.float32(np.std(r, ddof=0))

    return {
        "fC_mid_ret_1s_bps": ret_1,
        "fC_mid_ret_5s_bps": ret_5,
        f"fC_mid_ret_{w}s_bps": ret_w,
        f"fC_mid_trend_{w}s_bps": trend_w,
        f"fC_mid_vol_{w}s_bps": vol_w,
    }

def first_touch_label_L2(
    pr: PricesCache,
    t1_sec: np.ndarray,
    tp_bps: float,
    sl_bps: float,
    horizon_sec: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3-class label from t1:
    +1 LONG if TP_L touches before SL_L
    -1 SHORT if TP_S touches before SL_S
    0 CANCEL otherwise
    Also returns exit_reason (string) and exit_t_sec (int16).
    """
    n = len(t1_sec)
    y = np.zeros(n, dtype=np.int8)
    exit_t = np.full(n, -1, dtype=np.int16)
    reason = np.full(n, "TIME", dtype=object)

    H = int(horizon_sec)
    tp = float(tp_bps) / 1e4
    sl = float(sl_bps) / 1e4

    INF = 10**9

    for i in range(n):
        s1 = int(t1_sec[i])
        m0 = pr.get_mid_at(s1)
        if m0 is None or not np.isfinite(m0) or m0 <= 0:
            y[i] = 0
            exit_t[i] = -1
            reason[i] = "NO_PRICE"
            continue

        tpL = m0 * (1.0 + tp)
        slL = m0 * (1.0 - sl)
        tpS = m0 * (1.0 - tp)
        slS = m0 * (1.0 + sl)

        # scan future window
        sec_f, mid_f = pr.get_window(s1 + 1, s1 + H)
        if mid_f.size == 0:
            y[i] = 0
            exit_t[i] = -1
            reason[i] = "NO_PRICE"
            continue

        # compute first idx
        # LONG: TP if mid>=tpL; SL if mid<=slL
        # SHORT: TP if mid<=tpS; SL if mid>=slS
        mid = mid_f
        idx_tpL = np.flatnonzero(mid >= tpL)
        idx_slL = np.flatnonzero(mid <= slL)
        idx_tpS = np.flatnonzero(mid <= tpS)
        idx_slS = np.flatnonzero(mid >= slS)

        t_tpL = int(idx_tpL[0]) + 1 if idx_tpL.size else INF
        t_slL = int(idx_slL[0]) + 1 if idx_slL.size else INF
        t_tpS = int(idx_tpS[0]) + 1 if idx_tpS.size else INF
        t_slS = int(idx_slS[0]) + 1 if idx_slS.size else INF

        long_wins = (t_tpL < t_slL) and (t_tpL < INF)
        short_wins = (t_tpS < t_slS) and (t_tpS < INF)

        if long_wins and short_wins:
            # pick the earliest TP among the two
            if t_tpL <= t_tpS:
                y[i] = +1
                exit_t[i] = np.int16(min(t_tpL, 32767))
                reason[i] = "TP_L"
            else:
                y[i] = -1
                exit_t[i] = np.int16(min(t_tpS, 32767))
                reason[i] = "TP_S"
        elif long_wins:
            y[i] = +1
            exit_t[i] = np.int16(min(t_tpL, 32767))
            reason[i] = "TP_L"
        elif short_wins:
            y[i] = -1
            exit_t[i] = np.int16(min(t_tpS, 32767))
            reason[i] = "TP_S"
        else:
            # if SL touched for either side first, we can tag it for audit (still CANCEL label=0)
            t_min = min(t_slL, t_slS)
            if t_min < INF:
                exit_t[i] = np.int16(min(t_min, 32767))
                reason[i] = "SL_FIRST"
            else:
                exit_t[i] = -1
                reason[i] = "TIME"

    return y, reason.astype(object), exit_t


# -----------------------------
# Main builder
# -----------------------------
@dataclass
class Cfg:
    symbol: str
    stageb_root: str
    prices_root: str
    out_root: str
    book_root: str
    trade_root: str

    # timing
    delta_sec: int
    window_sec: int
    horizon_sec: int

    # label params
    tp_bps: float
    sl_bps: float

    # IO
    n_files: int
    max_rows_per_file: int
    seed: int

def build_for_split(fs: s3fs.S3FileSystem, cfg: Cfg, split: str):
    files_all = list_stageb_files(fs, cfg.stageb_root, cfg.symbol, split)
    files = sample_paths(files_all, int(cfg.n_files), int(cfg.seed))
    print(f"[stageC] split={split} files={len(files)} (of {len(files_all)})")

    # caches OUTSIDE loop (important perf)
    pr = PricesCache(fs, cfg.prices_root, cfg.symbol)
    book_cache = BookMonthCache(fs, cfg.book_root, cfg.symbol)
    tr_cache = TradesMonthCache(fs, cfg.trade_root, cfg.symbol)

    out_base = f"{cfg.out_root.rstrip('/')}/symbol={cfg.symbol}/split={split}"
    ensure_dir_s3(fs, out_base)

    part_id = 0
    n_in = 0
    n_allow = 0

    for fpath in files:
        cols_needed = [
            "id_t",
            "audit_p_thr_ev0",
            "stgB_p",
            "stgB_s",
            "stgB_allow",
            "stgB_thr_s",
            "stgB_model_stamp",
            "stgB_model_uri"
        ]
        # tolerate missing optional cols (some parquet writers may omit stgB_thr_s)
        df = pd.read_parquet(
            fpath,
            engine="pyarrow",
            columns=[
                "id_t","audit_p_thr_ev0","stgB_p","stgB_s","stgB_allow","stgB_thr_s",
                "stgB_model_stamp","stgB_model_uri"
            ],
        )
    
        # 최소 contract: id_t + stgB_allow must exist
        must = ["id_t", "stgB_allow"]
        for m in must:
            if m not in df.columns:
                raise RuntimeError(f"[stageC] missing required column '{m}' in {fpath}")

        # keep only what exists
        keep = [c for c in cols_needed if c in df.columns]
        df = df[keep].copy()

        # optional cap rows for dev
        if int(cfg.max_rows_per_file) > 0 and len(df) > int(cfg.max_rows_per_file):
            df = df.sample(n=int(cfg.max_rows_per_file), random_state=int(cfg.seed)).reset_index(drop=True)

        df["id_t"] = utc_dt(df["id_t"])
        df = df.dropna(subset=["id_t", "stgB_allow"])
        if df.empty:
            continue

        # filter allow-only
        allow = (pd.to_numeric(df["stgB_allow"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1)
        n_in += int(len(df))
        n_allow += int(allow.sum())

        df_go = df.loc[allow, :]   # pas de copy ici
        if df_go.empty:
            continue

        # t0/t1 epoch seconds
        CHUNK = 50_000  # ajuste: 20k si encore OOM

        for start in range(0, len(df_go), CHUNK):
            sub = df_go.iloc[start:start+CHUNK]

            # t0/t1 epoch seconds (chunk)
            t0 = sub["id_t"]
            t0_sec = to_epoch_sec(t0)
            t1_sec = (t0_sec + int(cfg.delta_sec)).astype(np.int64)

            # batch par date t1
            t1_date = pd.to_datetime(t1_sec, unit="s", utc=True).strftime("%Y-%m-%d")

            for d in np.unique(t1_date):
                mask = (t1_date == d)

                # slice arrays + slice sub df (IMPORTANT)                
                idx = np.flatnonzero(mask)
                sub_d = sub.iloc[idx]
                t0_sec_d = t0_sec[idx]
                t1_sec_d = t1_sec[idx]

                # features/labels (petits)
                feat = compute_mid_features_at_t1(pr, t1_sec_d, window_sec=int(cfg.window_sec))
                
                # label direction from mid return @ window_sec (ex: 20s)
                ret_col = f"fC_mid_ret_{int(cfg.window_sec)}s_bps"
                ret_arr = feat.get(ret_col)
                if ret_arr is None:
                    ret_arr = np.full(len(t1_sec_d), np.nan, dtype=np.float32)

                eps_bps = 10.0  # ta décision "10 bps"
                label_dir = np.zeros(len(ret_arr), dtype=np.int8)
                label_dir[ret_arr >= eps_bps] = 1
                label_dir[ret_arr <= -eps_bps] = -1

                # IMPORTANT: drop borderline (=0) car on "enlève CANCEL"
                keep_dir = (label_dir != 0)
                if keep_dir.sum() == 0:
                    continue

                # 1) filter rows ONCE
                keep_dir = (label_dir != 0)
                if keep_dir.sum() == 0:
                    continue

                idx2 = np.flatnonzero(keep_dir)

                sub_d = sub_d.iloc[idx2]
                t0_sec_d = t0_sec_d[idx2]
                t1_sec_d = t1_sec_d[idx2]
                y_dir = label_dir[idx2]

                feat = {k: v[idx2] for k, v in feat.items()}

                # 3) joins only AFTER filtering
                t1_utc_d = pd.to_datetime(t1_sec_d, unit="s", utc=True)
                b_at_t1 = book_cache.asof(t1_utc_d)
                t_at_t1 = tr_cache.asof(t1_utc_d)

                # 4) now labels that need future window
                yC, reasonC, exitC = first_touch_label_L2(
                    pr,
                    t1_sec_d,
                    tp_bps=float(cfg.tp_bps),
                    sl_bps=float(cfg.sl_bps),
                    horizon_sec=int(cfg.horizon_sec),
                )
 
                t0_dt = pd.to_datetime(t0_sec_d, unit="s", utc=True)
                
                # output (petit)
                out = pd.DataFrame({
                    "id_symbol": cfg.symbol,
                    "id_t0": pd.to_datetime(t0_sec_d, unit="s", utc=True),
                    "id_t1": pd.to_datetime(t1_sec_d, unit="s", utc=True),
                    "id_year": t0_dt.year.astype(np.int16),
                    "id_month": t0_dt.month.astype(np.int8),
                    "id_date": date_str_from_epoch(t0_sec_d),

                    "cfg_delta_sec": np.int16(cfg.delta_sec),
                    "cfg_window_sec": np.int16(cfg.window_sec),
                    "cfg_horizon_sec": np.int16(cfg.horizon_sec),
                    "cfg_tp_bps": np.float32(cfg.tp_bps),
                    "cfg_sl_bps": np.float32(cfg.sl_bps),

                    "stgB_p": col_as_float32(sub_d, "stgB_p"),
                    "stgB_p_thr_ev0": col_as_float32(sub_d, "audit_p_thr_ev0"),
                    "stgB_s": col_as_float32(sub_d, "stgB_s"),
                    "stgB_thr_s": col_as_float32(sub_d, "stgB_thr_s"),
                    "stgB_allow": np.ones(len(sub_d), dtype=np.int8),

                    "label_C": yC.astype(np.int8),
                    "label_C_exit_reason": pd.Series(reasonC, dtype="string"),
                    "label_C_exit_t_sec": exitC.astype(np.int16),

                    "label_dir": y_dir.astype(np.int8),
                })

                # add joins
                out["fC_flow_signed_vol_20s"] = pd.to_numeric(
                    t_at_t1.get("flow_signed_vol_20s", np.nan), errors="coerce"
                ).to_numpy(np.float32, copy=False)

                out["fC_flow_aggr_buy_ratio_20s"] = pd.to_numeric(
                    t_at_t1.get("flow_aggr_buy_ratio_20s", np.nan), errors="coerce"
                ).to_numpy(np.float32, copy=False)

                out["fC_spread_bps_at_t1"] = pd.to_numeric(
                    b_at_t1["spread_bps"], errors="coerce"
                ).to_numpy(np.float32, copy=False)

                out["fC_quote_updates_20s"] = pd.to_numeric(
                    b_at_t1["quote_updates_20s"], errors="coerce"
                ).fillna(0).to_numpy(np.int32, copy=False)

                # --- book features (robustes: NaN si pas dispo) ---
                def get_b(col, default=np.nan):
                    return pd.to_numeric(b_at_t1[col], errors="coerce").to_numpy(np.float32, copy=False) if col in b_at_t1.columns \
                        else np.full(len(out), default, dtype=np.float32)

                out["fC_book_imb_top5"] = get_b("imb_top5")
                out["fC_book_imb_L15"]  = get_b("imb_L15")
                out["fC_book_bid_depth_L15"] = get_b("bid_depth_L15")
                out["fC_book_ask_depth_L15"] = get_b("ask_depth_L15")
                out["fC_microprice_diff_bps"] = get_b("microprice_diff_bps")
                out["fC_queue_imb"] = get_b("queue_imb")

                # si tu n'as pas imb dans ton slice book, commente ces 2 lignes
                if "imb_L1" in b_at_t1.columns:
                    out["fC_book_imb_L1_at_t1"] = pd.to_numeric(b_at_t1["imb_L1"], errors="coerce").to_numpy(np.float32, copy=False)
                if "imb_L5" in b_at_t1.columns:
                    out["fC_book_imb_L5_at_t1"] = pd.to_numeric(b_at_t1["imb_L5"], errors="coerce").to_numpy(np.float32, copy=False)

                # model stamp/uri
                if "stgB_model_stamp" in sub_d.columns:
                    out["stgB_model_stamp"] = sub_d["stgB_model_stamp"].astype("string").fillna("")
                if "stgB_model_uri" in sub_d.columns:
                    out["stgB_model_uri"] = sub_d["stgB_model_uri"].astype("string").fillna("")

                # add mid features
                for k, v in feat.items():
                    out[k] = v
 
                eps = 1e-3

                vol_col = f"fC_mid_vol_{int(cfg.window_sec)}s_bps"
                trend_col = f"fC_mid_trend_{int(cfg.window_sec)}s_bps"

                vol_arr = pd.to_numeric(out.get(vol_col, np.nan), errors="coerce") \
                            .to_numpy(np.float32, copy=False)

                trend_arr = pd.to_numeric(out.get(trend_col, np.nan), errors="coerce") \
                            .to_numpy(np.float32, copy=False)

                flow = pd.to_numeric(out["fC_flow_signed_vol_20s"], errors="coerce") \
                        .fillna(0.0) \
                        .to_numpy(np.float32, copy=False)

                out["fC_flow_voladj_20s"] = (flow / (vol_arr + eps)).astype(np.float32)
                out["fC_flow_abs_voladj_20s"] = (np.abs(flow) / (vol_arr + eps)).astype(np.float32)

                trend_strength = np.abs(trend_arr) / (vol_arr + eps)

                w = int(cfg.window_sec)
                out[f"fC_regime_trend_strength_{w}s"] = trend_strength.astype(np.float32)
                out[f"fC_regime_is_trend_{w}s"] = (trend_strength >= 1.0).astype(np.int8)

                # write partition by date (t0 date)
                for d0, g in out.groupby("id_date", sort=True):
                    out_dir = f"{out_base}/date={d0}"
                    ensure_dir_s3(fs, out_dir)
                    uri = f"{out_dir}/part-{part_id:06d}.parquet"
                    write_parquet_s3(fs, uri, g.reset_index(drop=True))
                    part_id += 1

                # cleanup mini-batch (6)
                del sub_d, b_at_t1, t_at_t1, feat, out, yC, reasonC, exitC, t1_utc_d
                gc.collect()

            # cleanup chunk (6)
            del sub, t0_sec, t1_sec, t1_date
            gc.collect()

        print(f"[stageC] {split}: read={len(df):,} allow={int(allow.sum()):,} (chunked)")
        del df, df_go
        gc.collect()

def main():
    ap = argparse.ArgumentParser("Build StageC dataset v1 (go-only from StageB + midprice 1s).")

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--stageb-root", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/scored")
    ap.add_argument("--prices-root", default="s3://tradebot-config-tokyo/data/stageA/prices_1s")

    ap.add_argument("--book-root", default="s3://tradebot-config-tokyo/data/book")
    ap.add_argument("--trade-root", default="s3://tradebot-config-tokyo/data/trade")

    ap.add_argument("--out-root", default="s3://tradebot-config-tokyo/data/stageC/dataset=v1")

    # Timing
    ap.add_argument("--delta-sec", type=int, default=30)
    ap.add_argument("--window-sec", type=int, default=20)
    ap.add_argument("--horizon-sec", type=int, default=120)

    # Label params (bps)
    ap.add_argument("--tp-bps", type=float, default=8.0)
    ap.add_argument("--sl-bps", type=float, default=8.0)

    # IO dev controls
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--n-files", type=int, default=0, help="0=all files in split")
    ap.add_argument("--max-rows-per-file", type=int, default=0, help="0=disable cap")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    fs = s3fs.S3FileSystem()

    cfg = Cfg(
        symbol=str(args.symbol),
        stageb_root=str(args.stageb_root),
        prices_root=str(args.prices_root),
        out_root=str(args.out_root),
        book_root=str(args.book_root),
        trade_root=str(args.trade_root),
        delta_sec=int(args.delta_sec),
        window_sec=int(args.window_sec),
        horizon_sec=int(args.horizon_sec),
        tp_bps=float(args.tp_bps),
        sl_bps=float(args.sl_bps),
        n_files=int(args.n_files),
        max_rows_per_file=int(args.max_rows_per_file),
        seed=int(args.seed),
    )

    print(json.dumps({
        "timestamp": stamp(),
        "symbol": cfg.symbol,
        "stageb_root": cfg.stageb_root,
        "prices_root": cfg.prices_root,
        "out_root": cfg.out_root,
        "delta_sec": cfg.delta_sec,
        "window_sec": cfg.window_sec,
        "horizon_sec": cfg.horizon_sec,
        "tp_bps": cfg.tp_bps,
        "sl_bps": cfg.sl_bps,
        "splits": args.splits,
        "n_files": cfg.n_files,
        "max_rows_per_file": cfg.max_rows_per_file,
    }, indent=2))

    for sp in args.splits:
        build_for_split(fs, cfg, split=str(sp).strip())


if __name__ == "__main__":
    main()