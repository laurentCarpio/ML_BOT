#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage0/core_data_rightedge.py
Core data utilities (right-edge t0, causal, anti-OOM streaming)
- load candles yearly (1m)
- stream trades/book monthly -> aggregate to t0 right-edge
- build candle features at t0
- build label future range (based on 1m, aligned on t0)
"""

import gc
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# -----------------
# LOG
# -----------------
def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------
# TIME / BUCKETS
# -----------------
def to_utc_ts(x) -> pd.Series:
    return pd.to_datetime(x, utc=True, errors="coerce")


def bucket_right_edge(ts: pd.Series, freq_min: int) -> pd.DatetimeIndex:
    """
    Right-edge buckets: label is END of bucket.
    freq=5min: 00:00:01 -> 00:05:00
    """
    return ts.dt.ceil(f"{freq_min}min")


def month_range(start: str, end: str) -> Iterable[str]:
    s = pd.Timestamp(start, tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    e = pd.Timestamp(end, tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cur = s
    while cur <= e:
        yield cur.strftime("%Y-%m")
        cur = (cur + pd.offsets.MonthBegin(1))


# -----------------
# CANDLES (YEARLY)
# -----------------
def load_candles_yearly(s3_root: str, symbol: str, years: List[int]) -> pd.DataFrame:
    dfs = []

    for y in years:
        path = f"{s3_root.rstrip('/')}/{symbol}/{symbol}-1m-{y}.parquet"
        log(f"[load] {path}")

        df = pd.read_parquet(path)

        # Case A: timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        else:
            # Case B: timestamp already index
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[~df.index.isna()]

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].sort_index()
        dfs.append(df)

    out = pd.concat(dfs, axis=0)
    if out.index.min() < pd.Timestamp("2023-01-01", tz="UTC"):
        raise ValueError(f"candles index looks wrong: min={out.index.min()} max={out.index.max()}")

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def candles_to_t0_rightedge(candles_1m: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    # right-edge: label='right', closed='right'
    return candles_1m.resample(f"{freq_min}min", closed="right", label="right").last()


def candle_features_at_t0(candles_1m: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    t0 = candles_to_t0_rightedge(candles_1m, freq_min)
    close = t0["close"].astype(float)
    high = t0["high"].astype(float)
    low = t0["low"].astype(float)

    ret_bps = close.pct_change(fill_method=None) * 1e4

    w_1h = int(60 / freq_min)
    w_30m = max(1, int(30 / freq_min))
    w_15m = max(1, int(15 / freq_min))

    feat = pd.DataFrame(index=t0.index)
    feat["close"] = close

    feat["c_absret_15m"] = ret_bps.abs().rolling(w_15m, min_periods=max(2, w_15m // 2)).mean()
    feat["c_absret_30m"] = ret_bps.abs().rolling(w_30m, min_periods=max(2, w_30m // 2)).mean()
    feat["c_absret_1h"] = ret_bps.abs().rolling(w_1h, min_periods=max(3, w_1h // 3)).mean()

    feat["c_vol_30m"] = ret_bps.rolling(w_30m, min_periods=max(2, w_30m // 2)).std()
    feat["c_vol_1h"] = ret_bps.rolling(w_1h, min_periods=max(3, w_1h // 3)).std()

    feat["c_range_30m_bps"] = (
        high.rolling(w_30m, min_periods=max(2, w_30m // 2)).max()
        - low.rolling(w_30m, min_periods=max(2, w_30m // 2)).min()
    ) / close.replace(0, np.nan) * 1e4

    feat["c_range_1h_bps"] = (
        high.rolling(w_1h, min_periods=max(3, w_1h // 3)).max()
        - low.rolling(w_1h, min_periods=max(3, w_1h // 3)).min()
    ) / close.replace(0, np.nan) * 1e4

    feat["c_compression"] = feat["c_range_30m_bps"] / feat["c_range_1h_bps"].replace(0, np.nan)

    feat["c_trend_30m_bps"] = close.pct_change(w_30m, fill_method=None) * 1e4
    feat["c_trend_1h_bps"] = close.pct_change(w_1h, fill_method=None) * 1e4

    return feat.replace([np.inf, -np.inf], np.nan)


def build_label_future_range(
    candles_1m: pd.DataFrame,
    freq_min: int,
    horizon_min: int,
    train_end: str,
    label_q: float,
) -> pd.DataFrame:
    """
    y(t0): future range over horizon_min computed on 1m candles, indexed by t0 right-edge.

    Important: we compute future range starting AFTER t0 (shift(-1)) for stricter causality.
    """
    t0 = candles_to_t0_rightedge(candles_1m, freq_min)
    close_t0 = t0["close"].astype(float)

    H = int(horizon_min)
    high = candles_1m["high"]
    low = candles_1m["low"]

    fh = high.shift(-1).rolling(H, min_periods=H).max().shift(-(H - 1))
    fl = low.shift(-1).rolling(H, min_periods=H).min().shift(-(H - 1))

    fh = fh.reindex(t0.index)
    fl = fl.reindex(t0.index)

    range_h_bps = (fh - fl) / close_t0.replace(0, np.nan) * 1e4

    train_end_ts = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    thr = range_h_bps.loc[:train_end_ts].quantile(label_q)

    out = pd.DataFrame(index=t0.index)
    out["range_1h_bps"] = range_h_bps
    out["label"] = (range_h_bps >= thr).astype(np.int8)
    out["label_thr_bps_train"] = float(thr)
    return out


# -----------------
# STREAM AGG TRADES
# -----------------
def stream_agg_trades_month(path: str, freq_min: int, batch_rows: int = 300_000) -> pd.DataFrame:
    """
    Streaming aggregation to right-edge buckets:
    - tr_count, tr_vol, tr_vwap
    - tr_signed_vol, tr_aggr_buy_ratio (if side exists)
    """
    pf = pq.ParquetFile(path)
    schema_names = pf.schema.names

    cols = []
    for c in ["timestamp", "price", "qty", "side"]:
        if c in schema_names:
            cols.append(c)

    if "timestamp" not in cols or "price" not in cols:
        raise ValueError(f"Trades missing timestamp/price in {path}")

    if "qty" not in cols:
        if "quantity" in schema_names:
            cols.append("quantity")
        else:
            raise ValueError(f"Trades missing qty/quantity in {path}")

    acc = {}  # bucket_ns -> accum dict

    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        df = batch.to_pandas()
        ts = to_utc_ts(df["timestamp"])
        df = df.loc[~ts.isna()].copy()
        ts = ts.loc[df.index]

        bucket = bucket_right_edge(ts, freq_min)
        bucket_ns = bucket.astype("int64").to_numpy()

        qty_col = "qty" if "qty" in df.columns else "quantity"
        qty = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0).to_numpy()
        px = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).to_numpy()

        if "side" in df.columns:
            side = df["side"].astype(str).str.lower().to_numpy()
            is_buy = (side == "buy")
            is_sell = (side == "sell")
            buy_qty = qty * is_buy
            sell_qty = qty * is_sell
        else:
            buy_qty = None
            sell_qty = None

        order = np.argsort(bucket_ns)
        bucket_ns = bucket_ns[order]
        qty = qty[order]
        px = px[order]
        if buy_qty is not None:
            buy_qty = buy_qty[order]
            sell_qty = sell_qty[order]

        unique_b, idx_start = np.unique(bucket_ns, return_index=True)
        idx_end = np.append(idx_start[1:], len(bucket_ns))

        for b, s, e in zip(unique_b, idx_start, idx_end):
            qsum = float(qty[s:e].sum())
            cnt = float(e - s)
            pxqty = float((px[s:e] * qty[s:e]).sum())

            if b not in acc:
                acc[b] = {
                    "tr_count": 0.0,
                    "tr_vol": 0.0,
                    "tr_px_qty": 0.0,
                    "tr_buy_vol": 0.0,
                    "tr_sell_vol": 0.0,
                }

            acc[b]["tr_count"] += cnt
            acc[b]["tr_vol"] += qsum
            acc[b]["tr_px_qty"] += pxqty

            if buy_qty is not None:
                acc[b]["tr_buy_vol"] += float(buy_qty[s:e].sum())
                acc[b]["tr_sell_vol"] += float(sell_qty[s:e].sum())

        del df
        gc.collect()

    if not acc:
        return pd.DataFrame()

    keys = list(acc.keys())
    idx = pd.to_datetime(np.array(keys, dtype=np.int64), utc=True)

    out = pd.DataFrame(index=idx)
    out["tr_count"] = [acc[k]["tr_count"] for k in keys]
    out["tr_vol"] = [acc[k]["tr_vol"] for k in keys]

    px_qty = np.array([acc[k]["tr_px_qty"] for k in keys], dtype=float)
    vol = out["tr_vol"].to_numpy(dtype=float)
    out["tr_vwap"] = px_qty / np.where(vol == 0, np.nan, vol)

    buy = np.array([acc[k]["tr_buy_vol"] for k in keys], dtype=float)
    sell = np.array([acc[k]["tr_sell_vol"] for k in keys], dtype=float)

    out["tr_signed_vol"] = buy - sell
    out["tr_aggr_buy_ratio"] = np.where((buy + sell) == 0, 0.5, buy / (buy + sell))

    out = out.sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def load_trades_t0_streaming(s3_root: str, symbol: str, months: List[str], freq_min: int, batch_rows: int) -> pd.DataFrame:
    parts = []
    for ym in months:
        path = f"{s3_root.rstrip('/')}/{symbol}/{ym}.parquet"
        log(f"[load] {path}")
        part = stream_agg_trades_month(path, freq_min=freq_min, batch_rows=batch_rows)
        parts.append(part)

    out = pd.concat(parts, axis=0) if parts else pd.DataFrame()
    if len(out) == 0:
        return out
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# -----------------
# STREAM BOOK (LAST SNAPSHOT PER BUCKET)
# -----------------
def stream_book_last_month(path: str, freq_min: int, batch_rows: int = 300_000) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    schema_names = pf.schema.names

    needed = ["timestamp", "bid_0_price", "ask_0_price", "bid_0_size", "ask_0_size"]
    for c in needed:
        if c not in schema_names:
            raise ValueError(f"Book missing {c} in {path}")

    last = {}  # bucket_ns -> (ts_ns, bid, ask, bsz, asz)

    for batch in pf.iter_batches(batch_size=batch_rows, columns=needed):
        df = batch.to_pandas()

        ts = to_utc_ts(df["timestamp"])
        ok = ~ts.isna()
        if not ok.any():
            del df
            gc.collect()
            continue

        df = df.loc[ok].copy()
        ts = ts.loc[df.index]

        bucket = bucket_right_edge(ts, freq_min)
        bucket_ns = bucket.astype("int64").to_numpy()
        ts_ns = ts.astype("int64").to_numpy()

        bid = pd.to_numeric(df["bid_0_price"], errors="coerce").to_numpy()
        ask = pd.to_numeric(df["ask_0_price"], errors="coerce").to_numpy()
        bsz = pd.to_numeric(df["bid_0_size"], errors="coerce").fillna(0.0).to_numpy()
        asz = pd.to_numeric(df["ask_0_size"], errors="coerce").fillna(0.0).to_numpy()

        for b, t, bi, ak, bs, a_s in zip(bucket_ns, ts_ns, bid, ask, bsz, asz):
            cur = last.get(b)
            if cur is None or t > cur[0]:
                last[b] = (t, float(bi), float(ak), float(bs), float(a_s))

        del df
        gc.collect()

    if not last:
        return pd.DataFrame()

    keys = list(last.keys())
    idx = pd.to_datetime(np.array(keys, dtype=np.int64), utc=True)
    out = pd.DataFrame(index=idx)

    bid0 = np.array([last[k][1] for k in keys], dtype=float)
    ask0 = np.array([last[k][2] for k in keys], dtype=float)
    bsz0 = np.array([last[k][3] for k in keys], dtype=float)
    asz0 = np.array([last[k][4] for k in keys], dtype=float)

    mid = (bid0 + ask0) / 2.0
    spread_bps = (ask0 - bid0) / np.where(mid == 0, np.nan, mid) * 1e4
    micro = (ask0 * bsz0 + bid0 * asz0) / np.where((bsz0 + asz0) == 0, np.nan, (bsz0 + asz0))
    micro_diff_bps = (micro - mid) / np.where(mid == 0, np.nan, mid) * 1e4
    imb = (bsz0 - asz0) / np.where((bsz0 + asz0) == 0, np.nan, (bsz0 + asz0))

    out["bk_bid0"] = bid0
    out["bk_ask0"] = ask0
    out["bk_mid"] = mid
    out["bk_bid0_sz"] = bsz0
    out["bk_ask0_sz"] = asz0
    out["bk_spread_bps"] = spread_bps
    out["bk_micro_diff_bps"] = micro_diff_bps
    out["bk_imb_L1"] = imb

    out = out.sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def load_book_t0_streaming(s3_root: str, symbol: str, months: List[str], freq_min: int, batch_rows: int) -> pd.DataFrame:
    parts = []
    for ym in months:
        path = f"{s3_root.rstrip('/')}/{symbol}/{ym}.parquet"
        log(f"[load] {path}")
        part = stream_book_last_month(path, freq_min=freq_min, batch_rows=batch_rows)
        parts.append(part)

    out = pd.concat(parts, axis=0) if parts else pd.DataFrame()
    if len(out) == 0:
        return out
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# -----------------
# EVENT LOGIC (breakout) + t0 OHLC helper
# -----------------
def build_t0_ohlc(candles_1m: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    t0 = candles_to_t0_rightedge(candles_1m, freq_min)
    out = pd.DataFrame(index=t0.index)
    out["t0_high"] = t0["high"].astype(float)
    out["t0_low"] = t0["low"].astype(float)
    out["t0_close"] = t0["close"].astype(float)
    return out


def breakout_events(t0_ohlc: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    hi_prev = t0_ohlc["t0_high"].shift(1).rolling(lookback_bars, min_periods=lookback_bars).max()
    lo_prev = t0_ohlc["t0_low"].shift(1).rolling(lookback_bars, min_periods=lookback_bars).min()
    close = t0_ohlc["t0_close"]

    up = close > hi_prev
    dn = close < lo_prev

    ev = pd.DataFrame(index=t0_ohlc.index)
    ev["is_event"] = (up | dn).astype(np.int8)
    ev["dir"] = np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)  # +1/-1/0
    return ev