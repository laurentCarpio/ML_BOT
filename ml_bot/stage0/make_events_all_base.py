#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage0/make_events_all_base.py

Expert B: directionnel (follow-through) sur breakout events.

- Regime: breakout at t0 (right-edge)
- Risk unit: R = ATR(14) on 1h bars, converted to bps at t0
- SL = 1.0 * R, TP = 1.3 * R
- Horizon: H minutes (default 90)
- Outcome (3 states):
    +1 = TP_FIRST
     0 = SL_FIRST
    -1 = TIMEOUT (neither)
- Train binary model: y=1 if TP_FIRST else 0

Writes:
  events_b.parquet with:
    t0, side, entry_px, dir, pB, R_bps, tp_bps, sl_bps, outcome

Reports:
  AUC/Lift on TEST restricted to events
  Monthly stability table (TEST)
"""

import argparse
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from ml_bot.stage0.core_data_rightedge import (
    log,
    load_candles_yearly,
    month_range,
    load_trades_t0_streaming,
    load_book_t0_streaming,
    candles_to_t0_rightedge,
    candle_features_at_t0,
)

# -------------------------
# IO
# -------------------------
def write_parquet_any(df: pd.DataFrame, path: str):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        u = urlparse(path)
        bucket = u.netloc
        key = u.path.lstrip("/")
        s3_path = f"{bucket}/{key}"
        fs = pafs.S3FileSystem()
        with fs.open_output_stream(s3_path) as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, path)

def write_events_all_base(df_ev: pd.DataFrame, args, out_path: str):
    required = ["t0_close", "dir", "R_bps"]
    missing = [c for c in required if c not in df_ev.columns]
    if missing:
        raise RuntimeError(f"events_all_base missing columns: {missing}")

    out = df_ev.copy().sort_index()
    out["t0"] = pd.to_datetime(out.index, utc=True)
    out["symbol"] = args.symbol
    out["decision_freq_min"] = int(args.decision_freq_min)
    out["lookback_bars"] = int(args.lookback_bars)
    out["side"] = np.where(out["dir"].to_numpy() > 0, "long", "short")

    write_parquet_any(out.reset_index(drop=True), out_path)

# -------------------------
# Breakout events (same idea as Expert A)
# -------------------------
def build_t0_hilo(candles_1m: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    t0 = candles_to_t0_rightedge(candles_1m, freq_min)
    out = pd.DataFrame(index=t0.index)
    out["t0_high"] = t0["high"].astype(float)
    out["t0_low"] = t0["low"].astype(float)
    out["t0_close"] = t0["close"].astype(float)
    return out


def breakout_events(t0_hilo: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    hi_prev = t0_hilo["t0_high"].shift(1).rolling(lookback_bars, min_periods=lookback_bars).max()
    lo_prev = t0_hilo["t0_low"].shift(1).rolling(lookback_bars, min_periods=lookback_bars).min()
    close = t0_hilo["t0_close"]

    up = close > hi_prev
    dn = close < lo_prev

    ev = pd.DataFrame(index=t0_hilo.index)
    ev["is_event"] = (up | dn).astype(np.int8)
    ev["dir"] = np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)
    return ev


# -------------------------
# ATR(14) on 1h bars -> bps at t0
# -------------------------
def atr_1h_bps_at_t0(candles_1m: pd.DataFrame, t0_index: pd.DatetimeIndex, freq_min: int, atr_period: int = 14) -> pd.Series:
    # 1h right-edge bars
    h1 = candles_1m.resample("60min", closed="right", label="right").agg(
        {"high": "max", "low": "min", "close": "last"}
    ).dropna()

    prev_close = h1["close"].shift(1)
    tr = pd.concat(
        [
            (h1["high"] - h1["low"]).abs(),
            (h1["high"] - prev_close).abs(),
            (h1["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(atr_period, min_periods=atr_period).mean()

    # Align ATR to t0 grid: forward-fill ATR from 1h to each 5min t0
    atr_t0 = atr.reindex(t0_index, method="ffill")

    # Convert to bps relative to t0 close
    t0_close = candles_to_t0_rightedge(candles_1m, freq_min)["close"].reindex(t0_index).astype(float)
    atr_bps = atr_t0 / t0_close.replace(0, np.nan) * 1e4
    return atr_bps.astype(float)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("stage0 — make events_all_base (right-edge, feature-rich)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--s3-candles-root", default="s3://tradebot-config-tokyo/data/bougie")
    ap.add_argument("--s3-trades-root",  default="s3://tradebot-config-tokyo/data/trade")
    ap.add_argument("--s3-book-root",    default="s3://tradebot-config-tokyo/data/book")

    ap.add_argument("--events-base-out", default="s3://tradebot-config-tokyo/data/s1/events_all_base.parquet")

    ap.add_argument("--decision-freq-min", type=int, default=5)
    ap.add_argument("--lookback-bars", type=int, default=12)
    ap.add_argument("--end", default="2025-10-31")

    ap.add_argument("--batch-rows", type=int, default=300_000)
    ap.add_argument("--atr-period", type=int, default=14)

    args = ap.parse_args()

    log("===================================")
    log("STAGE0 — MAKE EVENTS_ALL_BASE (right-edge)")
    log("===================================")

    # 1) Load candles
    years = sorted({pd.Timestamp("2024-01-01").year, pd.Timestamp(args.end).year})
    years = list(range(min(years), max(years) + 1))
    candles = load_candles_yearly(args.s3_candles_root, args.symbol, years=years)
    log(f"[candles] rows={len(candles):,} span={candles.index.min()} -> {candles.index.max()}")

    # 2) Load trades/book aggregated to t0
    months = list(month_range("2024-01-01", args.end))
    trades_t0 = load_trades_t0_streaming(
        args.s3_trades_root, args.symbol, months, args.decision_freq_min, args.batch_rows
    )
    book_t0 = load_book_t0_streaming(
        args.s3_book_root, args.symbol, months, args.decision_freq_min, args.batch_rows
    )
    log(f"[trades_t0] rows={len(trades_t0):,}")
    log(f"[book_t0]   rows={len(book_t0):,}")

    # 3) Build t0 features + t0 hilo
    c_feat = candle_features_at_t0(candles, args.decision_freq_min)
    t0_hilo = build_t0_hilo(candles, args.decision_freq_min)
    t0_index = c_feat.index

    # align trades/book
    trades_t0 = trades_t0.reindex(t0_index)
    for c in ["tr_count", "tr_vol", "tr_signed_vol"]:
        if c in trades_t0.columns:
            trades_t0[c] = trades_t0[c].fillna(0.0)
    if "tr_aggr_buy_ratio" in trades_t0.columns:
        trades_t0["tr_aggr_buy_ratio"] = trades_t0["tr_aggr_buy_ratio"].fillna(0.5)

    book_t0 = book_t0.reindex(t0_index).sort_index().ffill()

    # neutral VWAP fill
    df = pd.concat([c_feat, trades_t0, book_t0, t0_hilo], axis=1)
    if "tr_vwap" in df.columns and "close" in df.columns:
        df["tr_vwap"] = df["tr_vwap"].fillna(df["close"])

    df = df.dropna(subset=["t0_close", "t0_high", "t0_low"])
    log(f"[df] rows={len(df):,} cols={df.shape[1]}")

    # 4) Events
    ev = breakout_events(df[["t0_high", "t0_low", "t0_close"]], lookback_bars=int(args.lookback_bars))
    df_ev = df.loc[ev["is_event"].to_numpy() == 1].copy()
    df_ev["dir"] = ev.loc[df_ev.index, "dir"].astype(np.int8)
    log(f"[events] breakout events (all): n={len(df_ev):,} (lookback_bars={args.lookback_bars})")

    if len(df_ev) == 0:
        raise RuntimeError("No events found. Check lookback-bars/frequency.")

    # 5) Compute R (ATR 1h bps at t0)
    atr_bps = atr_1h_bps_at_t0(candles, t0_index, args.decision_freq_min, atr_period=args.atr_period)
    df_ev["R_bps"] = atr_bps.reindex(df_ev.index).astype(float)
    df_ev["R_bps"] = df_ev["R_bps"].clip(lower=5.0, upper=500.0)

    # 6) Sanity logs
    log(f"[base] rows={len(df_ev):,} cols={df_ev.shape[1]}")
    log(f"[base] span={df_ev.index.min()} -> {df_ev.index.max()}")
    log(f"[base] dir+ ratio={(df_ev['dir']>0).mean():.3f}")
    r = df_ev["R_bps"].to_numpy(dtype=float)
    log(f"[base] R_bps p50={np.nanmedian(r):.2f} min={np.nanmin(r):.2f} max={np.nanmax(r):.2f}")

    # safety: make sure core columns are present
    required = ["t0_close", "dir", "R_bps"]
    missing = [c for c in required if c not in df_ev.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in df_ev: {missing}")

    # optional: reorder to keep them visible at the front
    front = ["t0_close", "dir", "R_bps"]
    rest = [c for c in df_ev.columns if c not in front]
    df_ev = df_ev[front + rest]

    # 7) Write base parquet (feature-rich)
    log(f"[write] {args.events_base_out} rows={len(df_ev):,}")
    write_events_all_base(df_ev, args, args.events_base_out)

    log("✅ Done.")

if __name__ == "__main__":
    main()