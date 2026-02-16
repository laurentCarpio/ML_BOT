#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage0/make_events_all.py

Core goals:
- Train ML on TRAIN (right-edge t0, embargo)
- Build events (breakout-only by default) with side/entry_px/p
- Write events_all.parquet (local or s3://) for replay / stageS1_1_run
- Optional --report: print AUC + Lift on TEST restricted to events

Depends on:
- ml_bot.stage0.core_data_rightedge providing:
  log, load_candles_yearly, month_range,
  load_trades_t0_streaming, load_book_t0_streaming,
  candles_to_t0_rightedge, candle_features_at_t0,
  build_label_future_range, fit_lgbm
"""

import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from sklearn.metrics import roc_auc_score
from ml_bot.stage0.core_model_lgbm import fit_lgbm, predict_proba

from ml_bot.stage0.core_data_rightedge import (
    log,
    load_candles_yearly,
    month_range,
    load_trades_t0_streaming,
    load_book_t0_streaming,
    candles_to_t0_rightedge,
    candle_features_at_t0,
    build_label_future_range,
)


# -------------------------
# IO
# -------------------------
def write_parquet_any(df: pd.DataFrame, path: str):
    """Write parquet locally or to s3:// using pyarrow.fs (no s3fs)."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        u = urlparse(path)
        bucket = u.netloc
        key = u.path.lstrip("/")
        s3_path = f"{bucket}/{key}"
        fs = pafs.S3FileSystem()  # uses IAM/env creds
        with fs.open_output_stream(s3_path) as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, path)


# -------------------------
# Metrics
# -------------------------
def lift_at_topk(y_true: pd.Series, score: np.ndarray, top_frac: float = 0.20) -> float:
    y = y_true.to_numpy(dtype=int)
    n = len(y)
    if n == 0:
        return np.nan
    k = max(1, int(n * float(top_frac)))
    idx = np.argsort(-score)[:k]
    base = float(y.mean())
    top = float(y[idx].mean())
    return float(top / base) if base > 0 else np.nan


# -------------------------
# Events logic (breakout)
# -------------------------
def build_t0_hilo(candles_1m: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    """Keep high/low/close at t0 right-edge to define breakouts."""
    t0 = candles_to_t0_rightedge(candles_1m, freq_min)
    out = pd.DataFrame(index=t0.index)
    out["t0_high"] = t0["high"].astype(float)
    out["t0_low"] = t0["low"].astype(float)
    out["t0_close"] = t0["close"].astype(float)
    return out

def breakout_events(t0_hilo: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    """
    Breakout event at t0 if close breaks above prior N-bar high or below prior N-bar low.
    dir: +1 (up breakout), -1 (down breakout)
    """
    hi_prev = t0_hilo["t0_high"].shift(1).rolling(lookback_bars, min_periods=lookback_bars).max()
    lo_prev = t0_hilo["t0_low"].shift(1).rolling(lookback_bars, min_periods=lookback_bars).min()
    close = t0_hilo["t0_close"]

    up = close > hi_prev
    dn = close < lo_prev

    ev = pd.DataFrame(index=t0_hilo.index)
    ev["event"] = (up | dn)
    ev["dir"] = np.where(up, 1, np.where(dn, -1, 0)).astype(int)
    return ev

def build_events_df(df_ev: pd.DataFrame, ev_dir: pd.Series, p: np.ndarray) -> pd.DataFrame:
    """
    Output schema for stageS1_1_run / S1.1:
      - t0 (UTC)
      - side ("long"/"short")
      - entry_px (float)
      - p (float)
    """
    out = pd.DataFrame()
    out["t0"] = pd.to_datetime(df_ev.index, utc=True)
    out["side"] = np.where(ev_dir.to_numpy() > 0, "long", "short")
    out["entry_px"] = df_ev["t0_close"].astype(float).to_numpy()
    out["p"] = p.astype(float)
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("stage0 — make events_all.parquet (with optional ML report)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--s3-candles-root", default="s3://tradebot-config-tokyo/data/bougie")
    ap.add_argument("--s3-trades-root",  default="s3://tradebot-config-tokyo/data/trade")
    ap.add_argument("--s3-book-root",    default="s3://tradebot-config-tokyo/data/book")

    ap.add_argument("--events-out", default="s3://tradebot-config-tokyo/data/s1/events_all.parquet")
    ap.add_argument("--decision-freq-min", type=int, default=5)
    ap.add_argument("--horizon-min", type=int, default=60)
    ap.add_argument("--label-q", type=float, default=0.70)  

    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--end", default="2025-10-31")

    ap.add_argument("--batch-rows", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)

    # event params
    ap.add_argument("--lookback-bars", type=int, default=12, help="breakout lookback in t0 bars (12 @5min=60min)")

    # reporting
    ap.add_argument("--report", action="store_true", help="Print AUC/Lift on TEST restricted to events")
    ap.add_argument("--top-frac", type=float, default=0.20, help="Lift computed on top fraction (e.g. 0.2 = top20%)")
    ap.add_argument("--min-events-test", type=int, default=1000, help="Minimum #events in TEST to report metrics")

    ap.add_argument("--events-scope", choices=["test", "all", "oos"], default="all")

    args = ap.parse_args()

    log("===================================")
    log("STAGE0 — MAKE EVENTS_ALL (right-edge)")
    log("===================================")

    # 1) Load candles
    years = sorted({pd.Timestamp("2024-01-01").year, pd.Timestamp(args.end).year})
    if len(years) == 1:
        years = [years[0]]
    else:
        years = list(range(min(years), max(years) + 1))

    candles = load_candles_yearly(args.s3_candles_root, args.symbol, years=years)
    log(f"[candles] rows={len(candles):,} span={candles.index.min()} -> {candles.index.max()}")

    # 2) Load trades/book aggregated to t0 (streaming)
    months = list(month_range("2024-01-01", args.end))
    trades_t0 = load_trades_t0_streaming(args.s3_trades_root, args.symbol, months, args.decision_freq_min, args.batch_rows)
    book_t0   = load_book_t0_streaming(args.s3_book_root, args.symbol, months, args.decision_freq_min, args.batch_rows)
    log(f"[trades_t0] rows={len(trades_t0):,}")
    log(f"[book_t0]   rows={len(book_t0):,}")

    # 3) Build features/label on t0 right-edge
    c_feat = candle_features_at_t0(candles, args.decision_freq_min)
    lab = build_label_future_range(candles, args.decision_freq_min, args.horizon_min, args.train_end, args.label_q)
    t0_hilo = build_t0_hilo(candles, args.decision_freq_min)

    # Align everything on the t0 grid = c_feat.index (truth)
    t0_index = c_feat.index

    trades_t0 = trades_t0.reindex(t0_index)
    for c in ["tr_count", "tr_vol", "tr_signed_vol"]:
        if c in trades_t0.columns:
            trades_t0[c] = trades_t0[c].fillna(0.0)
    if "tr_aggr_buy_ratio" in trades_t0.columns:
        trades_t0["tr_aggr_buy_ratio"] = trades_t0["tr_aggr_buy_ratio"].fillna(0.5)

    book_t0 = book_t0.reindex(t0_index).sort_index().ffill()

    df = pd.concat([c_feat, trades_t0, book_t0, lab, t0_hilo], axis=1)

    # neutral VWAP fill
    if "tr_vwap" in df.columns and "close" in df.columns:
        df["tr_vwap"] = df["tr_vwap"].fillna(df["close"])

    # Must-have: label and its underlying range
    df = df.dropna(subset=["label", "range_1h_bps", "t0_close", "t0_high", "t0_low"])
    log(f"[df] rows={len(df):,} cols={df.shape[1]} label_rate={df['label'].mean():.4f}")

    # 4) Split with embargo (avoid overlap label)
    tr_end_ts = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    te_start_ts = pd.Timestamp(args.test_start, tz="UTC")
    embargo = pd.Timedelta(minutes=int(args.horizon_min))

    train = df.loc[:tr_end_ts - embargo].copy()
    test  = df.loc[te_start_ts + embargo:].copy()

    log(f"[split+embargo] train_rows={len(train):,} test_rows={len(test):,} embargo={embargo}")

    drop_cols = {"label", "range_1h_bps", "label_thr_bps_train", "t0_high", "t0_low", "t0_close"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if len(train) < 1000 or len(test) < 1000:
        raise RuntimeError(f"Not enough data after split. train={len(train)} test={len(test)}")

    # ---- select export scope
    if args.events_scope == "test":
        df_events_src = test
    elif args.events_scope == "all":
        df_events_src = df
    else:  # oos
        df_events_src = df.loc[pd.Timestamp(args.test_start, tz="UTC") + embargo:].copy()

    # ---- train
    log("[fit] training LightGBM...")
    m = fit_lgbm(train[feature_cols], train["label"], seed=args.seed)

    # ---- build export events
    ev_export = breakout_events(df_events_src[["t0_high", "t0_low", "t0_close"]], lookback_bars=int(args.lookback_bars))
    df_ev_export = df_events_src.loc[ev_export["event"].to_numpy()].copy()
    ev_dir_export = ev_export.loc[df_ev_export.index, "dir"]

    log(f"[events] breakout events ({args.events_scope}): n={len(df_ev_export):,} (lookback_bars={args.lookback_bars})")
    if len(df_ev_export) == 0:
        raise RuntimeError("No events found. Check lookback-bars/frequency.")

    p_export = predict_proba(m, df_ev_export[feature_cols])

    # ---- Write events parquet
    events_out = build_events_df(df_ev_export, ev_dir_export, p_export)
    log(f"[events] writing: {args.events_out} rows={len(events_out):,}")
    write_parquet_any(events_out, args.events_out)

    # ---- optional report ALWAYS on test events (recommended)
    if args.report:
        ev_test = breakout_events(test[["t0_high", "t0_low", "t0_close"]], lookback_bars=int(args.lookback_bars))
        df_ev_test = test.loc[ev_test["event"].to_numpy()].copy()

        if len(df_ev_test) < int(args.min_events_test):
            log(f"[report] ❌ Too few TEST events: n={len(df_ev_test)} < {args.min_events_test}")
        else:
            p_test = predict_proba(m, df_ev_test[feature_cols])
            y = df_ev_test["label"].astype(int)

            auc = float(roc_auc_score(y, p_test))
            lift = float(lift_at_topk(y, p_test, top_frac=float(args.top_frac)))
            base = float(y.mean())

            auc_vol = np.nan
            lift_vol = np.nan
            if "c_range_1h_bps" in df_ev_test.columns:
                vol_score = df_ev_test["c_range_1h_bps"].astype(float).to_numpy()
                auc_vol = float(roc_auc_score(y, vol_score))
                lift_vol = float(lift_at_topk(y, vol_score, top_frac=float(args.top_frac)))

            log("\n============================")
            log("REPORT — TEST METRICS (EVENTS ONLY)")
            log("============================")
            log(f"events_scope(export)  : {args.events_scope}")
            log(f"events(test)          : {len(df_ev_test):,}")
            log(f"base rate (events)    : {base:.4f}")
            log(f"AUC ML (events)       : {auc:.4f}")
            log(f"Lift@Top{int(args.top_frac*100)}% ML   : {lift:.3f}")
            if np.isfinite(auc_vol):
                log(f"AUC VOL (events)      : {auc_vol:.4f}")
                log(f"Lift@Top{int(args.top_frac*100)}% VOL  : {lift_vol:.3f}")
            log("============================")

    log("✅ Done.")

if __name__ == "__main__":
    main()