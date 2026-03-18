#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ml_bot/backtest/scripts/build_vol_bucket_edges_prod.py

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

from ml_bot.backtest.lib.io_s3 import write_json_s3
from ml_bot.backtest.lib.s3_paths import candles_year_path
from ml_bot.backtest.lib.io_s3 import read_parquet_s3
from ml_bot.backtest.lib.atr_vol import compute_atr_bps_from_1m

def ym_add(ym: str, delta: int) -> str:
    """
    Add delta months to YYYY-MM string.
    """
    y, m = map(int, ym.split("-"))
    m0 = m - 1 + delta
    y += m0 // 12
    m = m0 % 12 + 1
    return f"{y:04d}-{m:02d}"

def ym_range_end_exclusive(end_ym: str, k_months: int) -> list[str]:
    if k_months <= 0:
        return []
    return [ym_add(end_ym, -k_months + i) for i in range(k_months)]

def load_candles_years_for_months(
    candles_root: str,
    symbol: str,
    months: list[str],
) -> pd.DataFrame:
    years = sorted({int(m[:4]) for m in months})
    parts = []

    for y in years:
        p = candles_year_path(candles_root, symbol, int(y))
        df = read_parquet_s3(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return out

def compute_bucket_edges(
    atr_bps_series: pd.Series,
    n_buckets: int,
) -> list[float]:
    s = pd.to_numeric(atr_bps_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        raise SystemExit("[vol_edges] no valid atr_bps values")

    nb = int(n_buckets)
    if nb < 2:
        raise SystemExit("[vol_edges] n_buckets must be >= 2")

    qs = [i / nb for i in range(1, nb)]
    edges = [float(s.quantile(q)) for q in qs]

    # optional: enforce monotonic unique-ish edges
    clean = []
    prev = None
    for e in edges:
        if prev is not None and e < prev:
            e = prev
        clean.append(float(e))
        prev = float(e)

    return clean

def main():
    ap = argparse.ArgumentParser("Build frozen ATR vol bucket edges for live production")

    ap.add_argument("--reference-month", required=True, help="Production effective month YYYY-MM")
    ap.add_argument("--train-window-months", type=int, default=3, help="Trailing months used to calibrate edges")

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--candles-root", default="s3://tradebot-config-tokyo/data/bougie")

    ap.add_argument("--atr-tf", default="15min")
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--n-vol-buckets", type=int, default=5)

    ap.add_argument(
        "--out-path",
        default="s3://tradebot-config-tokyo/research/ms_edge/stage0_thresholds/vol_bucket_edges_prod.json",
    )
    ap.add_argument(
        "--out-meta",
        default="s3://tradebot-config-tokyo/research/ms_edge/stage0_thresholds/vol_bucket_edges_prod_meta.json",
    )

    args = ap.parse_args()

    train_months = ym_range_end_exclusive(args.reference_month, int(args.train_window_months))
    if len(train_months) != int(args.train_window_months):
        raise SystemExit(f"[vol_edges] invalid train window: {args.train_window_months}")

    print(f"[vol_edges] symbol={args.symbol} reference_month={args.reference_month}", flush=True)
    print(f"[vol_edges] train_months={train_months}", flush=True)

    candles_1m = load_candles_years_for_months(
        candles_root=args.candles_root,
        symbol=args.symbol,
        months=train_months,
    )

    start_ts = pd.Timestamp(f"{train_months[0]}-01", tz="UTC")
    end_month = args.reference_month
    end_ts = pd.Timestamp(f"{end_month}-01", tz="UTC")

    candles_1m = candles_1m.loc[
        (candles_1m["timestamp"] >= start_ts)
        & (candles_1m["timestamp"] < end_ts)
    ].copy()

    if candles_1m.empty:
        raise SystemExit("[vol_edges] empty candles after month filtering")

    atr_tf_df = compute_atr_bps_from_1m(
        candles_1m,
        atr_n=int(args.atr_n),
        tf=str(args.atr_tf),
    )

    if atr_tf_df.empty:
        raise SystemExit("[vol_edges] empty atr_tf dataframe")

    edges = compute_bucket_edges(
        atr_bps_series=atr_tf_df["atr_bps"],
        n_buckets=int(args.n_vol_buckets),
    )

    payload = {
        "symbol": args.symbol,
        "reference_month": args.reference_month,
        "train_window_months": int(args.train_window_months),
        "train_months": train_months,
        "atr_tf": args.atr_tf,
        "atr_n": int(args.atr_n),
        "n_vol_buckets": int(args.n_vol_buckets),
        "edges": edges,
    }

    meta = {
        **payload,
        "n_atr_rows": int(len(atr_tf_df)),
        "atr_bps_min": float(pd.to_numeric(atr_tf_df["atr_bps"], errors="coerce").min()),
        "atr_bps_p25": float(pd.to_numeric(atr_tf_df["atr_bps"], errors="coerce").quantile(0.25)),
        "atr_bps_p50": float(pd.to_numeric(atr_tf_df["atr_bps"], errors="coerce").quantile(0.50)),
        "atr_bps_p75": float(pd.to_numeric(atr_tf_df["atr_bps"], errors="coerce").quantile(0.75)),
        "atr_bps_max": float(pd.to_numeric(atr_tf_df["atr_bps"], errors="coerce").max()),
    }

    write_json_s3(payload, args.out_path)
    print(f"[vol_edges] wrote edges: {args.out_path}", flush=True)

    write_json_s3(meta, args.out_meta)
    print(f"[vol_edges] wrote meta: {args.out_meta}", flush=True)

    print("[vol_edges] done", flush=True)

if __name__ == "__main__":
    main()