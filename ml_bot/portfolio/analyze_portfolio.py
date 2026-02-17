#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_bot/portfolio/analyze_portfolio.py

Reads portfolio outputs:
- trades:  s3://.../portfolio_trades.parquet
- equity:  s3://.../portfolio_equity.parquet  (optional)
- monthly: s3://.../portfolio_monthly.parquet (optional)

Prints:
1) Per-symbol contribution table: n, sum_R_contrib, expectancy_R_contrib, win/flat rates
2) Capacity/overlap diagnostics: how many signals were dropped at same t0 (if available)
3) Monthly stats for R_contrib

Assumptions:
- trades parquet has at least: t0, symbol, R (or R_trade), R_contrib
- If you saved "taken" / "dropped" flags or "reason", we'll use them; otherwise we infer overlap by grouping on t0.
"""

import argparse
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs


def read_parquet_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_input_file(s3_path) as f:
            return pq.read_table(f).to_pandas()
    return pq.read_table(path).to_pandas()


def stats_R(x: np.ndarray) -> dict:
    x = x.astype(float)
    n = len(x)
    if n == 0:
        return {"n": 0}
    win = float(np.mean(x > 0))
    loss = float(np.mean(x < 0))
    flat = float(np.mean(x == 0))
    exp = float(np.nanmean(x))
    s = float(np.nansum(x))
    gp = float(np.nansum(x[x > 0]))
    gl = float(-np.nansum(x[x < 0]))
    pf = (gp / gl) if gl > 0 else np.nan
    eq = np.nancumsum(x)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([])
    dd = eq - peak if len(eq) else np.array([])
    mdd = float(np.nanmin(dd)) if len(dd) else np.nan
    return {
        "n": int(n),
        "expectancy_R": exp,
        "sum_R": s,
        "win_rate": win,
        "loss_rate": loss,
        "flat_rate": flat,
        "profit_factor": pf,
        "max_drawdown_R": mdd,
    }


def monthly_stats(df: pd.DataFrame, col: str, t_col: str = "t0") -> pd.DataFrame:
    tmp = df.copy()
    tmp[t_col] = pd.to_datetime(tmp[t_col], utc=True)
    tmp["ym"] = tmp[t_col].dt.to_period("M").astype(str)
    out = (
        tmp.groupby("ym", sort=True)[col]
        .apply(lambda s: pd.Series(stats_R(s.to_numpy())))
        .reset_index()
    )
    return out


def main():
    ap = argparse.ArgumentParser("Analyze portfolio backtest outputs")
    ap.add_argument("--trades", required=True, help="s3://.../portfolio_trades.parquet")
    ap.add_argument("--equity", default="", help="optional s3://.../portfolio_equity.parquet")
    ap.add_argument("--monthly", default="", help="optional s3://.../portfolio_monthly.parquet")
    ap.add_argument("--t-col", default="t0")
    ap.add_argument("--symbol-col", default="symbol")
    ap.add_argument("--rcontrib-col", default="R_contrib")
    ap.add_argument("--rtrade-col", default="R")  # fallback if needed
    args = ap.parse_args()

    trades = read_parquet_any(args.trades)
    if args.t_col not in trades.columns:
        raise RuntimeError(f"trades missing time column: {args.t_col}")
    if args.symbol_col not in trades.columns:
        raise RuntimeError(f"trades missing symbol column: {args.symbol_col}")

    # Choose R_contrib (preferred), else compute from alloc * R if columns exist
    if args.rcontrib_col in trades.columns:
        rcol = args.rcontrib_col
    else:
        # try to build it
        if "risk_alloc" in trades.columns and args.rtrade_col in trades.columns:
            trades[args.rcontrib_col] = trades["risk_alloc"].astype(float) * trades[args.rtrade_col].astype(float)
            rcol = args.rcontrib_col
        else:
            raise RuntimeError(f"trades missing {args.rcontrib_col} and cannot infer it")

    # Clean types
    trades[args.t_col] = pd.to_datetime(trades[args.t_col], utc=True)
    trades = trades.sort_values(args.t_col)

    # Drop NaN contrib rows (diagnostic)
    n0 = len(trades)
    trades_ok = trades[np.isfinite(trades[rcol].astype(float))].copy()
    dropped_nan = n0 - len(trades_ok)

    print("===================================")
    print("PORTFOLIO ANALYSIS")
    print("===================================")
    print(f"trades rows: {n0}")
    print(f"dropped due to NaN {rcol}: {dropped_nan}")

    # 1) Per-symbol table
    def sym_agg(g: pd.DataFrame) -> pd.Series:
        x = g[rcol].to_numpy(dtype=float)
        st = stats_R(x)
        return pd.Series({
            "n": st["n"],
            "sum_R_contrib": st["sum_R"],
            "expectancy_R_contrib": st["expectancy_R"],
            "win_rate": st["win_rate"],
            "flat_rate": st["flat_rate"],
        })

    per_symbol = (
        trades_ok.groupby(args.symbol_col, sort=False)
        .apply(sym_agg)
        .reset_index()
        .sort_values(["sum_R_contrib", "n"], ascending=[False, False])
        .reset_index(drop=True)
    )

    print("\n=== PER SYMBOL (sorted by sum_R_contrib) ===")
    print(per_symbol.to_string(index=False))

    # 2) Capacity / overlap diagnostics (inference)
    # If you didn't store dropped signals, we can still estimate concurrency pressure:
    # how many trades share the same t0 (if your selection engine emits same-minute signals)
    same_t0 = trades_ok.groupby(args.t_col).size()
    overlaps = same_t0[same_t0 > 1]
    print("\n=== OVERLAP / CONCURRENCY (inferred from same t0) ===")
    print(f"unique t0: {same_t0.size}")
    print(f"t0 with >=2 trades: {overlaps.size}")
    if overlaps.size:
        print(f"max trades at same t0: {int(overlaps.max())}")
        print(f"mean trades per overlapped t0: {float(overlaps.mean()):.3f}")

        # show top 10 crowded timestamps
        top = overlaps.sort_values(ascending=False).head(10)
        print("\nTop crowded t0 (count):")
        print(top.to_string())

    # 3) Monthly stats
    print("\n=== MONTHLY (R_contrib) ===")
    m = monthly_stats(trades_ok, col=rcol, t_col=args.t_col)
    print(m.to_string(index=False))

    # Optional: compare with saved monthly/equity files if provided
    if args.monthly:
        saved_m = read_parquet_any(args.monthly)
        print("\n=== SAVED MONTHLY FILE (head) ===")
        print(saved_m.head(10).to_string(index=False))

    if args.equity:
        eq = read_parquet_any(args.equity)
        print("\n=== EQUITY FILE (head) ===")
        print(eq.head(10).to_string(index=False))


if __name__ == "__main__":
    main()