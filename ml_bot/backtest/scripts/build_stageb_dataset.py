#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ml_bot/backtest/scripts/build_stageb_dataset.py

from __future__ import annotations

import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import fsspec

from ml_bot.backtest.lib.utils import months_between
from ml_bot.backtest.lib.io_s3 import read_parquet_s3, write_parquet_s3
from ml_bot.backtest.lib.s3_paths import book_month_path, candles_year_path
from ml_bot.backtest.lib.windows import build_windows
from ml_bot.backtest.lib.book_windows import read_book_with_next_windows
from ml_bot.backtest.lib.pnl import pnl_at_T_bps, apply_exit_rule
from ml_bot.backtest.lib.atr_vol import compute_atr_bps_from_1m, attach_vol_bucket


# =========================================================
# CONFIG
# =========================================================

BO_NAME = "BO | rw>-15 | gate=T60>0 | 60_if_neg_else_180"
MR_NAME = "MR | rw>-15 | gate=T60>0 | 60_if_neg_else_180"

FEATURE_COLS_V1 = [
    # core
    "timestamp",
    "dir0",
    "regime",
    "MS",
    # spread / liquidity
    "spread_bps",
    "spread_ticks_1s",
    "spread_rel_5m",
    # micro / imbalance
    "micro_bias_bps",
    "OBI_1",
    "OBI_3",
    "OBI_5",
    "OBI_10",
    "OBI_14",
    # tape / flow
    "TI",
    "nps",
    "Ntot",
    "notional_buy",
    "notional_sell",
    "qty_buy",
    "qty_sell",
    "ntr",
    "Nb",
    "Ns",
    # persistence / thinning
    "persist_micro_ms",
    "persist_obi10_ms",
    "thinning_opp_3",
    "thinning_opp_10",
    # local range / vol
    "range_60s_bps",
    "range_10m_bps",
]

REQUIRED_EVALREADY = [
    "timestamp",
    "dir0",
    "pass_all_hard",
    "fees_rt_bps",
    "slip_bps",
    "MS",
]


# =========================================================
# HELPERS
# =========================================================

def split_from_month(month: str) -> str:
    if "2024-04" <= month <= "2024-12":
        return "train"
    if "2025-01" <= month <= "2025-04":
        return "val"
    if "2025-05" <= month <= "2025-10":
        return "test"
    return "other"


def router_branch_from_bucket(vol_bucket: str) -> str:
    return "MR" if vol_bucket == "b2" else "BO"


def load_tagged_month(tagged_root: str, month: str) -> pd.DataFrame:
    path = f"{tagged_root.rstrip('/')}/wf_stage0_tagged_{month}.parquet"
    df = read_parquet_s3(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    missing = [c for c in FEATURE_COLS_V1 if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing feature columns: {missing}")

    if "pass_all_hard" not in df.columns:
        raise ValueError(f"{path}: missing pass_all_hard")

    out = df.loc[df["pass_all_hard"] == True, FEATURE_COLS_V1].copy()

    # numeric coercion
    for c in out.columns:
        if c in ("timestamp", "regime"):
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["timestamp", "dir0", "MS"]).copy()
    return out


def load_evalready_month(evalready_root: str, month: str) -> pd.DataFrame:
    path = f"{evalready_root.rstrip('/')}/wf_stage0_evalready_{month}.parquet"
    df = read_parquet_s3(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    missing = [c for c in REQUIRED_EVALREADY if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")

    df = df.loc[df["pass_all_hard"] == True, REQUIRED_EVALREADY].copy()

    for c in ["dir0", "fees_rt_bps", "slip_bps", "MS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "dir0", "fees_rt_bps", "slip_bps", "MS"]).copy()
    return df


def load_candles_years(candles_root: str, symbol: str, years: List[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = candles_year_path(candles_root, symbol, int(y))
        df = read_parquet_s3(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    return out


def evaluate_branch_labels(
    events_m: pd.DataFrame,
    book_m: pd.DataFrame,
    router_branch_col: str,
    runaway_T: int,
    tol_s: float,
    runaway_min_bps: float = -15.0,
    gate: str = "T60>0",
    exit_rule: str = "exit:60_if_neg_else_180",
) -> pd.DataFrame:
    """
    Build baseline labels on the chosen router branch.
    events_m must contain:
      timestamp, dir0, fees_rt_bps, slip_bps, router_branch
    """

    df = events_m.copy()
    d0 = df["dir0"].astype("int64")
    dir_use = np.where(df[router_branch_col].values == "BO", d0.values, (-d0).values)
    dir_use = pd.Series(dir_use, index=df.index)

    pnl30  = pnl_at_T_bps(df, book_m, dir_use, runaway_T, tol_s)
    pnl45  = pnl_at_T_bps(df, book_m, dir_use, 45, tol_s)
    pnl60  = pnl_at_T_bps(df, book_m, dir_use, 60, tol_s)
    pnl120 = pnl_at_T_bps(df, book_m, dir_use, 120, tol_s)
    pnl180 = pnl_at_T_bps(df, book_m, dir_use, 180, tol_s)

    # strict notebook-compatible matching
    mask = (
        pnl30.notna()
        & pnl45.notna()
        & pnl60.notna()
        & pnl120.notna()
        & pnl180.notna()
    )

    # runaway
    mask &= (pnl30 > float(runaway_min_bps))

    # gate
    if gate == "T45>0":
        mask &= (pnl45 > 0)
    elif gate == "T60>0":
        mask &= (pnl60 > 0)
    elif gate != "none":
        raise ValueError(f"Unknown gate: {gate}")

    pnl_exit = apply_exit_rule(
        pnl45.values,
        pnl60.values,
        pnl120.values,
        pnl180.values,
        exit_rule,
    )
    pnl_exit = pd.Series(pnl_exit, index=df.index).where(mask)

    out = df[["timestamp"]].copy()
    out["pnl_net_bps"] = pnl_exit
    out["y_go"] = (out["pnl_net_bps"] > 0).astype("Int8")
    out["y_strong"] = (out["pnl_net_bps"] > 4.0).astype("Int8")
    out["is_tradeable_baseline"] = out["pnl_net_bps"].notna().astype("Int8")
    return out


# =========================================================
# MAIN
# =========================================================

def build_dataset(
    tagged_root: str,
    evalready_root: str,
    book_root: str,
    candles_root: str,
    out_path: str,
    symbol: str,
    start: str,
    end: str,
    pad_pre_s: float,
    pad_post_s: float,
    tol_s: float,
    atr_tf: str,
    atr_n: int,
    n_vol_buckets: int,
    runaway_T: int,
) -> None:
    months = months_between(start, end)
    fs = fsspec.filesystem("s3")

    print("months:", months)

    # preload candles once for all years needed
    years = sorted({int(m[:4]) for m in months})
    candles_1m = load_candles_years(candles_root, symbol, years)
    atr_tf_df = compute_atr_bps_from_1m(candles_1m, atr_n=atr_n, tf=atr_tf)

    parts = []

    for month in months:
        print(f"---- {month} ----")

        tagged = load_tagged_month(tagged_root, month)
        evalr = load_evalready_month(evalready_root, month)

        # intersection on timestamp only, keep features from tagged and stable trade params from evalready
        df = tagged.merge(
            evalr[["timestamp", "dir0", "fees_rt_bps", "slip_bps", "MS"]],
            on="timestamp",
            how="inner",
            suffixes=("_tag", ""),
        )

        if len(df) == 0:
            print("no rows after tagged/evalready merge")
            continue

        # normalize dir0/MS after merge
        if "dir0_tag" in df.columns:
            df = df.drop(columns=["dir0_tag"])
        if "MS_tag" in df.columns:
            df = df.drop(columns=["MS_tag"])

        df["month"] = month
        df["split"] = split_from_month(month)
        df["symbol"] = symbol

        # attach ATR / vol bucket
        tmp_for_atr = df[["timestamp"]].copy()
        tmp_for_atr["pnl_net_bps"] = 0.0
        tmp_for_atr["config"] = BO_NAME  # dummy; attach_vol_bucket only needs timestamp + atr merge
        tmp_for_atr["day"] = df["timestamp"].dt.floor("D")

        tmp_atr = attach_vol_bucket(tmp_for_atr, atr_tf_df, n_buckets=n_vol_buckets)
        tmp_atr = tmp_atr[["timestamp", "atr_bps", "vol_bucket", "t_atr"]].copy()

        df = df.merge(tmp_atr, on="timestamp", how="left")

        # router fixed
        df["router_branch"] = df["vol_bucket"].map(router_branch_from_bucket)

        # build windows around current month candidate timestamps only
        win_s, win_e = build_windows(df["timestamp"], pre_s=pad_pre_s, post_s=pad_post_s)
        book_m = read_book_with_next_windows(book_root, symbol, month, win_s, win_e, fs=fs)
        print(f"rows={len(df)} book_rows={len(book_m)}")

        if len(book_m) == 0:
            print("empty book, skip month")
            continue

        # build baseline labels on fixed router branch
        labels = evaluate_branch_labels(
            df[["timestamp", "dir0", "fees_rt_bps", "slip_bps", "router_branch"]].copy(),
            book_m,
            router_branch_col="router_branch",
            runaway_T=runaway_T,
            tol_s=tol_s,
            runaway_min_bps=-15.0,
            gate="T60>0",
            exit_rule="exit:60_if_neg_else_180",
        )

        df = df.merge(labels, on="timestamp", how="left")

        # keep a compact, t0-safe dataset
        keep_cols = [
            "timestamp",
            "month",
            "split",
            "symbol",
            "router_branch",
            "vol_bucket",
            "atr_bps",
            "t_atr",
            # features
            "dir0",
            "regime",
            "MS",
            "spread_bps",
            "spread_ticks_1s",
            "spread_rel_5m",
            "micro_bias_bps",
            "OBI_1",
            "OBI_3",
            "OBI_5",
            "OBI_10",
            "OBI_14",
            "TI",
            "nps",
            "Ntot",
            "notional_buy",
            "notional_sell",
            "qty_buy",
            "qty_sell",
            "ntr",
            "Nb",
            "Ns",
            "persist_micro_ms",
            "persist_obi10_ms",
            "thinning_opp_3",
            "thinning_opp_10",
            "range_60s_bps",
            "range_10m_bps",
            # labels
            "pnl_net_bps",
            "y_go",
            "y_strong",
            "is_tradeable_baseline",
            # trade cost fields for audit
            "fees_rt_bps",
            "slip_bps",
        ]

        miss = [c for c in keep_cols if c not in df.columns]
        if miss:
            raise ValueError(f"{month}: missing final cols: {miss}")

        df = df[keep_cols].copy()
        parts.append(df)

    if not parts:
        raise SystemExit("No dataset rows built.")

    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    print("final dataset shape:", out.shape)
    print("split counts:")
    print(out["split"].value_counts(dropna=False))
    print("tradeable rate:", out["is_tradeable_baseline"].mean())
    print("y_go rate:", out["y_go"].dropna().mean())
    print("y_strong rate:", out["y_strong"].dropna().mean())

    write_parquet_s3(out, out_path)
    print(f"✅ dataset written: {out_path}")


def main():
    ap = argparse.ArgumentParser("Build StageB dataset without lookahead bias")
    ap.add_argument("--tagged-root", default="s3://tradebot-config-tokyo/data/v1")
    ap.add_argument("--evalready-root", default="s3://tradebot-config-tokyo/data/v1_evalready")
    ap.add_argument("--book-root", default="s3://tradebot-config-tokyo/data/book")
    ap.add_argument("--candles-root", default="s3://tradebot-config-tokyo/data/bougie")
    ap.add_argument("--out-path", default="s3://tradebot-config-tokyo/research/ms_edge/ml/stageb_dataset.parquet")

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2024-04")
    ap.add_argument("--end", default="2025-10")

    ap.add_argument("--pad-pre-s", type=float, default=2.0)
    ap.add_argument("--pad-post-s", type=float, default=185.0)
    ap.add_argument("--tol-s", type=float, default=0.25)

    ap.add_argument("--atr-tf", default="15min")
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--n-vol-buckets", type=int, default=5)

    ap.add_argument("--runaway-T", type=int, default=30)

    args = ap.parse_args()

    build_dataset(
        tagged_root=args.tagged_root,
        evalready_root=args.evalready_root,
        book_root=args.book_root,
        candles_root=args.candles_root,
        out_path=args.out_path,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        pad_pre_s=args.pad_pre_s,
        pad_post_s=args.pad_post_s,
        tol_s=args.tol_s,
        atr_tf=args.atr_tf,
        atr_n=args.atr_n,
        n_vol_buckets=args.n_vol_buckets,
        runaway_T=args.runaway_T,
    )


if __name__ == "__main__":
    main()