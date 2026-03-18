#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ml_bot/backtest/scripts/build_stage0_thresholds_prod.py

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import time

import pandas as pd

from ml_bot.backtest.lib.io_s3 import write_json_s3
from ml_bot.core.stage0.spr_v1 import (
    Stage0SPRConfig,
    fit_thresholds,
)
from ml_bot.cli.run_stage0_spr_walkforward import (
    ym_range_end_exclusive,
    mk_paths,
    sample_train_month,
    fit_regime_from_train,
)


def main():
    ap = argparse.ArgumentParser("Build Stage0 production thresholds from trailing train window")

    ap.add_argument("--reference-month", required=True, help="Production effective month YYYY-MM")
    ap.add_argument("--train-window-months", type=int, default=3, help="Trailing train months")
    ap.add_argument("--train-sample-n", type=int, default=2_000_000, help="Rows sampled per train month")
    ap.add_argument("--train-sample-seed", type=int, default=42)

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--root-book", default="s3://tradebot-config-tokyo/data/book")
    ap.add_argument("--root-trades", default="s3://tradebot-config-tokyo/data/trade")

    ap.add_argument(
        "--out-thresholds",
        default="s3://tradebot-config-tokyo/research/ms_edge/stage0_thresholds/stage0_thresholds_prod.json",
    )
    ap.add_argument(
        "--out-config",
        default="s3://tradebot-config-tokyo/research/ms_edge/stage0_thresholds/stage0_cfg_prod.json",
    )
    ap.add_argument(
        "--out-meta",
        default="s3://tradebot-config-tokyo/research/ms_edge/stage0_thresholds/stage0_thresholds_prod_meta.json",
    )

    ap.add_argument("--batch-rows", type=int, default=300000)
    ap.add_argument("--freq-ms", type=int, default=250)
    ap.add_argument("--max-level", type=int, default=15)

    # cfg core
    ap.add_argument("--max-horizon-s", type=float, default=180.0)
    ap.add_argument("--trades-window-s", type=float, default=2.0)
    ap.add_argument("--persist-window-ms", type=int, default=800)
    ap.add_argument("--thinning-window-s", type=float, default=2.0)

    ap.add_argument("--profit-net-target-bps", type=float, default=6.0)
    ap.add_argument("--fee-maker-bps", type=float, default=2.0)
    ap.add_argument("--fee-taker-bps", type=float, default=6.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)

    # thresholds percentiles
    ap.add_argument("--p-spread", type=float, default=60.0)
    ap.add_argument("--p-micro", type=float, default=80.0)
    ap.add_argument("--p-obi10", type=float, default=97.0)
    ap.add_argument("--p-ti", type=float, default=85.0)
    ap.add_argument("--p-nps", type=float, default=60.0)
    ap.add_argument("--p-thin", type=float, default=85.0)
    ap.add_argument("--p-range60s", type=float, default=90.0)

    ap.add_argument("--persist-ms-min", type=int, default=250)
    ap.add_argument("--ms-keep-quantile", type=float, default=97.0)

    ap.add_argument("--depths", type=str, default="1,3,5,10,14")
    ap.add_argument("--range-win-60s", type=int, default=60)
    ap.add_argument("--range-win-10m", type=int, default=600)

    # regime info, useful for metadata / future reuse
    ap.add_argument("--regime-src-col", type=str, default="range_10m_bps")
    ap.add_argument("--regime-q-lo", type=float, default=0.33)
    ap.add_argument("--regime-q-hi", type=float, default=0.66)

    # spread / liquidity params
    ap.add_argument("--tick-size", type=float, default=0.1)
    ap.add_argument("--spread-roll-med-s", type=int, default=300)
    ap.add_argument("--p-spread-rel", type=float, default=80.0)
    ap.add_argument("--p-spread-ticks", type=float, default=80.0)

    args = ap.parse_args()
    cfg = Stage0SPRConfig.from_args(args)

    train_months = ym_range_end_exclusive(args.reference_month, int(args.train_window_months))
    if len(train_months) != int(args.train_window_months):
        raise SystemExit(f"[prod_thr] invalid train window: {args.train_window_months}")

    print(f"[prod_thr] symbol={args.symbol} reference_month={args.reference_month}", flush=True)
    print(f"[prod_thr] train_months={train_months}", flush=True)

    t0 = time.time()
    train_samples = []

    for i, ym in enumerate(train_months):
        book_p, trades_p = mk_paths(args.root_book, args.root_trades, args.symbol, ym)
        seed_i = int(args.train_sample_seed) + (i + 1) * 10_000

        print(f"[prod_thr][train] month={ym} book={book_p}", flush=True)

        df_s = sample_train_month(
            book_p,
            trades_p,
            cfg,
            args,
            regime_src_col=args.regime_src_col,
            sample_n=int(args.train_sample_n),
            seed=seed_i,
        )

        if df_s.empty:
            raise SystemExit(f"[prod_thr][train] empty feats for month={ym}")

        print(f"[prod_thr][train] month={ym} sample_rows={len(df_s):,}", flush=True)
        train_samples.append(df_s)
        gc.collect()

    train_df = pd.concat(train_samples, axis=0, ignore_index=True)
    del train_samples
    gc.collect()

    print(f"[prod_thr] total_sample_rows={len(train_df):,} elapsed={time.time()-t0:.1f}s", flush=True)

    thr = fit_thresholds(train_df, cfg)

    # optional but useful: save regime meta too for future production reuse
    reg_meta = fit_regime_from_train(
        train_df,
        src_col=args.regime_src_col,
        q_lo=args.regime_q_lo,
        q_hi=args.regime_q_hi,
        min_rows=50_000,
    )

    thresholds_payload = asdict(thr)
    cfg_payload = asdict(cfg)

    meta_payload = {
        "symbol": args.symbol,
        "reference_month": args.reference_month,
        "train_window_months": int(args.train_window_months),
        "train_months": train_months,
        "train_sample_n": int(args.train_sample_n),
        "train_sample_seed": int(args.train_sample_seed),
        "n_train_rows": int(len(train_df)),
        "regime_meta": reg_meta,
        "paths": {
            "root_book": args.root_book,
            "root_trades": args.root_trades,
            "out_thresholds": args.out_thresholds,
            "out_config": args.out_config,
            "out_meta": args.out_meta,
        },
    }

    write_json_s3(thresholds_payload, args.out_thresholds)
    print(f"[prod_thr] wrote thresholds: {args.out_thresholds}", flush=True)

    write_json_s3(cfg_payload, args.out_config)
    print(f"[prod_thr] wrote cfg: {args.out_config}", flush=True)

    write_json_s3(meta_payload, args.out_meta)
    print(f"[prod_thr] wrote meta: {args.out_meta}", flush=True)

    print("[prod_thr] done", flush=True)


if __name__ == "__main__":
    main()