#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ml_bot/cli/run_stage0_spr_walkforward.py
#
# WALK-FORWARD SPR v1
#   TRAIN window: months (N-k ... N-1) sampled for OOM safety
#   TEST: month N full
#
# Output (Option B):
#   One parquet with ALL test rows + candidate flags
#   Labels only for rows passing 

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from ml_bot.core.stage0.spr_stream_250ms import build_stage0_features_streaming_250ms

from ml_bot.core.stage0.spr_v1 import (
    Stage0SPRConfig,
    compute_book_core_features,
    attach_spread_liquidity_features,
    attach_range_features,
    attach_persistence_features,
    attach_thinning_features,
    fit_thresholds,
    label_resolution_endogenous,
    tag_candidates,
)


# -----------------------
# Month helpers
# -----------------------

def ym_add(ym: str, delta_months: int) -> str:
    """'2024-01' + 1 -> '2024-02', '2024-01' - 1 -> '2023-12'."""
    y, m = map(int, ym.split("-"))
    total = y * 12 + (m - 1) + int(delta_months)
    y2 = total // 12
    m2 = (total % 12) + 1
    return f"{y2:04d}-{m2:02d}"

def ym_range_end_exclusive(end_ym: str, k_months: int) -> list[str]:
    """
    Return the k months ending at end_ym (exclusive):
      end_ym='2024-04', k=3 => ['2024-01','2024-02','2024-03']
    """
    if k_months <= 0:
        return []
    return [ym_add(end_ym, -k_months + i) for i in range(k_months)]

def mk_paths(root_book: str, root_trades: str, symbol: str, ym: str) -> tuple[str, str]:
    book = f"{root_book.rstrip('/')}/{symbol}/{ym}.parquet"
    trades = f"{root_trades.rstrip('/')}/{symbol}/{ym}.parquet"
    return book, trades

# -----------------------
# Feature build (TEST full)
# -----------------------

def build_feats_month_full(book_path: str, trades_path: str, cfg: Stage0SPRConfig, args) -> pd.DataFrame:
    feats = build_stage0_features_streaming_250ms(
        book_path=book_path,
        trades_path=trades_path,
        cfg=cfg,
        freq_ms=args.freq_ms,
        batch_rows=args.batch_rows,
        max_level=args.max_level,
    )
    if feats is None or len(feats) == 0:
        return pd.DataFrame()

    feats = compute_book_core_features(feats, cfg.depths)

    feats = attach_spread_liquidity_features(
        feats,
        tick_size=cfg.tick_size,
        roll_med_s=cfg.spread_roll_med_s,
    )

    feats = attach_range_features(
        feats,
        freq_ms=args.freq_ms,
        win_60s=args.range_win_60s,
        win_10m=args.range_win_10m,
    )

    # dir0 required downstream
    feats["dir0"] = np.sign(feats["TI"].values + 0.5 * np.sign(feats["micro_bias_bps"].values)).astype("int8")

    feats = attach_persistence_features(feats, window_ms=cfg.persist_window_ms)
    feats = attach_thinning_features(feats, window_s=cfg.thinning_window_s)

    return feats


# -----------------------
# TRAIN sampling (OOM-safe)
# -----------------------

def train_need_cols(regime_src_col: str) -> list[str]:
    # minimal for fit_thresholds + regime quantiles
    cols = [
        "timestamp",
        "spread_ticks_1s",
        "spread_rel_5m",
        "range_60s_bps",
        "spread_bps",
        "micro_bias_bps",
        "OBI_10",
        "TI",
        "nps",
        "thinning_opp_3",
    ]
    if regime_src_col and regime_src_col not in cols:
        cols.append(regime_src_col)
    return cols

def sample_train_month(
    book_path: str,
    trades_path: str,
    cfg: Stage0SPRConfig,
    args,
    *,
    regime_src_col: str,
    sample_n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Build month feats, then keep only necessary columns and sample rows.
    Frees big frames ASAP.
    """
    df = build_feats_month_full(book_path, trades_path, cfg, args)
    if df.empty:
        return df

    need = train_need_cols(regime_src_col)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"[train] missing required cols in feats: {miss}")

    df = df[need].copy(deep=False)

    # sample
    n = len(df)
    if sample_n and n > int(sample_n):
        df = df.sample(n=int(sample_n), random_state=int(seed), replace=False)

    # cleanup
    df = df.reset_index(drop=True)
    return df


# -----------------------
# Regime from TRAIN quantiles (walk-forward)
# -----------------------

def fit_regime_from_train(
    train_df: pd.DataFrame,
    *,
    src_col: str,
    q_lo: float,
    q_hi: float,
    min_rows: int = 50_000,
) -> dict:
    if src_col not in train_df.columns:
        raise RuntimeError(f"[regime] src_col='{src_col}' missing in train sample")

    s = pd.to_numeric(train_df[src_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < int(min_rows):
        raise RuntimeError(f"[regime] not enough rows for src_col='{src_col}' in train sample: n={len(s)}")

    qlo_v = float(s.quantile(float(q_lo)))
    qhi_v = float(s.quantile(float(q_hi)))
    return {"src_col": src_col, "q_lo": float(q_lo), "q_hi": float(q_hi), "qlo_v": qlo_v, "qhi_v": qhi_v}

def attach_regime(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    src = meta["src_col"]
    qlo_v = float(meta["qlo_v"])
    qhi_v = float(meta["qhi_v"])

    x = pd.to_numeric(df[src], errors="coerce").replace([np.inf, -np.inf], np.nan)
    reg = np.where(x <= qlo_v, "CALM", np.where(x <= qhi_v, "NORMAL", "VOLATILE"))
    reg = pd.Series(reg, index=df.index).where(x.notna(), other=np.nan)

    out = df.copy(deep=False)
    out["regime"] = reg.astype("object")

    # persist meta as constant columns for debugging
    out["regime_src_col"] = str(meta["src_col"])
    out["regime_q_lo"] = float(meta["q_lo"])
    out["regime_q_hi"] = float(meta["q_hi"])
    out["regime_qlo_v"] = float(meta["qlo_v"])
    out["regime_qhi_v"] = float(meta["qhi_v"])
    return out

# -----------------------
# Reporting
# -----------------------

def report_summary(tag: str, out_df: pd.DataFrame) -> None:
    n = len(out_df)
    n_hard = int(out_df["pass_all_hard"].sum()) if n else 0
    n_ms = int(out_df["pass_ms_keep"].sum()) if n else 0

    print(f"\n=== WALKFORWARD {tag} ===", flush=True)
    print(f"rows_total:        {n:,}", flush=True)
    print(f"pass_all_hard:     {n_hard:,}", flush=True)
    print(f"pass_ms_keep:      {n_ms:,}", flush=True)

    if "label" in out_df.columns:
        m = out_df["pnl_net_max_bps"].notna()
        print(f"labeled_rows:      {int(m.sum()):,}", flush=True)
        if int(m.sum()) > 0:
            hr = float((out_df.loc[m, "label"] == 1).mean())
            print(f"hit_rate(on labeled): {hr:.6f}", flush=True)

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser("run stage0 SPR v1 WALK-FORWARD — TRAIN window sampled, TEST full, Option B flags")

    ap.add_argument("--symbol", required=True)
    ap.add_argument("--month", required=True, help="TEST month YYYY-MM")
    ap.add_argument("--train-window-months", type=int, default=3, help="TRAIN months = N-k..N-1")
    ap.add_argument("--train-sample-n", type=int, default=2_000_000, help="rows sampled PER TRAIN month (OOM safety)")
    ap.add_argument("--train-sample-seed", type=int, default=42)

    ap.add_argument("--root-book", default="s3://tradebot-config-tokyo/data/book")
    ap.add_argument("--root-trades", default="s3://tradebot-config-tokyo/data/trade")
    ap.add_argument("--out", required=True, help="output parquet path (local or s3)")

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

    # thresholds percentiles (fit on train)
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

    # regime (train quantiles)
    ap.add_argument("--regime-src-col", type=str, default="range_10m_bps")
    ap.add_argument("--regime-q-lo", type=float, default=0.33)
    ap.add_argument("--regime-q-hi", type=float, default=0.66)

    # t2 endogenous params (applied on test)
    ap.add_argument("--p-spread-exp", type=float, default=99.0)
    ap.add_argument("--cont-win", type=int, default=3)

    # spread/liquidity params
    ap.add_argument("--tick-size", type=float, default=0.1)
    ap.add_argument("--spread-roll-med-s", type=int, default=300)
    ap.add_argument("--p-spread-rel", type=float, default=80.0)
    ap.add_argument("--p-spread-ticks", type=float, default=80.0)

    args = ap.parse_args()
    cfg = Stage0SPRConfig.from_args(args)

    train_months = ym_range_end_exclusive(args.month, int(args.train_window_months))
    if len(train_months) != int(args.train_window_months):
        raise SystemExit(f"[wf] invalid train window: {args.train_window_months}")

    print(f"[wf] symbol={args.symbol} test={args.month} train_window={train_months}", flush=True)
    print(f"[wf] thr_exec_bps={cfg.thr_exec_bps:.3f}", flush=True)

    # -------------------------
    # TRAIN: build sampled frames month-by-month (OOM safe)
    # -------------------------
    t0 = time.time()
    train_samples = []
    for i, ym in enumerate(train_months):
        book_p, trades_p = mk_paths(args.root_book, args.root_trades, args.symbol, ym)
        seed_i = int(args.train_sample_seed) + (i + 1) * 10_000

        print(f"[train] month={ym} book={book_p}", flush=True)
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
            raise SystemExit(f"[train] empty feats for month={ym}")

        print(f"[train] month={ym} sample_rows={len(df_s):,}", flush=True)
        train_samples.append(df_s)

        # free
        gc.collect()

    train_df = pd.concat(train_samples, axis=0, ignore_index=True)
    del train_samples
    gc.collect()

    print(f"[train] total_sample_rows={len(train_df):,} elapsed={time.time()-t0:.1f}s", flush=True)

    # fit thresholds on TRAIN sample
    thr = fit_thresholds(train_df, cfg)

    # fit regime quantiles on TRAIN sample
    reg_meta = fit_regime_from_train(
        train_df,
        src_col=args.regime_src_col,
        q_lo=args.regime_q_lo,
        q_hi=args.regime_q_hi,
        min_rows=50_000,
    )

    # done with train_df
    del train_df
    gc.collect()

    print(
        f"[train] regime src={reg_meta['src_col']} "
        f"q_lo={reg_meta['q_lo']}({reg_meta['qlo_v']:.4f}) "
        f"q_hi={reg_meta['q_hi']}({reg_meta['qhi_v']:.4f})",
        flush=True,
    )

    # -------------------------
    # TEST: full month
    # -------------------------
    t1 = time.time()
    test_book, test_trades = mk_paths(args.root_book, args.root_trades, args.symbol, args.month)
    print(f"[test] month={args.month} book={test_book}", flush=True)

    test_feats = build_feats_month_full(test_book, test_trades, cfg, args)
    if test_feats.empty:
        raise SystemExit(f"[test] empty feats for month={args.month}")

    # stable id for merges
    test_feats = test_feats.reset_index(drop=True)
    test_feats["row_id"] = np.arange(len(test_feats), dtype=np.int64)

    # attach regime + meta constants
    test_feats = attach_regime(test_feats, reg_meta)

    # persist train-learned range60 threshold
    test_feats["thr_range60s_min_train"] = float(getattr(thr, "range60s_min"))

    print(f"[test] rows={len(test_feats):,} elapsed={time.time()-t1:.1f}s", flush=True)

    # -------------------------
    # Option B: tag candidates (flags) on ALL rows
    # -------------------------
    out_df = tag_candidates(
        test_feats,
        cfg,
        thr,
        compute_ms_score=True,
    )

    # hard fail immédiat si ce n’est pas la bonne version chargée
    assert "n_total" in out_df.columns, f"Missing n_total. cols_head={out_df.columns[:30].tolist()}"

    print("[funnel] n_total =", int(out_df["n_total"].iloc[0]))

    # imprime le funnel dans l’ordre du cascade (masks OrderedDict)
    funnel_cols = [c for c in out_df.columns if c.startswith("n_up_to_")]
    for c in funnel_cols:
        print("[funnel]", c, "=", int(out_df[c].iloc[0]))

    print("[funnel] n_pass_all_hard =", int(out_df["n_pass_all_hard"].iloc[0]))
    print("[funnel] n_pass_ms_keep  =", int(out_df["pass_ms_keep"].sum()))

    # -------------------------
    # Label candidates (StageB dataset) — choose label_mask
    # -------------------------
    # Choose label mask with safety cap
    MASK_ORDER = ["pass_all_hard", "pass_ms_keep", "pass_up_to_sign_ti"]
    cap = 50_000

    label_mask = None
    for name in MASK_ORDER:
        m = out_df[name].astype(bool).to_numpy(copy=False)
        if int(m.sum()) > 0 and int(m.sum()) <= cap:
            label_mask = m
            print("[label] using mask:", name, "n=", int(m.sum()), flush=True)
            break

    if label_mask is None:
        # fallback: still use sign_ti but sample it
        m = out_df["pass_up_to_sign_ti"].astype(bool).to_numpy(copy=False)
        idx = np.where(m)[0]
        if len(idx) > cap:
            idx = np.random.RandomState(42).choice(idx, size=cap, replace=False)
        label_mask = np.zeros(len(out_df), dtype=bool)
        label_mask[idx] = True
        print("[label] using sampled pass_up_to_sign_ti:", int(label_mask.sum()), flush=True)
        
    cands = out_df.loc[label_mask].copy()

    print("[label] label_candidates =", int(label_mask.sum()), " / ", len(out_df), flush=True)

    if len(cands):
        labeled = label_resolution_endogenous(
            cands,
            test_feats,
            cfg,
            p_spread_exp=args.p_spread_exp,
            cont_win=args.cont_win,
        )

        label_cols = [
            "row_id",
            "label",
            "reason_res",
            "reason_hit",
            "t_res",
            "t_hit",
            "time_to_res_s",
            "time_to_hit_s",
            "pnl_raw_at_res_bps",
            "pnl_net_at_res_bps",
            "pnl_net_at_hit_bps",
            "pnl_raw_max_bps",
            "pnl_net_max_bps",
            "pnl_raw_min_bps",
            "pnl_net_min_bps",
            "target_net_bps",
            "fees_rt_bps",
            "slip_bps",
            "spread_exp_thr_bps",
        ]
        label_cols = [c for c in label_cols if c in labeled.columns]

        out_df = out_df.merge(labeled[label_cols], on="row_id", how="left")

        # diagnostics AFTER merge
        n_pnl = int(out_df.loc[label_mask, "pnl_net_max_bps"].notna().sum())
        n_hit = int((out_df.loc[label_mask, "label"] == 1).sum())
        print("[label] labeled pnl rows on label_mask =", n_pnl, "/", int(label_mask.sum()), flush=True)
        print("[label] hit-rate on label_mask =", (n_hit / max(1, int(label_mask.sum()))), flush=True)

    else:
        # ensure label cols exist
        for c in [
            "label","reason_res","reason_hit","t_res","t_hit",
            "time_to_res_s","time_to_hit_s",
            "pnl_raw_at_res_bps","pnl_net_at_res_bps","pnl_net_at_hit_bps",
            "pnl_raw_max_bps","pnl_net_max_bps","pnl_raw_min_bps","pnl_net_min_bps",
            "target_net_bps","fees_rt_bps","slip_bps","spread_exp_thr_bps",
        ]:
            if c not in out_df.columns:
                out_df[c] = np.nan

    # annotate run meta
    out_df.attrs["walkforward_train_months"] = train_months
    out_df.attrs["walkforward_test_month"] = args.month
    out_df.attrs["cfg"] = asdict(cfg)

    report_summary(f"{train_months} -> {args.month}", out_df)

    out_df.to_parquet(args.out, index=False)
    print(f"[wf] wrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()