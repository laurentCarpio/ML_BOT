#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ml_bot/backtest/scripts/ms_edge_backtest.py

from __future__ import annotations

import numpy as np
import pandas as pd
import fsspec

from ml_bot.backtest.lib.utils import months_between
from ml_bot.backtest.lib.io_s3 import read_parquet_s3, write_parquet_s3, write_json_s3, make_run_id
from ml_bot.backtest.lib.s3_paths import events_tagged_month_path, events_evalready_month_path
from ml_bot.backtest.lib.windows import build_windows
from ml_bot.backtest.lib.book_windows import read_book_with_next_windows
from ml_bot.backtest.lib.pnl import pnl_at_T_bps, apply_exit_rule
from ml_bot.backtest.lib.bootstrap import bootstrap_iid, bootstrap_block_by_day, bootstrap_block_delta_by_day
from ml_bot.backtest.lib.atr_vol import load_candles_years, compute_atr_bps_from_1m, attach_vol_bucket


def read_events_month(cfg: dict, month: str, mode: str) -> pd.DataFrame:
    if mode == "evalready":
        p = events_evalready_month_path(cfg["events_evalready_root"], month)
    else:
        p = events_tagged_month_path(cfg["events_tagged_root"], month)
    df = read_parquet_s3(p)
    # If tagged: filter hard here
    if "pass_all_hard" in df.columns:
        df = df.loc[df["pass_all_hard"] == True].copy()
    # enforce required cols
    need = ["timestamp", "dir0", "fees_rt_bps", "slip_bps", "MS"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{p}: missing {miss}")
    df = df[need].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["dir0"] = pd.to_numeric(df["dir0"], errors="coerce")
    df["fees_rt_bps"] = pd.to_numeric(df["fees_rt_bps"], errors="coerce")
    df["slip_bps"] = pd.to_numeric(df["slip_bps"], errors="coerce")
    df["MS"] = pd.to_numeric(df["MS"], errors="coerce")
    df = df.dropna(subset=["timestamp", "dir0", "fees_rt_bps", "slip_bps", "MS"])
    return df

def _needed_horizons(exit_rule: str, gate: str, runaway_T: int) -> list[int]:
    need = {runaway_T}  # ex 30s (runaway_T)
    # gate
    if gate == "T45>0":
        need.add(45)
    elif gate == "T60>0":
        need.add(60)
    elif gate != "none":
        raise ValueError(gate)

    # exit rule
    if exit_rule == "exit:60_if_neg_else_180":
        need.update([60, 180])
    elif exit_rule == "exit:60_if_neg_else_120":
        need.update([60, 120])
    elif exit_rule == "exit:45_if_neg_else_180":
        need.update([45, 180])
    else:
        raise ValueError(f"Unknown rule: {exit_rule}")

    return sorted(need)

def evaluate_config(events_m: pd.DataFrame, book_m: pd.DataFrame, *,
                    mode: str,
                    runaway_min_bps: float,
                    gate: str,
                    exit_rule: str,
                    runaway_T: int,
                    tol_s: float,
                    audit_print: bool = False,
                    audit_Ts: tuple[int, int] = (60, 180),
                   ) -> tuple[dict, pd.DataFrame]:

    d0 = events_m["dir0"].astype("int64")
    dir_use = d0 if mode == "BO" else (-d0)

    needed = _needed_horizons(exit_rule, gate, runaway_T)

    # Notebook compatibility: require all legs to be matchable (prevents silent drift)
    strict_Ts = {int(runaway_T), 45, 60, 120, 180}
    needed = sorted(set(map(int, needed)).union(strict_Ts))

    pnl = {}
    for T in needed:
        pnl[T] = pnl_at_T_bps(events_m, book_m, dir_use, T, tol_s)

    def _get_pnl(T: int) -> pd.Series:
        s = pnl.get(T, None)
        return s if s is not None else pnl_at_T_bps(events_m, book_m, dir_use, T, tol_s)

    pnl45  = _get_pnl(45)
    pnl60  = _get_pnl(60)
    pnl120 = _get_pnl(120)
    pnl180 = _get_pnl(180)

    # ---- AUDIT PRINT (optional) ----
    if audit_print:
        for TT in audit_Ts:
            _pnl, _aud = pnl_at_T_bps(events_m, book_m, dir_use, TT, tol_s, return_audit=True)
            print(f"AUDIT{TT}:", _aud)

    pnl_run = pnl[int(runaway_T)]

    # -------------------------
    # AUDIT: where does it drop?
    # -------------------------
    audit = {
        "n_events": int(len(events_m)),
        "need_Ts": ",".join(map(str, needed)),
    }
    for T in needed:
        audit[f"na_T{T}"] = int(pnl[T].isna().sum())
        audit[f"ok_T{T}"] = int(pnl[T].notna().sum())

    # -------------------------
    # Mask building (ONLY needed horizons)
    # -------------------------
    mask = pd.Series(True, index=events_m.index)

    # require matches for STRICT horizons (match notebook behavior)
    for T in sorted(strict_Ts):
        mask &= pnl[T].notna()
    audit["strict_Ts"] = ",".join(map(str, sorted(strict_Ts)))
    audit["n_after_strict_match"] = int(mask.sum())

    # runaway filter (uses pnl_run)
    if runaway_min_bps is not None:
        before = int(mask.sum())
        mask &= (pnl_run > float(runaway_min_bps))
        audit["n_after_runaway"] = int(mask.sum())
        audit["drop_runaway"] = before - int(mask.sum())
    else:
        audit["n_after_runaway"] = int(mask.sum())
        audit["drop_runaway"] = 0

    # gate
    before = int(mask.sum())
    if gate == "T45>0":
        mask &= (pnl45 > 0)
    elif gate == "T60>0":
        mask &= (pnl60 > 0)
    elif gate != "none":
        raise ValueError(gate)
    audit["n_after_gate"] = int(mask.sum())
    audit["drop_gate"] = before - int(mask.sum())

    # -------------------------
    # exit + stats
    # -------------------------
    if mask.sum() == 0:
        stats = {
            "n": 0, "EV_T_bps": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan,
            **audit
        }
        return stats, pd.DataFrame(columns=["timestamp", "pnl_net_bps"])

    pnl_exit = apply_exit_rule(pnl45.values, pnl60.values, pnl120.values, pnl180.values, exit_rule)
    pnl_exit = pd.Series(pnl_exit, index=events_m.index).where(mask)

    x = pnl_exit.dropna().values.astype("float64")
    stats = {
        "n": int(len(x)),
        "EV_T_bps": float(np.mean(x)) if len(x) else np.nan,
        "p05": float(np.quantile(x, 0.05)) if len(x) else np.nan,
        "p50": float(np.quantile(x, 0.50)) if len(x) else np.nan,
        "p95": float(np.quantile(x, 0.95)) if len(x) else np.nan,
        **audit
    }

    trades = pd.DataFrame({
        "timestamp": events_m["timestamp"],
        "pnl_net_bps": pnl_exit,
    }).dropna(subset=["pnl_net_bps"]).copy()

    return stats, trades

def router_pick(trades_vol: pd.DataFrame, bo_name: str, mr_name: str, n_buckets: int) -> pd.DataFrame:
    t = trades_vol.copy()
    if "mid_vol" in t["vol_bucket"].unique():
        is_mr = (t["vol_bucket"] == "mid_vol")
    else:
        mid_idx = n_buckets // 2   # 5->2, 7->3
        is_mr = (t["vol_bucket"] == f"b{mid_idx}")
    t["router_pick"] = np.where(is_mr, mr_name, bo_name)
    return t.loc[t["config"] == t["router_pick"]].copy()

def run_backtest(cfg: dict, start: str, end: str, mode: str = "tagged") -> None:
    months = months_between(start, end)
    symbol = cfg["symbol"]
    p = cfg["params"]
    fs = fsspec.filesystem("s3")

    run_id = make_run_id("ms_edge")
    out_root = cfg["out_root"].rstrip("/")
    out_run = f"{out_root}/run_id={run_id}"

    rows = []
    trades_all = []

    for month in months:
        ev_m = read_events_month(cfg, month, mode=mode)
        print(f"---- {month} ----")
        print("events_hard:", len(ev_m))
        if len(ev_m) == 0:
            continue

        win_s, win_e = build_windows(ev_m["timestamp"], pre_s=float(p["pad_pre_s"]), post_s=float(p["pad_post_s"]))
        print("windows:", len(win_s))

        book_m = read_book_with_next_windows(cfg["book_root"], symbol, month, win_s, win_e, fs=fs)
        print("book kept rows:", len(book_m))

        for c in cfg["configs"]:
            stats, trades_cfg = evaluate_config(
                ev_m, book_m,
                mode=c["mode"],
                runaway_min_bps=c["runaway_min_bps"],
                gate=c["gate"],
                exit_rule=c["exit_rule"],
                runaway_T=int(p["runaway_T"]),
                tol_s=float(p["tol_s"]),
            )
            name = c["name"]
            rows.append({"month": month, "config": name, **stats})
            trades_cfg["day"] = trades_cfg["timestamp"].dt.floor("D")
            trades_cfg["month"] = month
            trades_cfg["config"] = name
            trades_all.append(trades_cfg[["timestamp","pnl_net_bps","day","month","config"]])

            print(f"{name} => n {stats['n']} EV {stats['EV_T_bps']}")

    res = pd.DataFrame(rows)
    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame(
        columns=["timestamp","pnl_net_bps","day","month","config"]
    )
    print("res:", res.shape, "| trades:", trades.shape)

    # ----------------------------
    # GOLDEN CHECK (repro guardrail)
    # ----------------------------
    #expected = cfg.get("expected_trades_rows", None)
    #if expected is not None:
    #    got = int(len(trades))
    #    if got != int(expected):
    #        raise RuntimeError(
    #            f"[GOLDEN CHECK] trades rows mismatch: expected={expected} got={got} "
    #            f"(start={start}, end={end}, mode={mode})"
    #        )

    # outputs tables
    ev_pivot = res.pivot_table(index="config", columns="month", values="EV_T_bps", aggfunc="mean")
    n_pivot  = res.pivot_table(index="config", columns="month", values="n", aggfunc="mean")

    stability = pd.DataFrame({
        "mean_EV": ev_pivot.mean(axis=1),
        "std_EV": ev_pivot.std(axis=1),
        "pos_months": (ev_pivot > 0).sum(axis=1),
        "n_months": ev_pivot.notna().sum(axis=1),
    })
    stability["pos_rate"] = stability["pos_months"] / stability["n_months"]

    # write base artifacts
    write_parquet_s3(res, f"{out_run}/res.parquet")
    write_parquet_s3(trades, f"{out_run}/trades.parquet")
    write_parquet_s3(ev_pivot.reset_index(), f"{out_run}/ev_pivot.parquet")
    write_parquet_s3(n_pivot.reset_index(), f"{out_run}/n_pivot.parquet")
    write_parquet_s3(stability.reset_index(), f"{out_run}/stability.parquet")
    write_json_s3({"cfg": cfg, "start": start, "end": end, "mode": mode, "run_id": run_id}, f"{out_run}/run_config.json")


    # ----------------------------
    # Pool + bootstrap for target configs
    # ----------------------------
    target = [c["name"] for c in cfg["configs"]]
    pool_rows = []
    for name in target:
        t = trades.loc[trades["config"] == name].copy()
        x = t["pnl_net_bps"].dropna().to_numpy(dtype="float64")
        ci_iid = bootstrap_iid(x, n_boot=int(p["n_boot"]), seed=int(p["seed"]))
        ci_blk = bootstrap_block_by_day(t, value_col="pnl_net_bps", day_col="day",
                                        n_boot=int(p["n_boot"]), seed=int(p["seed"]))
        pool_rows.append({
            "config": name,
            "n_trades": int(len(x)),
            "n_days": int(t["day"].nunique()),
            "EV_bps": float(np.mean(x)) if len(x) else np.nan,
            "CI_iid_lo": ci_iid[0], "CI_iid_hi": ci_iid[1],
            "CI_blk_lo": ci_blk[0], "CI_blk_hi": ci_blk[1],
        })
    pool = pd.DataFrame(pool_rows)
    write_parquet_s3(pool, f"{out_run}/bootstrap_pool.parquet")

    # ----------------------------
    # ATR buckets + router
    # ----------------------------
    if len(trades):
        years = sorted(pd.to_datetime(trades["timestamp"], utc=True).dt.year.unique().tolist())
        candles_1m = load_candles_years(cfg["candles_root"], symbol, years)
        atr_tf = compute_atr_bps_from_1m(candles_1m, atr_n=int(p["atr_n"]), tf=str(p["atr_tf"]))

        nb = int(cfg.get("n_vol_buckets", 3))
        trades_vol = attach_vol_bucket(trades, atr_tf, n_buckets=nb)
        
        print("vol buckets:", sorted(trades_vol["vol_bucket"].unique().tolist()))
        print("bucket counts:\n", trades_vol.groupby(["config","vol_bucket"]).size())

        write_parquet_s3(atr_tf, f"{out_run}/atr_tf.parquet")
        write_parquet_s3(trades_vol, f"{out_run}/trades_with_atr_buckets.parquet")

        # bucket summary per config
        bucket_rows = []
        for (name, bucket), g in trades_vol.groupby(["config", "vol_bucket"]):
            x = g["pnl_net_bps"].to_numpy(dtype="float64")
            ci_iid = bootstrap_iid(x, n_boot=int(p["n_boot"]), seed=int(p["seed"]))
            ci_blk = bootstrap_block_by_day(g, value_col="pnl_net_bps", day_col="day",
                                            n_boot=int(p["n_boot"]), seed=int(p["seed"]))
            bucket_rows.append({
                "config": name,
                "bucket": bucket,
                "n_trades": int(len(x)),
                "n_days": int(g["day"].nunique()),
                "EV_bps": float(np.mean(x)),
                "CI_iid_lo": ci_iid[0], "CI_iid_hi": ci_iid[1],
                "CI_blk_lo": ci_blk[0], "CI_blk_hi": ci_blk[1],
                "atr_bps_median": float(g["atr_bps"].median()),
            })
        bucket_summary = pd.DataFrame(bucket_rows)
        write_parquet_s3(bucket_summary, f"{out_run}/bucket_summary.parquet")

        # router
        bo_name = cfg["router"]["bo_name"]
        mr_name = cfg["router"]["mr_name"]
        router_trades = router_pick(trades_vol, bo_name=bo_name, mr_name=mr_name, n_buckets=nb)

        print("router pick counts:\n", router_trades.groupby(["router_pick","vol_bucket"]).size())
        print("router trades:", router_trades.shape)

        x = router_trades["pnl_net_bps"].to_numpy(dtype="float64")
        ci_iid = bootstrap_iid(x, n_boot=int(p["n_boot"]), seed=int(p["seed"]))
        ci_blk = bootstrap_block_by_day(router_trades, value_col="pnl_net_bps", day_col="day",
                                        n_boot=int(p["n_boot"]), seed=int(p["seed"]))
        router_summary = pd.DataFrame([{
            "strategy": "ROUTER",
            "n_trades": int(len(x)),
            "n_days": int(router_trades["day"].nunique()),
            "EV_bps": float(np.mean(x)),
            "CI_iid_lo": ci_iid[0], "CI_iid_hi": ci_iid[1],
            "CI_blk_lo": ci_blk[0], "CI_blk_hi": ci_blk[1],
        }])

        # ----------------------------
        # Router candidates search (which buckets -> MR)
        # Objective: maximize CI_blk_lo (robust edge)
        # ----------------------------
        buckets = sorted(trades_vol["vol_bucket"].dropna().unique().tolist())
        bo_name = cfg["router"]["bo_name"]
        mr_name = cfg["router"]["mr_name"]

        def router_pick_custom(trades_vol_df: pd.DataFrame, mr_buckets: set[str]) -> pd.DataFrame:
            t = trades_vol_df.copy()
            pick = np.where(t["vol_bucket"].isin(list(mr_buckets)), mr_name, bo_name)
            t["router_pick"] = pick
            return t.loc[t["config"] == t["router_pick"]].copy()

        cand_rows = []

        # candidate sets: single bucket + contiguous ranges (by index) + all-but-one
        # (simple + interpretable, avoids 2^K explosion)
        bucket_sets = []

        # singles
        for b in buckets:
            bucket_sets.append({b})

        # contiguous ranges
        for i in range(len(buckets)):
            for j in range(i, len(buckets)):
                bucket_sets.append(set(buckets[i:j+1]))

        # all-but-one
        for b in buckets:
            bucket_sets.append(set([x for x in buckets if x != b]))

        # de-dup
        uniq = []
        seen = set()
        for s in bucket_sets:
            key = ",".join(sorted(s))
            if key not in seen:
                seen.add(key)
                uniq.append(s)

        for mr_set in uniq:
            rt = router_pick_custom(trades_vol, mr_buckets=mr_set)
            x = rt["pnl_net_bps"].to_numpy(dtype="float64")
            if len(x) == 0:
                continue
            ci_blk = bootstrap_block_by_day(rt, value_col="pnl_net_bps", day_col="day",
                                            n_boot=int(p["n_boot"]), seed=int(p["seed"]))
            cand_rows.append({
                "mr_buckets": ",".join(sorted(mr_set)),
                "n_trades": int(len(x)),
                "n_days": int(rt["day"].nunique()),
                "EV_bps": float(np.mean(x)),
                "CI_blk_lo": float(ci_blk[0]),
                "CI_blk_hi": float(ci_blk[1]),
            })

        router_candidates = pd.DataFrame(cand_rows).sort_values(["CI_blk_lo","EV_bps"], ascending=[False, False])
        write_parquet_s3(router_candidates, f"{out_run}/router_candidates.parquet")

        write_parquet_s3(router_trades, f"{out_run}/router_trades.parquet")
        write_parquet_s3(router_summary, f"{out_run}/router_summary.parquet")

        # ----------------------------
        # Bucket DELTA: MR - BO (block bootstrap by day)
        # ----------------------------
        bo_name = cfg["router"]["bo_name"]
        mr_name = cfg["router"]["mr_name"]

        # sanity
        if bo_name not in set(trades_vol["config"].unique()) or mr_name not in set(trades_vol["config"].unique()):
            print("[WARN] bucket_delta skipped: bo/mr config names not found in trades_vol")
        else:
            delta_rows = []
            for bucket in sorted(trades_vol["vol_bucket"].dropna().unique().tolist()):
                bo = trades_vol[(trades_vol["config"] == bo_name) & (trades_vol["vol_bucket"] == bucket)].copy()
                mr = trades_vol[(trades_vol["config"] == mr_name) & (trades_vol["vol_bucket"] == bucket)].copy()

                x_bo = bo["pnl_net_bps"].to_numpy(dtype="float64")
                x_mr = mr["pnl_net_bps"].to_numpy(dtype="float64")

                ev_bo = float(np.mean(x_bo)) if len(x_bo) else np.nan
                ev_mr = float(np.mean(x_mr)) if len(x_mr) else np.nan
                delta_ev = (ev_mr - ev_bo) if np.isfinite(ev_bo) and np.isfinite(ev_mr) else np.nan

                # block bootstrap CI on EV difference (by day)
                # simplest + robust: resample days within each group separately, then subtract means
                ci_lo, ci_hi = bootstrap_block_delta_by_day(
                    bo, mr,
                    value_col="pnl_net_bps", day_col="day",
                    n_boot=int(p["n_boot"]), seed=int(p["seed"])
                )

                delta_rows.append({
                    "bucket": bucket,
                    "delta_EV_bps": float(delta_ev) if np.isfinite(delta_ev) else np.nan,
                    "CI_blk_lo": float(ci_lo),
                    "CI_blk_hi": float(ci_hi),
                    "n_bo": int(len(x_bo)),
                    "n_mr": int(len(x_mr)),
                    "days_bo": int(bo["day"].nunique()),
                    "days_mr": int(mr["day"].nunique()),
                })

            bucket_delta = pd.DataFrame(delta_rows).sort_values("bucket")
            write_parquet_s3(bucket_delta, f"{out_run}/bucket_delta.parquet")

    print(f"✅ done. outputs at: {out_run}")