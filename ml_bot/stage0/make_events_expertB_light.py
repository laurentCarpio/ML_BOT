#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage0/make_events_expertB_light.py

Expert B "léger" (clean & reproducible):
- Input: events_all_base.parquet (feature-rich)
- Computes mfe_R / mae_R from 1m candles over mfe_horizon_min
- Builds label y from (mfe_R >= tp_rr) AND (mae_R <= max_pullback_R)
- Strict time split with embargo (anti-leak)
- Fit LightGBM -> pB
- Output: events_b_light.parquet (t0, dir, side, t0_close, R_bps, mfe_R, mae_R, tp_rr, max_pullback_R, y, pB)

Notes:
- y is NOT expected in base parquet.
- This script always recomputes y (no y-mode).
"""

import argparse
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from sklearn.metrics import roc_auc_score

from ml_bot.stage0.core_model_lgbm import fit_lgbm, predict_proba
from ml_bot.stage0.core_data_rightedge import log, load_candles_yearly


# -------------------------
# IO
# -------------------------
def read_parquet_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_input_file(s3_path) as f:
            return pq.read_table(f).to_pandas()
    return pq.read_table(path).to_pandas()


def write_parquet_any(df: pd.DataFrame, path: str):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_output_stream(s3_path) as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, path)


# -------------------------
# MFE/MAE in R
# -------------------------
def compute_mfe_mae_R_on_events(
    candles_1m: pd.DataFrame,
    events_t0: pd.DatetimeIndex,
    entry_px: np.ndarray,
    ev_dir: np.ndarray,
    *,
    mfe_horizon_min: int,
    R_bps: np.ndarray,
) -> pd.DataFrame:
    if not candles_1m.index.is_monotonic_increasing:
        candles_1m = candles_1m.sort_index()

    idx_1m = candles_1m.index
    pos = idx_1m.get_indexer(events_t0)

    # snap missing timestamps to next candle
    miss = np.where(pos < 0)[0]
    if len(miss) > 0:
        pos2 = idx_1m.searchsorted(events_t0, side="left")
        ok = pos2 < len(idx_1m)
        pos[pos < 0] = np.where(ok[pos < 0], pos2[pos < 0], -1)
        miss = np.where(pos < 0)[0]
        if len(miss) > 0:
            raise RuntimeError(f"{len(miss)} event t0 not found/snappable in 1m index")

    H = int(mfe_horizon_min)
    high = candles_1m["high"].to_numpy(dtype=float)
    low = candles_1m["low"].to_numpy(dtype=float)

    mfe_R = np.full(len(events_t0), np.nan, dtype=float)
    mae_R = np.full(len(events_t0), np.nan, dtype=float)

    entry = entry_px.astype(float)
    d = ev_dir.astype(int)
    Rb = R_bps.astype(float)

    for i, p0 in enumerate(pos):
        s = p0 + 1
        e = p0 + 1 + H
        if e > len(high):
            continue

        R = float(Rb[i]) if np.isfinite(Rb[i]) else np.nan
        if not np.isfinite(R) or R <= 0:
            continue

        hh = np.nanmax(high[s:e])
        ll = np.nanmin(low[s:e])

        if d[i] > 0:  # long
            fav_bps = (hh - entry[i]) / entry[i] * 1e4
            adv_bps = (entry[i] - ll) / entry[i] * 1e4
        else:         # short
            fav_bps = (entry[i] - ll) / entry[i] * 1e4
            adv_bps = (hh - entry[i]) / entry[i] * 1e4

        mfe_R[i] = fav_bps / R
        mae_R[i] = adv_bps / R

    return pd.DataFrame({"mfe_R": mfe_R, "mae_R": mae_R}, index=events_t0)


def label_followthrough(mfe_R: np.ndarray, mae_R: np.ndarray, tp_rr: float, max_pullback_R: float) -> np.ndarray:
    return ((mfe_R >= float(tp_rr)) & (mae_R <= float(max_pullback_R))).astype(np.int8)


def lift_at_topk(y_true: np.ndarray, score: np.ndarray, top_frac: float = 0.20) -> float:
    n = len(y_true)
    if n == 0:
        return np.nan
    k = max(1, int(n * float(top_frac)))
    idx = np.argsort(-score)[:k]
    base = float(np.mean(y_true))
    top = float(np.mean(y_true[idx]))
    return float(top / base) if base > 0 else np.nan


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("stage0 — make ExpertB light (from events_all_base)")
    ap.add_argument("--events-base-in", required=True)
    ap.add_argument("--events-b-out", required=True)

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--s3-candles-root", default="s3://tradebot-config-tokyo/data/bougie")

    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--end", default="2025-10-31")

    ap.add_argument("--mfe-horizon-min", type=int, default=60)
    ap.add_argument("--embargo-min", type=int, default=60)

    ap.add_argument("--tp-rr", type=float, default=1.3)
    ap.add_argument("--max-pullback-R", type=float, default=0.6)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--top-frac", type=float, default=0.20)
    ap.add_argument("--min-events-test", type=int, default=2000)

    args = ap.parse_args()

    log("===================================")
    log("STAGE0 — MAKE EXPERT B LIGHT (CLEAN)")
    log("===================================")
    log(f"[load] {args.events_base_in}")

    df = read_parquet_any(args.events_base_in)

    # time index
    if "t0" in df.columns:
        df["t0"] = pd.to_datetime(df["t0"], utc=True)
        df = df.sort_values("t0").set_index("t0")
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

    # required cols in base
    required = ["dir", "R_bps", "t0_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Base parquet missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    # load candles 1m
    years = list(range(df.index.min().year, pd.Timestamp(args.end).year + 1))
    log(f"[load] candles 1m years={years}")
    candles_1m = load_candles_yearly(args.s3_candles_root, args.symbol, years=years).sort_index()

    # compute mfe/mae
    mfe_mae = compute_mfe_mae_R_on_events(
        candles_1m=candles_1m,
        events_t0=df.index,
        entry_px=df["t0_close"].astype(float).to_numpy(),
        ev_dir=df["dir"].to_numpy(dtype=np.int8),
        mfe_horizon_min=int(args.mfe_horizon_min),
        R_bps=df["R_bps"].astype(float).to_numpy(),
    )
    df = df.join(mfe_mae, how="left")
    df = df.dropna(subset=["mfe_R", "mae_R"]).copy()

    # label y (always recompute)
    df["y"] = label_followthrough(
        df["mfe_R"].to_numpy(dtype=float),
        df["mae_R"].to_numpy(dtype=float),
        tp_rr=float(args.tp_rr),
        max_pullback_R=float(args.max_pullback_R),
    )

    base_rate = float(df["y"].mean())
    log(f"[labels] mfe_horizon={args.mfe_horizon_min} tp_rr={args.tp_rr} max_pullback_R={args.max_pullback_R} base_rate(y=1)={base_rate:.4f}")

    # strict split + embargo
    tr_end_ts = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    te_start_ts = pd.Timestamp(args.test_start, tz="UTC")
    embargo = pd.Timedelta(minutes=int(args.embargo_min))

    train = df.loc[:tr_end_ts - embargo].copy()
    test = df.loc[te_start_ts + embargo:].copy()

    log(f"[split+embargo] train_events={len(train):,} test_events={len(test):,} embargo={embargo}")

    if len(train) < 2000 or len(test) < 2000:
        raise RuntimeError(f"Not enough events after split. train={len(train)} test={len(test)}")

    # numeric features (exclude targets & outcome proxies)
    drop_cols = {"y", "mfe_R", "mae_R"}
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    X_train = train[numeric_cols]
    y_train = train["y"].astype(np.int8)
    X_test = test[numeric_cols]
    y_test = test["y"].astype(np.int8)

    log(f"[fit] LightGBM features={len(numeric_cols)} ...")
    m = fit_lgbm(X_train, y_train, seed=args.seed)

    # report
    if args.report and len(test) >= int(args.min_events_test):
        p_test = predict_proba(m, X_test)
        auc = float(roc_auc_score(y_test.to_numpy(dtype=int), p_test))
        lift = float(lift_at_topk(y_test.to_numpy(dtype=int), p_test, top_frac=float(args.top_frac)))
        log("\n============================")
        log("REPORT — TEST METRICS (EXPERT B LIGHT)")
        log("============================")
        log(f"events(test)        : {len(test):,}")
        log(f"base rate (y=1)     : {float(y_test.mean()):.4f}")
        log(f"AUC                : {auc:.4f}")
        log(f"Lift@Top{int(args.top_frac*100)}%      : {lift:.3f}")
        log("============================")

    # score all
    pB_all = predict_proba(m, df[numeric_cols]).astype(float)

    out = pd.DataFrame(index=df.index)
    out["t0"] = pd.to_datetime(df.index, utc=True)
    out["dir"] = df["dir"].astype(np.int8)
    out["side"] = np.where(out["dir"].to_numpy() > 0, "long", "short")
    out["t0_close"] = df["t0_close"].astype(float)
    out["R_bps"] = df["R_bps"].astype(float)
    out["mfe_R"] = df["mfe_R"].astype(float)
    out["mae_R"] = df["mae_R"].astype(float)
    out["tp_rr"] = float(args.tp_rr)
    out["max_pullback_R"] = float(args.max_pullback_R)
    out["y"] = df["y"].astype(np.int8)
    out["pB"] = pB_all

    log(f"[write] {args.events_b_out} rows={len(out):,}")
    write_parquet_any(out.reset_index(drop=True), args.events_b_out)
    log("✅ Done.")


if __name__ == "__main__":
    main()