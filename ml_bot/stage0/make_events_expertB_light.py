#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage0/make_events_expertB_light.py

Expert B "léger":
- Lit un parquet base (events_all_base.parquet) déjà feature-rich
- (Re)construit y depuis mfe_R/mae_R si tp_rr / max_pullback_R changent
- Split time-based + embargo horizon
- Fit LightGBM, écrit events_b.parquet avec pB

Input attendu (dans base): t0 (ou index), dir, R_bps, t0_close + features numériques
Candles 1m requis pour calculer mfe_R/mae_R.

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
# Compute MFE_R and MAE_R 
# -------------------------
def compute_mfe_mae_R_on_events(
    candles_1m: pd.DataFrame,
    events_t0: pd.DatetimeIndex,
    entry_px: np.ndarray,
    ev_dir: np.ndarray,
    *,
    horizon_min: int,
    R_bps: np.ndarray,
) -> pd.DataFrame:
    # Ensure sorted index
    if not candles_1m.index.is_monotonic_increasing:
        candles_1m = candles_1m.sort_index()

    idx_1m = candles_1m.index
    pos = idx_1m.get_indexer(events_t0)

    miss = np.where(pos < 0)[0]
    if len(miss) > 0:
        pos2 = idx_1m.searchsorted(events_t0, side="left")
        ok = pos2 < len(idx_1m)
        pos[pos < 0] = np.where(ok[pos < 0], pos2[pos < 0], -1)
        miss = np.where(pos < 0)[0]
        if len(miss) > 0:
            raise RuntimeError(f"{len(miss)} event t0 not found/snappable in 1m index")

    H = int(horizon_min)
    high = candles_1m["high"].to_numpy(dtype=float)
    low  = candles_1m["low"].to_numpy(dtype=float)

    mfe_R = np.full(len(events_t0), np.nan, dtype=float)
    mae_R = np.full(len(events_t0), np.nan, dtype=float)

    entry = entry_px.astype(float)
    d = ev_dir.astype(int)

    for i, p0 in enumerate(pos):
        s = p0 + 1
        e = p0 + 1 + H
        if e > len(high):
            continue

        hh = np.nanmax(high[s:e])
        ll = np.nanmin(low[s:e])

        if d[i] > 0:  # long
            fav_bps = (hh - entry[i]) / entry[i] * 1e4
            adv_bps = (entry[i] - ll) / entry[i] * 1e4
        else:         # short
            fav_bps = (entry[i] - ll) / entry[i] * 1e4
            adv_bps = (hh - entry[i]) / entry[i] * 1e4

        R = float(R_bps[i]) if np.isfinite(R_bps[i]) else np.nan
        if not np.isfinite(R) or R <= 0:
            continue

        mfe_R[i] = fav_bps / R
        mae_R[i] = adv_bps / R

    out = pd.DataFrame(index=events_t0)
    out["mfe_R"] = mfe_R
    out["mae_R"] = mae_R
    return out


def lift_at_topk(y_true: np.ndarray, score: np.ndarray, top_frac: float = 0.20) -> float:
    n = len(y_true)
    if n == 0:
        return np.nan
    k = max(1, int(n * float(top_frac)))
    idx = np.argsort(-score)[:k]
    base = float(np.mean(y_true))
    top = float(np.mean(y_true[idx]))
    return float(top / base) if base > 0 else np.nan


def label_followthrough_clean(df: pd.DataFrame, tp_rr: float, max_pullback_R: float) -> pd.Series:
    mfe = df["mfe_R"].to_numpy(dtype=float)
    mae = df["mae_R"].to_numpy(dtype=float)
    y = (mfe >= float(tp_rr)) & (mae <= float(max_pullback_R))
    return pd.Series(y.astype(np.int8), index=df.index, name="y")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("stage0 — make ExpertB light (from events_all_base)")
    ap.add_argument("--events-base-in", required=True, help="s3://.../events_all_base.parquet")
    ap.add_argument("--events-base-out", default="s3://tradebot-config-tokyo/data/s1/events_b.parquet")

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--s3-candles-root", default="s3://tradebot-config-tokyo/data/bougie")
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--end", default="2025-10-31")
    ap.add_argument("--y-mode", default="auto", choices=["auto","use-existing","recompute"])

    ap.add_argument("--horizon-min", type=int, default=60)  # embargo only
    ap.add_argument("--mfe-horizon-min", type=int, default=60)
    ap.add_argument("--embargo-min", type=int, default=60)
    ap.add_argument("--tp-rr", type=float, default=1.3)
    ap.add_argument("--max-pullback-R", type=float, default=0.6)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-frac", type=float, default=0.20)
    ap.add_argument("--min-events-test", type=int, default=2000)
    ap.add_argument("--report", action="store_true")

    args = ap.parse_args()

    log("===================================")
    log("STAGE0 — MAKE EXPERT B LIGHT")
    log("===================================")
    log(f"[load] {args.events_base_in}")

    df = read_parquet_any(args.events_base_in)

    # --- Ensure time index
    if "t0" in df.columns:
        df["t0"] = pd.to_datetime(df["t0"], utc=True)
        df = df.sort_values("t0").set_index("t0")
    else:
        # si ton parquet base n'a pas t0 en colonne, on tente index
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

    # --- Required columns
    required = ["dir", "R_bps", "t0_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Base parquet missing required columns: {missing}")

    # --- Load candles 1m to compute mfe/mae
    years = sorted({pd.Timestamp(df.index.min()).year, pd.Timestamp(args.end).year})
    years = list(range(min(years), max(years) + 1))
    log(f"[load] candles 1m years={years}")
    candles_1m = load_candles_yearly(args.s3_candles_root, args.symbol, years=years)
    candles_1m = candles_1m.sort_index()

    # --- Compute mfe_R / mae_R for each event row
    df = df.dropna(subset=["dir", "R_bps", "t0_close"]).copy()
    mfe_mae = compute_mfe_mae_R_on_events(
        candles_1m=candles_1m,
        events_t0=df.index,
        entry_px=df["t0_close"].astype(float).to_numpy(),
        ev_dir=df["dir"].to_numpy(dtype=np.int8),
        horizon_min=int(args.horizon_min),
        R_bps=df["R_bps"].astype(float).to_numpy(),
    )

    df = df.join(mfe_mae, how="left")

    y_mode = args.y_mode
    has_y = "y" in df.columns
    if y_mode == "use-existing":
        if not has_y:
            raise RuntimeError("y-mode=use-existing but base parquet has no 'y' column")
        df["y"] = df["y"].astype(np.int8)
        log("[labels] using existing y from base parquet")
    elif y_mode == "recompute":
        df["y"] = label_followthrough_clean(df, tp_rr=args.tp_rr, max_pullback_R=args.max_pullback_R)
        log("[labels] recomputed y from mfe/mae")
    else:  # auto
        if has_y:
            df["y"] = df["y"].astype(np.int8)
            log("[labels] auto: using existing y from base parquet")
        else:
            df["y"] = label_followthrough_clean(df, tp_rr=args.tp_rr, max_pullback_R=args.max_pullback_R)
            log("[labels] auto: recomputed y from mfe/mae")

    # Drop rows where we couldn't compute
    df = df.dropna(subset=["mfe_R", "mae_R"]).copy()

    # --- Build y from mfe/mae using current tp_rr/pullback
    df["y"] = label_followthrough_clean(df, tp_rr=args.tp_rr, max_pullback_R=args.max_pullback_R)

    base = float(df["y"].mean())
    log(f"[labels] horizon={args.horizon_min}min tp_rr={args.tp_rr} max_pullback_R={args.max_pullback_R} base_rate(y=1)={base:.4f}")

    # --- Split + embargo
    tr_end_ts = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    te_start_ts = pd.Timestamp(args.test_start, tz="UTC")
    embargo = pd.Timedelta(minutes=int(args.horizon_min))

    train = df.loc[:tr_end_ts - embargo].copy()
    test  = df.loc[te_start_ts + embargo:].copy()
    log(f"[split+embargo] train_events={len(train):,} test_events={len(test):,} embargo={embargo}")
    log(f"[labels] mfe_horizon={args.mfe_horizon_min} embargo={args.embargo_min} ...")

    if len(train) < 2000 or len(test) < 2000:
        raise RuntimeError(f"Not enough events after split. train={len(train)} test={len(test)}")

    # --- Feature selection: numeric only, exclude targets
    drop_cols = {"y", "mfe_R", "mae_R"}
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    # On garde explicitement dir/R_bps/t0_close si jamais ils ne sont pas numeric dtype
    for c in ["dir", "R_bps", "t0_close"]:
        if c in df.columns and c not in numeric_cols and pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    X_train = train[numeric_cols].copy()
    y_train = train["y"].astype(np.int8).copy()
    X_test  = test[numeric_cols].copy()
    y_test  = test["y"].astype(np.int8).copy()

    log("[fit] training LightGBM (Expert B light)...")
    m = fit_lgbm(X_train, y_train, seed=args.seed)

    pB_test = predict_proba(m, X_test)

    if args.report:
        if len(test) >= int(args.min_events_test):
            auc = float(roc_auc_score(y_test.to_numpy(dtype=int), pB_test))
            lift = float(lift_at_topk(y_test.to_numpy(dtype=int), pB_test, top_frac=float(args.top_frac)))
            base_test = float(y_test.mean())

            log("\n============================")
            log("REPORT — TEST METRICS (EXPERT B LIGHT)")
            log("============================")
            log(f"horizon             : {args.horizon_min}min")
            log(f"label               : y=1 if MFE>= {args.tp_rr}R and MAE<= {args.max_pullback_R}R")
            log("----------------------------")
            log(f"events(test)        : {len(test):,}")
            log(f"base rate (y=1)     : {base_test:.4f}")
            log(f"AUC ML (events)     : {auc:.4f}")
            log(f"Lift@Top{int(args.top_frac*100)}% ML : {lift:.3f}")
            log("============================")
        else:
            log(f"[report] ❌ Too few events to report reliably: n={len(test)} < {args.min_events_test}")

    # --- Score all + write
    pB_all = predict_proba(m, df[numeric_cols])

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
    out["pB"] = pB_all.astype(float)

    log(f"[write] {args.events_base_out} rows={len(out):,}")
    write_parquet_any(out.reset_index(drop=True), args.events_base_out)
    log("✅ Done.")


if __name__ == "__main__":
    main()