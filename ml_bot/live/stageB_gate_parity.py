#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import s3fs
import xgboost as xgb

from ml_bot.live.stageB_gate import StageBGate, StageBGateConfig


def main():
    ap = argparse.ArgumentParser("StageB gate parity check (offline vs StageBGate)")
    ap.add_argument("--model-uri", default="s3://tradebot-config-tokyo/models/xgb-baseline/stageB/baseline_xgb_stageB_20260203-100728.json")
    ap.add_argument("--columns-json", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/_meta/columns.json")
    
    ap.add_argument("--thr-s", type=float, default="0.3977672740")
    ap.add_argument("--stageb-root", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n-files", type=int, default=1)
    ap.add_argument("--n-rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pthr-col", default="audit_p_thr_ev0")
    args = ap.parse_args()

    fs = s3fs.S3FileSystem()

    cfg = StageBGateConfig(
        model_uri=args.model_uri,
        columns_json=args.columns_json,
        thr_s=float(args.thr_s),
        strict_features=True,
    )
    gate = StageBGate(cfg, fs=fs)

    # ---- pick parquet files
    prefix = f"{args.stageb_root}/split={args.split}/parquet"
    pattern = prefix.replace("s3://", "") + "/*.parquet"
    paths = sorted([f"s3://{p}" for p in fs.glob(pattern)])
    if not paths:
        raise FileNotFoundError(f"No parquet files under {prefix}")

    rng = np.random.default_rng(int(args.seed))
    sel = paths if int(args.n_files) <= 0 else [paths[i] for i in rng.choice(len(paths), size=min(int(args.n_files), len(paths)), replace=False)]

    # ---- load sample rows
    cols = gate.feature_cols + [args.pthr_col]
    df = pd.concat([pd.read_parquet(p, columns=cols, engine="pyarrow") for p in sel], ignore_index=True)
    if len(df) > int(args.n_rows):
        df = df.sample(n=int(args.n_rows), random_state=int(args.seed)).reset_index(drop=True)

    # ---- OFFLINE compute p,s,allow in one batch
    X = df[gate.feature_cols].to_numpy(np.float32, copy=False)
    pthr = df[args.pthr_col].to_numpy(np.float64, copy=False)

    dm = xgb.DMatrix(X, feature_names=gate.feature_cols, missing=np.nan)
    p_off = gate.booster.predict(dm, output_margin=False).astype(np.float64)
    s_off = p_off - pthr
    allow_off = (s_off >= gate.thr_s)

    # ---- ONLINE (StageBGate) per-row
    p_on = np.empty_like(p_off)
    s_on = np.empty_like(s_off)
    allow_on = np.empty_like(allow_off)

    for i in range(len(df)):
        row = df.iloc[i][gate.feature_cols].to_dict()
        allow, out = gate.decide(row, p_thr_ev0=float(pthr[i]))
        p_on[i] = out["p"]
        s_on[i] = out["s"]
        allow_on[i] = allow

    # ---- Assertions (tight)
    # proba should match extremely closely (same booster)
    if not np.allclose(p_on, p_off, rtol=0.0, atol=1e-12):
        j = int(np.argmax(np.abs(p_on - p_off)))
        raise RuntimeError(f"Mismatch p at row {j}: on={p_on[j]} off={p_off[j]} diff={p_on[j]-p_off[j]}")

    if not np.allclose(s_on, s_off, rtol=0.0, atol=1e-12):
        j = int(np.argmax(np.abs(s_on - s_off)))
        raise RuntimeError(f"Mismatch s at row {j}: on={s_on[j]} off={s_off[j]} diff={s_on[j]-s_off[j]}")

    if not np.array_equal(allow_on, allow_off):
        j = int(np.where(allow_on != allow_off)[0][0])
        raise RuntimeError(f"Mismatch allow at row {j}: on={allow_on[j]} off={allow_off[j]} (s_on={s_on[j]} thr_s={gate.thr_s})")

    print(
        f"[OK] parity check passed on split={args.split} rows={len(df)} "
        f"| allow_frac={float(np.mean(allow_on)):.6g} | mean_s_cond_allow={float(np.mean(s_on[allow_on])) if np.any(allow_on) else float('nan'):.6g}"
    )


if __name__ == "__main__":
    main()