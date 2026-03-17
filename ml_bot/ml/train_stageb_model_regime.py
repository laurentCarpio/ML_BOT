#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ml_bot/ml/train_stageb_model_regime.py

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import json
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from xgboost import XGBClassifier

from ml_bot.backtest.lib.io_s3 import read_parquet_s3, write_parquet_s3, write_json_s3, make_run_id


@dataclass
class TrainConfig:
    dataset_path: str
    out_root: str

    label_col: str = "y_go"
    vol_regime3: str = "mid"
    min_train_rows: int = 80

    random_state: int = 7
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 4.0
    reg_alpha: float = 1.0
    min_child_weight: float = 8.0

    top_fracs: tuple[float, ...] = (0.05, 0.10, 0.20)


CORE_FEATURES = [
    "MS",
    "micro_bias_bps",
    "abs_micro_bias_bps",
    "OBI_10",
    "abs_OBI_10",
    "TI",
    "abs_TI",
    "thinning_opp_3",
    "persist_micro_ms",
    "persist_obi10_ms",
    "range_60s_bps",
    "atr_bps",
]

BASE_EXCLUDE = {
    "timestamp", "month", "split", "symbol", "router_branch", "vol_bucket", "t_atr",
    "pnl_net_bps", "y_go", "y_strong", "is_tradeable_baseline", "fees_rt_bps", "slip_bps",
}

def write_file_s3(local_path: str, s3_path: str) -> None:
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Expected s3 path, got: {s3_path}")

    no_scheme = s3_path[len("s3://"):]
    bucket, key = no_scheme.split("/", 1)

    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)

def map_regime3(v: str) -> str:
    if v in ("b0", "b1"):
        return "low"
    if v == "b2":
        return "mid"
    if v in ("b3", "b4"):
        return "high"
    return "other"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "micro_bias_bps" in x.columns:
        x["abs_micro_bias_bps"] = x["micro_bias_bps"].abs()
    if "TI" in x.columns:
        x["abs_TI"] = x["TI"].abs()
    if "OBI_10" in x.columns:
        x["abs_OBI_10"] = x["OBI_10"].abs()
    return x

def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {}
    if len(np.unique(y_true)) >= 2:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["ap"] = float(average_precision_score(y_true, y_prob))
        out["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    else:
        out["auc"] = np.nan
        out["ap"] = np.nan
        out["logloss"] = np.nan
    return out

def make_decile_table(df_scored: pd.DataFrame, label_col: str) -> pd.DataFrame:
    x = df_scored.copy().dropna(subset=["score", "pnl_net_bps", label_col])
    x["decile"] = pd.qcut(x["score"].rank(method="first"), 10, labels=False)
    x["decile"] = 9 - x["decile"].astype(int)

    out = (
        x.groupby("decile", as_index=False)
        .agg(
            n=("pnl_net_bps", "count"),
            y_rate=(label_col, "mean"),
            ev_bps=("pnl_net_bps", "mean"),
            pnl_p50=("pnl_net_bps", "median"),
            pnl_p95=("pnl_net_bps", lambda s: np.quantile(s, 0.95)),
            score_min=("score", "min"),
            score_max=("score", "max"),
        )
        .sort_values("decile")
        .reset_index(drop=True)
    )
    return out

def make_topk_table(df_scored: pd.DataFrame, top_fracs: tuple[float, ...], label_col: str) -> pd.DataFrame:
    x = df_scored.copy().dropna(subset=["score", "pnl_net_bps", label_col]).sort_values("score", ascending=False)
    n = len(x)

    rows = []
    for frac in top_fracs:
        k = max(1, int(round(n * frac)))
        sub = x.iloc[:k]
        rows.append({
            "top_frac": float(frac),
            "n": int(len(sub)),
            "y_rate": float(sub[label_col].mean()),
            "ev_bps": float(sub["pnl_net_bps"].mean()),
            "p50_bps": float(sub["pnl_net_bps"].median()),
            "p95_bps": float(np.quantile(sub["pnl_net_bps"], 0.95)),
            "score_cut": float(sub["score"].iloc[-1]),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser("Train StageB BO model by vol regime")
    ap.add_argument("--dataset-path", default="s3://tradebot-config-tokyo/research/ms_edge/ml/stageb_dataset.parquet")
    ap.add_argument("--out-root", default="s3://tradebot-config-tokyo/research/ms_edge/ml/runs")
    ap.add_argument("--label-col", default="y_go", choices=["y_go", "y_strong"])
    ap.add_argument("--vol-regime3", default="mid", choices=["low", "mid", "high"])
    args = ap.parse_args()

    cfg = TrainConfig(
        dataset_path=args.dataset_path,
        out_root=args.out_root,
        label_col=args.label_col,
        vol_regime3=args.vol_regime3,
    )

    run_id = make_run_id(f"stageb_ml_{cfg.vol_regime3}")
    out_run = f"{cfg.out_root.rstrip('/')}/run_id={run_id}"

    df = read_parquet_s3(cfg.dataset_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.loc[df["is_tradeable_baseline"] == 1].copy()
    df = df.loc[df["router_branch"] == "BO"].copy()
    df["vol_regime3"] = df["vol_bucket"].map(map_regime3)
    df = df.loc[df["vol_regime3"] == cfg.vol_regime3].copy()
    df = df.dropna(subset=[cfg.label_col, "pnl_net_bps", "split"]).copy()
    df[cfg.label_col] = pd.to_numeric(df[cfg.label_col], errors="coerce").astype("int8")

    print("vol_regime3:", cfg.vol_regime3)
    print("shape:", df.shape)
    print("split counts:\n", df["split"].value_counts())
    print(f"{cfg.label_col} rate by split:\n", df.groupby("split")[cfg.label_col].mean())

    if len(df.loc[df["split"] == "train"]) < cfg.min_train_rows:
        raise SystemExit(f"Not enough training rows: {len(df.loc[df['split']=='train'])}")

    df_feat = build_features(df)
    feature_cols = [c for c in CORE_FEATURES if c in df_feat.columns]

    train = df_feat.loc[df_feat["split"] == "train"].copy()
    val = df_feat.loc[df_feat["split"] == "val"].copy()
    test = df_feat.loc[df_feat["split"] == "test"].copy()

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train[cfg.label_col].to_numpy(dtype=np.int8)

    X_val = val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_val = val[cfg.label_col].to_numpy(dtype=np.int8)

    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test[cfg.label_col].to_numpy(dtype=np.int8)

    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        min_child_weight=cfg.min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=cfg.random_state,
        n_jobs=4,
        tree_method="hist",
    )

    print("training model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    for name, split_df, X, y in [
        ("train", train, X_train, y_train),
        ("val", val, X_val, y_val),
        ("test", test, X_test, y_test),
    ]:
        split_df["score"] = model.predict_proba(X)[:, 1]
        metrics = eval_metrics(y, split_df["score"].to_numpy())

        print(f"\n=== {name.upper()} METRICS ===")
        print(metrics)

        dec = make_decile_table(split_df, label_col=cfg.label_col)
        topk = make_topk_table(split_df, top_fracs=cfg.top_fracs, label_col=cfg.label_col)

        print(f"\n{name.upper()} deciles")
        print(dec.to_string(index=False))

        print(f"\n{name.upper()} top-k")
        print(topk.to_string(index=False))

        write_parquet_s3(split_df[[
            "timestamp", "month", "split", "symbol", "vol_bucket",
            "pnl_net_bps", cfg.label_col, "score"
        ]].copy(), f"{out_run}/scored_{name}.parquet")

        write_parquet_s3(dec, f"{out_run}/deciles_{name}.parquet")
        write_parquet_s3(topk, f"{out_run}/topk_{name}.parquet")
        write_json_s3(metrics, f"{out_run}/metrics_{name}.json")

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print("\n=== FEATURE IMPORTANCE ===")
    print(imp.to_string(index=False))
    write_parquet_s3(imp, f"{out_run}/feature_importance.parquet")

    # save model artifact
    with tempfile.TemporaryDirectory() as td:
        model_path = str(Path(td) / "xgb_model.json")
        model.get_booster().save_model(model_path)
        write_file_s3(model_path, f"{out_run}/xgb_model.json")

    write_json_s3({
        "run_id": run_id,
        "config": asdict(cfg),
        "feature_cols": feature_cols,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "model_artifact": "xgb_model.json",
        "score_column": "score",
        "model_type": "xgboost_binary_classifier",
    }, f"{out_run}/run_config.json")

    print(f"\n✅ run written to: {out_run}")

if __name__ == "__main__":
    main()