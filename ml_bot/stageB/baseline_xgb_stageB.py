#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import s3fs
import xgboost as xgb


# -----------------------------
# S3 helpers
# -----------------------------
def read_json_s3(fs: s3fs.S3FileSystem, uri: str) -> dict:
    with fs.open(uri, "rb") as f:
        return json.load(f)

def sample_paths(paths: List[str], n_files: int, seed: int) -> List[str]:
    if n_files <= 0 or n_files >= len(paths):
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=n_files, replace=False)
    return [paths[i] for i in idx]


# -----------------------------
# Data loading
# -----------------------------
def load_split_df(
    fs: s3fs.S3FileSystem,
    paths: List[str],
    label_col: str,
    feature_cols: List[str],
    artifact_key: str,
    max_rows: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """
    Charge un split depuis une liste de paths (parquet ou csv.gz).
    - max_rows: si défini, sous-échantillonne globalement après concat.
    """
    use_cols = [label_col] + feature_cols

    dfs = []
    for p in paths:
        if artifact_key == "stageB_parquet":
            dfp = pd.read_parquet(p, columns=use_cols, engine="pyarrow")
        else:
            dfp = pd.read_csv(p, compression="gzip", usecols=use_cols)
        dfs.append(dfp)

    df = pd.concat(dfs, ignore_index=True)

    # shuffle + cap rows (optionnel)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    return df


def sanity_check_df(df: pd.DataFrame, label_col: str, feature_cols: List[str], where: str):
    # label in {0,1}
    u = pd.unique(df[label_col])
    if not np.all(np.isin(u, [0, 1])):
        raise RuntimeError(f"[{where}] label not binary. uniques={u[:10]}")

    # numeric features
    bad = []
    for c in feature_cols:
        dt = df[c].dtype
        if not (pd.api.types.is_float_dtype(dt) or pd.api.types.is_integer_dtype(dt)):
            bad.append((c, str(dt)))
    if bad:
        raise RuntimeError(f"[{where}] non-numeric feature dtypes (first 10): {bad[:10]}")

    # inf check
    X = df[feature_cols].to_numpy(dtype="float64", copy=False)
    if np.isinf(X).any():
        raise RuntimeError(f"[{where}] found +/-inf in features")


# -----------------------------
# Training
# -----------------------------
def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[List[str]],
    seed: int,
    scale_pos_weight: Optional[float],
    params_override: Optional[Dict[str, str]] = None,
) -> Tuple[xgb.Booster, dict]:
    # baseline params (simples, robustes)
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 3,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 15.0,
        "reg_alpha": 1.0,
        "gamma": 2.0,
        "max_delta_step": 1,
        "seed": int(seed),
        "eval_metric": ["logloss", "aucpr"],
    }

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)

    # overrides CLI (clé=valeur)
    if params_override:
        for k, v in params_override.items():
            # cast simple
            if v.lower() in ("true", "false"):
                params[k] = (v.lower() == "true")
            else:
                try:
                    if "." in v or "e" in v.lower():
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except Exception:
                    params[k] = v

    dtr = xgb.DMatrix(X_train, label=y_train, missing=np.nan, feature_names=feature_names)
    dva = xgb.DMatrix(X_val, label=y_val, missing=np.nan, feature_names=feature_names)

    evals = [(dtr, "train"), (dva, "val")]

    num_round = 2000
    early_stopping_rounds = 80

    t0 = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )
    train_sec = time.time() - t0

    best = {
        "best_iteration": int(booster.best_iteration) if booster.best_iteration is not None else None,
        "best_score": float(booster.best_score) if booster.best_score is not None else None,
        "train_seconds": float(train_sec),
        "params": params,
    }
    return booster, best


def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad --xgb-param '{it}' (expected key=value)")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def list_stageb_paths_by_split(
    fs: s3fs.S3FileSystem,
    stageb_root: str,
    split: str,
    artifact_key: str,
) -> List[str]:
    if artifact_key == "stageB_parquet":
        sub = "parquet"
        suffix = ".parquet"
    else:
        sub = "xgb"
        suffix = ".csv.gz"

    prefix = f"{stageb_root}/split={split}/{sub}"
    pattern = prefix.replace("s3://", "") + f"/*{suffix}"

    paths = [f"s3://{p}" for p in fs.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"No StageB files for split={split} under {prefix}")
    return sorted(paths)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser("Baseline XGBoost on StageB (StageB columns.json + S3 listing)")
    ap.add_argument("--stageb-root", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1")

    # StageB meta contract (chez nous: _meta/columns.json)
    ap.add_argument("--columns-json", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/_meta/columns.json")

    ap.add_argument("--artifact-key", default="stageB_parquet", choices=["stageB_parquet", "stageB_csv_gz"])
    ap.add_argument("--train-files", type=int, default=10, help="Nb de fichiers train à lire (0=all)")
    ap.add_argument("--val-files", type=int, default=5, help="Nb de fichiers val à lire (0=all)")
    ap.add_argument("--train-max-rows", type=int, default=300000, help="Cap lignes train après concat (0=disable)")
    ap.add_argument("--val-max-rows", type=int, default=200000, help="Cap lignes val après concat (0=disable)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out-model", default="/tmp/baseline_xgb_stageB.json", help="Chemin local de sortie modèle")
    ap.add_argument("--out-report", default="/tmp/baseline_xgb_stageB_report.json", help="Chemin local du report")
    ap.add_argument("--save-to-s3", default="s3://tradebot-config-tokyo/models/xgb-baseline/stageB/baseline_xgb_stageB.json")
    ap.add_argument("--save-report-to-s3", default="s3://tradebot-config-tokyo/models/xgb-baseline/stageB/baseline_xgb_stageB_report.json")

    ap.add_argument("--xgb-param", action="append", default=[], help="Override param: key=value (repeatable)")
    args = ap.parse_args()

    fs = s3fs.S3FileSystem()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    def _with_stamp(uri_or_path: str, stamp: str) -> str:
        # gère /tmp/xxx.json ou s3://.../xxx.json
        base = uri_or_path
        if base.endswith(".json"):
            return base[:-5] + f"_{stamp}.json"
        return base + f"_{stamp}"

    out_model_local  = _with_stamp(args.out_model, stamp)
    out_report_local = _with_stamp(args.out_report, stamp)

    out_model_s3  = _with_stamp(args.save_to_s3, stamp) if args.save_to_s3.strip() else ""
    out_report_s3 = _with_stamp(args.save_report_to_s3, stamp) if args.save_report_to_s3.strip() else ""
    
    # --- meta ---
    columns_meta = read_json_s3(fs, args.columns_json)

    label_col = columns_meta.get("label_col_for_csv") or columns_meta.get("label_col") or "label_A"
    feature_cols = columns_meta.get("feature_cols_xgb") or columns_meta.get("feature_cols_all")
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError("No feature cols in columns.json (expected feature_cols_xgb or feature_cols_all)")

    # --- list data paths ---
    tr_paths = list_stageb_paths_by_split(fs, args.stageb_root, "train", args.artifact_key)
    va_paths = list_stageb_paths_by_split(fs, args.stageb_root, "val", args.artifact_key)

    # --- sample files ---
    tr_sel = sample_paths(tr_paths, int(args.train_files), int(args.seed)) if int(args.train_files) > 0 else tr_paths
    va_sel = sample_paths(va_paths, int(args.val_files), int(args.seed) + 1) if int(args.val_files) > 0 else va_paths

    train_max_rows = int(args.train_max_rows) if int(args.train_max_rows) > 0 else None
    val_max_rows = int(args.val_max_rows) if int(args.val_max_rows) > 0 else None

    print(f"[meta] label_col={label_col} | n_features={len(feature_cols)}")
    print(f"[data] train_paths={len(tr_sel)} (of {len(tr_paths)}) | val_paths={len(va_sel)} (of {len(va_paths)})")

    # --- load dataframes ---
    df_tr = load_split_df(fs, tr_sel, label_col, feature_cols, args.artifact_key, train_max_rows, args.seed)
    df_va = load_split_df(fs, va_sel, label_col, feature_cols, args.artifact_key, val_max_rows, args.seed + 1)

    # --- sanity ---
    sanity_check_df(df_tr, label_col, feature_cols, "TRAIN")
    sanity_check_df(df_va, label_col, feature_cols, "VAL")

    # --- label stats ---
    pos_tr = int((df_tr[label_col] == 1).sum())
    neg_tr = int((df_tr[label_col] == 0).sum())
    pos_va = int((df_va[label_col] == 1).sum())
    neg_va = int((df_va[label_col] == 0).sum())

    pos_rate_tr = pos_tr / max(pos_tr + neg_tr, 1)
    pos_rate_va = pos_va / max(pos_va + neg_va, 1)

    spw = (neg_tr / max(pos_tr, 1))
    print(f"[label] train pos/neg={pos_tr}/{neg_tr} pos_rate={pos_rate_tr:.6g} scale_pos_weight={spw:.3f}")
    print(f"[label] val   pos/neg={pos_va}/{neg_va} pos_rate={pos_rate_va:.6g}")

    # --- matrices ---
    Xtr = df_tr[feature_cols].to_numpy(dtype=np.float32, copy=False)
    ytr = df_tr[label_col].to_numpy(dtype=np.int32, copy=False)
    Xva = df_va[feature_cols].to_numpy(dtype=np.float32, copy=False)
    yva = df_va[label_col].to_numpy(dtype=np.int32, copy=False)

    overrides = parse_kv_list(args.xgb_param) if args.xgb_param else None

    booster, train_info = train_xgb(
        Xtr, ytr, Xva, yva,
        feature_names=feature_cols,
        seed=args.seed,
        scale_pos_weight=spw,
        params_override=overrides
    )

    # --- save local model ---
    booster.save_model(out_model_local)
    print(f"[save] model -> {out_model_local}")

    # --- report local ---
    report = {
        "stageb_root": args.stageb_root,
        "columns_json": args.columns_json,
        "artifact_key": args.artifact_key,
        "label_col": label_col,
        "features": feature_cols,
        "train": {
            "n_rows": int(len(df_tr)),
            "pos": pos_tr, "neg": neg_tr, "pos_rate": float(pos_rate_tr),
            "scale_pos_weight": float(spw),
            "n_files": int(len(tr_sel)),
        },
        "val": {
            "n_rows": int(len(df_va)),
            "pos": pos_va, "neg": neg_va, "pos_rate": float(pos_rate_va),
            "n_files": int(len(va_sel)),
        },
        "training": train_info,
        "best_iteration": train_info.get("best_iteration"),
        "timestamp": stamp,
        "out_model_local": out_model_local,
        "out_report_local": out_report_local,
        "out_model_s3": out_model_s3,
        "out_report_s3": out_report_s3,
    }

    with open(out_report_local, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[save] report -> {out_report_local}")

    # --- upload model to S3 (optional) ---
    if out_model_s3:
        with open(out_model_local, "rb") as f:
            data = f.read()
        with fs.open(out_model_s3, "wb") as fo:
            fo.write(data)
        print(f"[save] uploaded model -> {out_model_s3}")
    
    # --- upload report to S3 (optional) ---
    if out_report_s3:
        with open(out_report_local, "rb") as f:
            data = f.read()
        with fs.open(out_report_s3, "wb") as fo:
            fo.write(data)
        print(f"[save] uploaded report -> {out_report_s3}")

    print("[done] baseline training OK")

if __name__ == "__main__":
    main()