#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StageB live gate (prod-ready):
- Non-directional gating: compute p = model(features), s = p - p_thr_ev0, allow = (s >= thr_s)
- model_uri can be local path or s3://...
- columns_json provides feature list (feature_cols_xgb or feature_cols_all)
- "feed" in production = how you compute features + p_thr_ev0 from live data.
  This file stays the same.

Usage examples:
  # 1) Just import and use in production:
    from ml_bot.live.stageB_gate import StageBGate, StageBGateConfig

    cfg = StageBGateConfig(
        model_uri="s3://.../baseline_xgb_stageB_YYYYMMDD-HHMMSS.json",
        columns_json="s3://.../dataset=v1/_meta/columns.json",
        thr_s=0.3977672740,
    )
    gate = StageBGate(cfg)
    allow, out = gate.decide(feature_row_dict, p_thr_ev0=pthr)

  # 2) CLI replay/debug on parquet split:
    python -m ml_bot.live.stageB_gate \
      --model-uri s3://.../baseline_xgb_stageB_20260203-100728.json \
      --columns-json s3://.../dataset=v1/_meta/columns.json \
      --stageb-root s3://.../dataset=v1 \
      --split val --n-files 2 --max-rows 20000 --thr-s 0.3977672740
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import s3fs
import xgboost as xgb


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class StageBGateConfig:
    model_uri: str
    columns_json: str
    thr_s: float

    # If you want to fail hard when any feature is missing:
    strict_features: bool = True

    # If strict_features=False, missing features are filled with 0.0 (NOT recommended unless you know why)
    fill_missing_value: float = 0.0

    # Optional: if you want clipping paranoia
    clip_proba_01: bool = True


# -----------------------------
# Utilities (S3 / JSON / Model)
# -----------------------------

def _is_s3(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("s3://")


def _read_json(fs: s3fs.S3FileSystem, uri: str) -> dict:
    if _is_s3(uri):
        with fs.open(uri, "rb") as f:
            return json.load(f)
    with open(uri, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_booster(fs: s3fs.S3FileSystem, model_uri: str) -> xgb.Booster:
    booster = xgb.Booster()
    if _is_s3(model_uri):
        with fs.open(model_uri, "rb") as f:
            raw = f.read()
        # xgb can load from bytes via bytearray
        booster.load_model(bytearray(raw))
    else:
        booster.load_model(model_uri)
    return booster


def _get_feature_cols(columns_meta: dict) -> List[str]:
    feats = columns_meta.get("feature_cols_xgb") or columns_meta.get("feature_cols_all")
    if not isinstance(feats, list) or not feats:
        raise RuntimeError("columns.json missing feature_cols_xgb/feature_cols_all")
    # normalize to str
    return [str(c) for c in feats]


def _ensure_float01(x: Union[float, np.ndarray], name: str) -> Union[float, np.ndarray]:
    if isinstance(x, np.ndarray):
        if not np.isfinite(x).all():
            raise RuntimeError(f"{name} has NaN/inf")
        return x
    if not np.isfinite(float(x)):
        raise RuntimeError(f"{name} is NaN/inf")
    return float(x)


# -----------------------------
# Core Gate
# -----------------------------

class StageBGate:
    """
    Production gate:
      p = model.predict(features)
      s = p - p_thr_ev0
      allow = (s >= thr_s)

    IMPORTANT:
    - This class does NOT compute features nor p_thr_ev0.
      Your production "feed" must supply them.
    """

    def __init__(self, cfg: StageBGateConfig, fs: Optional[s3fs.S3FileSystem] = None):
        self.cfg = cfg
        self.fs = fs or s3fs.S3FileSystem()

        # load contract
        meta = _read_json(self.fs, cfg.columns_json)
        self.feature_cols = _get_feature_cols(meta)

        # load model
        self.booster = _load_booster(self.fs, cfg.model_uri)

        # sanity: feature count
        n_model = int(self.booster.num_features())
        if n_model != len(self.feature_cols):
            msg = (
                f"Feature count mismatch: model expects {n_model}, "
                f"columns.json has {len(self.feature_cols)}.\n"
                f"model_uri={cfg.model_uri}\ncolumns_json={cfg.columns_json}"
            )
            # In production, this should be a hard fail.
            raise RuntimeError(msg)

        self.thr_s = float(cfg.thr_s)
        if not np.isfinite(self.thr_s):
            raise RuntimeError("thr_s is NaN/inf")

    def _row_to_vector(self, row: Union[Dict[str, Any], pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        """
        Convert a single row of features to a (1, n_features) float32 array in correct order.
        Accepts:
          - dict/Series keyed by feature name
          - numpy array/list already ordered (must match length exactly)
        """
        if isinstance(row, (np.ndarray, list, tuple)):
            arr = np.asarray(row, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != len(self.feature_cols):
                raise RuntimeError(
                    f"Row array must be shape (n_features,), got {arr.shape}, expected {len(self.feature_cols)}"
                )
            return arr.reshape(1, -1)

        # dict-like
        if isinstance(row, pd.Series):
            row_dict = row.to_dict()
        elif isinstance(row, dict):
            row_dict = row
        else:
            raise TypeError(f"Unsupported row type: {type(row)}")

        missing = [c for c in self.feature_cols if c not in row_dict]
        if missing and self.cfg.strict_features:
            raise RuntimeError(f"Missing features in feed (first 20): {missing[:20]} (total={len(missing)})")

        # build ordered vector
        vec = np.empty((len(self.feature_cols),), dtype=np.float32)
        for i, c in enumerate(self.feature_cols):
            v = row_dict.get(c, self.cfg.fill_missing_value)
            # robust casting; keep NaN if present
            try:
                vec[i] = np.float32(v)
            except Exception:
                # if weird type -> NaN (better to fail upstream, but avoid crashing here)
                vec[i] = np.nan

        return vec.reshape(1, -1)

    def predict_proba(self, feature_row: Union[Dict[str, Any], pd.Series, np.ndarray, List[float]]) -> float:
        X = self._row_to_vector(feature_row)
        dm = xgb.DMatrix(X, feature_names=self.feature_cols, missing=np.nan)
        p = float(self.booster.predict(dm, output_margin=False)[0])

        if self.cfg.clip_proba_01:
            # only clip tiny numerical drifts; still fail if totally out of range
            if p < -1e-6 or p > 1.0 + 1e-6:
                raise RuntimeError(f"Predicted proba out of [0,1]: p={p}")
            p = float(np.clip(p, 0.0, 1.0))

        if not np.isfinite(p):
            raise RuntimeError("Predicted proba is NaN/inf")

        return p

    def decide(
        self,
        feature_row: Union[Dict[str, Any], pd.Series, np.ndarray, List[float]],
        p_thr_ev0: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Returns:
          allow (bool),
          out dict with p, p_thr_ev0, s, thr_s
        """
        p_thr_ev0 = float(_ensure_float01(p_thr_ev0, "p_thr_ev0"))
        p = self.predict_proba(feature_row)
        s = float(p - p_thr_ev0)

        allow = bool(s >= self.thr_s)
        out = {
            "p": float(p),
            "p_thr_ev0": float(p_thr_ev0),
            "s": float(s),
            "thr_s": float(self.thr_s),
        }
        return allow, out


# -----------------------------
# CLI replay / debug
# -----------------------------

def _list_stageb_paths_by_split(fs: s3fs.S3FileSystem, stageb_root: str, split: str) -> List[str]:
    prefix = f"{stageb_root}/split={split}/parquet"
    pattern = prefix.replace("s3://", "") + "/*.parquet"
    paths = [f"s3://{p}" for p in fs.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"No StageB parquet files for split={split} under {prefix}")
    return sorted(paths)


def _sample_paths(paths: List[str], n_files: int, seed: int) -> List[str]:
    if n_files <= 0 or n_files >= len(paths):
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=n_files, replace=False)
    return [paths[i] for i in idx]


def _load_parquet_subset(paths: List[str], cols: List[str], max_rows: Optional[int], seed: int) -> pd.DataFrame:
    dfs = [pd.read_parquet(p, columns=cols, engine="pyarrow") for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    if max_rows is not None and int(max_rows) > 0 and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)
    return df


def _cli():
    ap = argparse.ArgumentParser("StageB live gate (replay/debug)")
    ap.add_argument("--model-uri", default="s3://tradebot-config-tokyo/models/xgb-baseline/stageB/baseline_xgb_stageB_20260203-100728.json")
    ap.add_argument("--columns-json", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/_meta/columns.json")
    ap.add_argument("--thr-s", type=float, default="0.3977672740")

    ap.add_argument("--stageb-root", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1")
    #ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--split", default="train")
    
    ap.add_argument("--n-files", type=int, default=0)    #remettre 5 apres 
    ap.add_argument("--max-rows", type=int, default=500000)    #remettre 200000 apres 
    ap.add_argument("--seed", type=int, default=42)

    # required columns in parquet for replay
    ap.add_argument("--audit-pthr-col", default="audit_p_thr_ev0")
    ap.add_argument("--label-col", default="label_A")  # not used for gating but useful to print stats

    args = ap.parse_args()

    fs = s3fs.S3FileSystem()

    cfg = StageBGateConfig(
        model_uri=args.model_uri,
        columns_json=args.columns_json,
        thr_s=float(args.thr_s),
        strict_features=True,
    )
    gate = StageBGate(cfg, fs=fs)

    paths_all = _list_stageb_paths_by_split(fs, args.stageb_root, args.split)
    paths = _sample_paths(paths_all, int(args.n_files), int(args.seed))

    cols = [args.audit_pthr_col, args.label_col] + gate.feature_cols
    df = _load_parquet_subset(paths, cols=cols, max_rows=int(args.max_rows), seed=int(args.seed))

    # compute decisions
    pthr = df[args.audit_pthr_col].to_numpy(np.float64, copy=False)

    X = df[gate.feature_cols].to_numpy(np.float32, copy=False)
    dm = xgb.DMatrix(X, feature_names=gate.feature_cols, missing=np.nan)
    p = gate.booster.predict(dm, output_margin=False).astype(np.float64)

    s = p - pthr
    allow = (s >= gate.thr_s)

    # summarize
    out = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "split": args.split,
        "n_rows": int(len(df)),
        "n_allow": int(np.sum(allow)),
        "frac_allow": float(np.mean(allow)),
        "s_stats": {
            "mean": float(np.mean(s)),
            "p05": float(np.quantile(s, 0.05)),
            "p50": float(np.quantile(s, 0.50)),
            "p95": float(np.quantile(s, 0.95)),
            "p99": float(np.quantile(s, 0.99)),
        },
        "margin_mean_cond_allow": float(np.mean(s[allow])) if np.any(allow) else float("nan"),
        "p_stats": {
            "mean": float(np.mean(p)),
            "p95": float(np.quantile(p, 0.95)),
            "p99": float(np.quantile(p, 0.99)),
            "max": float(np.max(p)),
        },
        "pthr_stats": {
            "mean": float(np.mean(pthr)),
            "p95": float(np.quantile(pthr, 0.95)),
            "p99": float(np.quantile(pthr, 0.99)),
            "max": float(np.max(pthr)),
        },
        "thr_s": float(gate.thr_s),
        "model_uri": args.model_uri,
        "columns_json": args.columns_json,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    # if run as module: python -m ml_bot.live.stageB_gate ...
    _cli()


if __name__ == "__main__":
    main()