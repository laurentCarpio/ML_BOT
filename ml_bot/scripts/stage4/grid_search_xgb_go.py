# ml_bot/scripts/stage4/grid_search_xgb_go.py
import json
import time
import tarfile
import tempfile
import io, re
import sys
import traceback
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import boto3
from botocore.config import Config as BotoConfig
import numpy as np
import pandas as pd
import argparse
import xgboost as xgb
import s3fs
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve as image_uri_retrieve
from sagemaker.estimator import Estimator

# ========= PARAMS =========
REGION    = "ap-northeast-1"
ROLE_ARN  = "arn:aws:iam::174175447862:role/AmazonSageMaker-ExecutionRole"

VERSION = "go-v1"
DATA_ROOT     = "s3://tradebot-config-tokyo/data/stage3/go"
MODEL_BASE_S3 = f"s3://tradebot-config-tokyo/models/xgb-{VERSION}"

TRN_DIR = f"{DATA_ROOT}/train"
VAL_DIR = f"{DATA_ROOT}/val"
TST_DIR = f"{DATA_ROOT}/test"

TRW_DIR = f"{DATA_ROOT}/train_weight"
VAW_DIR = f"{DATA_ROOT}/val_weight"

META_DIR = f"{DATA_ROOT}/_meta"

INSTANCE_TYPE   = "ml.c5.xlarge"
SPOT            = True
MAX_RUN_SEC     = 60 * 20
MAX_WAIT_SEC    = 60 * 40

_AUDIT_COLS_RE = re.compile(r"columns\s*:\s*([^()]+)", re.IGNORECASE)

# =========================
# Stage4 selection policy:
#   - VAL-only selection of:
#       * decision threshold thr
#       * (tp_R, sl_R, side) combo using new audit_* fields
#   - hard gates always:
#       allowed = (p>=thr) & (p>=budget_thr(tp,sl,cost_R)) & (flags==0)
#   - objective for selection:
#       maximize EV_R_net = sum(simulated_netR[allowed])
#
# Notes:
#   We assume Stage3 now writes audit_first_touch_{L,S} such that:
#     >0 means first touch was profit barrier with magnitude in R (ex: 1..5)
#     <0 means first touch was stop barrier with magnitude in R (ex: -1..-3)
#      0 means no barrier touched before exit.
#   If your encoding differs, adjust _tp_sl_value_from_first_touch().
# =========================

HP_BASE: Dict[str, Any] = dict(
    tree_method="hist",
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_round=600,
    early_stopping_rounds=80,
    objective="binary:logistic",
    eval_metric="logloss",
    reg_lambda=15.0,
    reg_alpha=1.0,
    gamma=2.0,
    max_depth=3,
    min_child_weight=30,
    max_delta_step=1,
)

# SageMaker built-in XGBoost container
XGB_IMAGE = image_uri_retrieve("xgboost", REGION, version="1.7-1")

# ========= AUDIT expected fields =========
# Core used for gating + baseline EV
AUDIT_REQUIRED_CORE = [
    "audit_pnl_net_bps",
    "audit_early_abort",
    "audit_timeout",
    "audit_market_toxic",
    "audit_p_thr_ev0",
]

AUDIT_REQUIRED_STAGE4 = [
    "audit_cost_R",
    "audit_cost_bps",
    "audit_R_bps",
    "audit_rr_min",
    "audit_exit_reason_code",
]

# Extra diagnostics (optional, but if present we summarize them)
AUDIT_DIAG_OPTIONAL = [
    "audit_mfe_R_L", "audit_mae_R_L",
    "audit_mfe_R_S", "audit_mae_R_S",
    "audit_first_touch_step_L",
    "audit_first_touch_step_S",
    "audit_hit_p1R_L", "audit_hit_p1R_S",
    "audit_hit_p2R_L", "audit_hit_p2R_S",
    "audit_hit_p3R_L", "audit_hit_p3R_S",
    "audit_hit_p4R_L", "audit_hit_p4R_S",
    "audit_hit_p5R_L", "audit_hit_p5R_S",
    "audit_hit_m1R_L", "audit_hit_m1R_S",
    "audit_hit_m2R_L", "audit_hit_m2R_S",
    "audit_hit_m3R_L", "audit_hit_m3R_S",
]

# ========= HELPERS =========

def _tp_sl_value_netR_from_hit_steps(
    aud: Dict[str, np.ndarray],
    side: str,
    tp_r: int,
    sl_r: int,
    cost_R: np.ndarray,
) -> np.ndarray:
    t_tp = np.asarray(aud[f"audit_hit_step_p{tp_r}R_{side}"], dtype=np.int64)
    t_sl = np.asarray(aud[f"audit_hit_step_m{sl_r}R_{side}"], dtype=np.int64)

    if np.any(t_tp < -1) or np.any(t_sl < -1):
        raise RuntimeError(f"Bad hit steps (<-1) for side={side} tp={tp_r} sl={sl_r}")

    cR = np.asarray(cost_R, dtype=np.float64)
    if cR.shape[0] != t_tp.shape[0]:
        raise RuntimeError("cost_R length mismatch vs audit steps")

    out = np.zeros_like(cR, dtype=np.float64)

    tp_hit = (t_tp >= 0)
    sl_hit = (t_sl >= 0)

    tp_wins = tp_hit & (~sl_hit | (t_tp < t_sl))
    sl_wins = sl_hit & (~tp_hit | (t_sl < t_tp))

    # tie => SL (conservative)
    ties = tp_hit & sl_hit & (t_tp == t_sl)

    # coût payé uniquement si une barrière est touchée (TP/SL/tie)
    touched = tp_wins | sl_wins | ties

    out[tp_wins] = float(tp_r)
    out[sl_wins] = -float(sl_r)
    out[ties]    = -float(sl_r)

    out[touched] = out[touched] - cR[touched]
    return out

def _p_thr_ev0_from_costR(cost_R: np.ndarray, tp_r: int, sl_r: int) -> np.ndarray:
    cost_R = np.asarray(cost_R, dtype=np.float64)
    denom = float(tp_r + sl_r)
    thr = (float(sl_r) + cost_R) / max(denom, 1e-12)
    thr = np.clip(thr, 0.0, 1.0)
    # NaN/inf -> 1.0 (super conservateur, bloque)
    thr[~np.isfinite(thr)] = 1.0
    return thr

def _required_hit_step_cols(tp_max: int, sl_max: int) -> List[str]:
    cols = []
    for side in ["L", "S"]:
        for k in range(1, tp_max + 1):
            cols.append(f"audit_hit_step_p{k}R_{side}")
        for k in range(1, sl_max + 1):
            cols.append(f"audit_hit_step_m{k}R_{side}")
    return cols

def _read_json_s3(fs: s3fs.S3FileSystem, uri: str) -> dict:
    with fs.open(uri, "rb") as f:
        return json.load(f)

def _load_meta_contract(fs: s3fs.S3FileSystem, meta_dir: str) -> Tuple[str, List[str], List[str], Dict[str, str]]:
    """
    Fail-hard:
      - columns.json doit contenir label/features
      - + audit_format avec "columns: a,b,c"
      - + audit_paths dict avec train_audit/val_audit/test_audit
    """
    cols_path = f"{meta_dir}/columns.json"
    meta = _read_json_s3(fs, cols_path)

    label = str(meta.get("label", ""))
    feats = meta.get("features") or meta.get("feature_cols") or []
    if not label or not isinstance(feats, list) or not feats:
        raise RuntimeError(f"columns.json invalide: label/features manquants ({cols_path})")

    audit_format = meta.get("audit_format", "")
    if not isinstance(audit_format, str) or not audit_format.strip():
        raise RuntimeError(f"columns.json invalide: audit_format manquant ({cols_path})")

    m = _AUDIT_COLS_RE.search(audit_format)
    if not m:
        raise RuntimeError(f"audit_format invalide: attendu 'columns: ...' | got={audit_format}")

    audit_cols = [c.strip() for c in m.group(1).split(",") if c.strip()]
    if not audit_cols:
        raise RuntimeError(f"audit_format invalide: aucune colonne parsée | got={audit_format}")

    audit_paths = meta.get("audit_paths", None)
    if not isinstance(audit_paths, dict):
        raise RuntimeError(f"columns.json invalide: audit_paths manquant ou non-dict ({cols_path})")

    for k in ["train_audit", "val_audit", "test_audit"]:
        if k not in audit_paths or not isinstance(audit_paths[k], str) or not audit_paths[k].strip():
            raise RuntimeError(f"columns.json invalide: audit_paths['{k}'] manquant/mauvais")
        if "part-*.csv" not in audit_paths[k]:
            raise RuntimeError(f"columns.json invalide: audit_paths['{k}'] doit contenir 'part-*.csv' | got={audit_paths[k]}")

    return label, [str(x) for x in feats], audit_cols, {str(k): str(v) for k, v in audit_paths.items()}

def _check_flags_binary(aud: Dict[str, np.ndarray], keys: List[str], where: str):
    for k in keys:
        if k not in aud:
            raise RuntimeError(f"[{where}] missing audit col: {k}")
        v = aud[k].astype(np.int64, copy=False)
        if not np.isin(v, [0, 1]).all():
            raise RuntimeError(f"[{where}] {k} must be 0/1.")

def _assert_ev_inputs_finite(aud: Dict[str, np.ndarray], where: str, tp_max: int, sl_max: int):
    keys = [
        "audit_p_thr_ev0",
        "audit_market_toxic",
        "audit_timeout",
        "audit_early_abort",
    ] + _required_hit_step_cols(tp_max=tp_max, sl_max=sl_max)

    for k in keys:
        v = np.asarray(aud[k], dtype=np.float64)
        if not np.isfinite(v).all():
            raise RuntimeError(f"[{where}] {k} has NaN/inf (EV path unsafe)")
                
def _validate_side_audit_contract(aud: Dict[str, np.ndarray], where: str):
    # hit flags doivent être 0/1 (EXCLURE hit_step_*)
    hit_cols = [k for k in aud.keys() if k.startswith("audit_hit_") and not k.startswith("audit_hit_step_")]
    for k in hit_cols:
        v = np.asarray(aud[k], dtype=np.float64)
        if not np.isfinite(v).all():
            raise RuntimeError(f"[{where}] {k} has NaN/inf")
        if np.any((v < -1e-6) | (v > 1.0 + 1e-6)):
            raise RuntimeError(f"[{where}] {k} out of [0,1]")

    # (optionnel) check steps
    step_cols = [k for k in aud.keys() if k.startswith("audit_hit_step_")]
    for k in step_cols:
        v = np.asarray(aud[k], dtype=np.float64)
        if not np.isfinite(v).all():
            raise RuntimeError(f"[{where}] {k} has NaN/inf")
        if np.any(v < -1.0 - 1e-6):
            raise RuntimeError(f"[{where}] {k} has values < -1 (bad step encoding)")
        frac = np.abs(v - np.round(v))
        if np.nanmax(frac) > 1e-6:
            raise RuntimeError(f"[{where}] {k} not integer-like (max frac={np.nanmax(frac)})")

    # first_touch optional checks (si présents)
    for side in ["L", "S"]:
        k_val  = f"audit_first_touch_{side}"
        k_step = f"audit_first_touch_step_{side}"
        if k_val in aud and k_step in aud:
            ft = np.asarray(aud[k_val], dtype=np.float64)
            fs = np.asarray(aud[k_step], dtype=np.int64)
            
            # step=-1 => ft==0
            bad1 = (fs < 0) & (ft != 0)
            if np.any(bad1):
                raise RuntimeError(f"[{where}] incoherent first_touch vs step for {side}: step=-1 but touch!=0")
           
            # step>=0 => ft!=0 (optionnel, tu peux le rendre warning)
            bad2 = (fs >= 0) & (ft == 0)
            if np.any(bad2):
                print(f"[WARN][{where}] incoherent first_touch vs step for {side}: step>=0 but touch==0 on some rows")

def parse_args():
    ap = argparse.ArgumentParser(
        "Stage4 grid search + VAL-only threshold + TP/SL sweep maximizing EV_R with hard gates."
    )
    ap.add_argument("--mode", choices=["train", "eval"], default="train",
                    help="train = lance des jobs SageMaker, eval = évalue un model_uri existant (sans retrain).")
    ap.add_argument("--model-uri", default="",
                    help="Requis en mode eval: s3://.../model.tar.gz")
    ap.add_argument("--allow-feature-mismatch", action="store_true", default=False)
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Epsilon pour tie-break / seuils.")
    # TP/SL sweep ranges
    ap.add_argument("--tp-max", type=int, default=3, help="Max TP in R (inclusive).")
    ap.add_argument("--sl-max", type=int, default=2, help="Max SL in R (inclusive).")

    # A) Serialize candidates for audit/debug (optional)
    ap.add_argument("--emit-candidates", action="store_true", default=False,
                    help="Sérialise candidates_topk (best TP/SL combos) dans le summary JSON.")
    ap.add_argument("--candidates-topk", type=int, default=10,
                    help="Nombre de candidates à sérialiser si --emit-candidates.")

    # D2) Risk-adjusted selection on VAL
    ap.add_argument("--risk-z", type=float, default=2.5,
                    help="Z pour score risk-adjusted: mean(netR) - z*std/sqrt(n). Ex: 1.0 ou 1.64.")
    ap.add_argument("--min-allowed", type=int, default=300,                     
                    help="Ignore thr/TP/SL avec moins de min_allowed trades (VAL) pour éviter le sur-fit.")
    
    ap.add_argument("--max-mae-over-mfe", type=float, default=0.90,
                    help="Hard gate: require mae <= ratio*mfe on the chosen side (VAL selection + reporting).")
    ap.add_argument("--min-mfe-R", type=float, default=0.25,
                    help="Hard gate: require mfe >= min_mfe_R (avoid unstable ratios when mfe ~ 0).")
    
    ap.add_argument("--min-ev-score", type=float, default=-1e9,
                help="Ignore thresholds with EV_score < min_ev_score.")
    
    return ap.parse_args()

def _merge_hps(base: dict, overlay: dict):
    h = base.copy()
    h.update(overlay)
    for k in ["max_depth", "min_child_weight", "max_delta_step"]:
        if k in h: h[k] = int(h[k])
    for k in ["reg_lambda", "reg_alpha", "gamma", "subsample", "colsample_bytree", "eta"]:
        if k in h: h[k] = float(h[k])
    if "num_round" in h: h["num_round"] = int(h["num_round"])
    if "early_stopping_rounds" in h: h["early_stopping_rounds"] = int(h["early_stopping_rounds"])
    return h

def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _json_dumps_safe(payload: dict) -> bytes:
    return json.dumps(_json_sanitize(payload), indent=2, ensure_ascii=False).encode("utf-8")

def _boto3_client(service, region=REGION):
    return boto3.client(
        service,
        region_name=region,
        config=BotoConfig(
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
        ),
    )

def _s3_put_bytes_uri(s3_client, s3_uri: str, data: bytes):
    bkt = s3_uri.split("/", 3)[2]
    key = s3_uri.split("/", 3)[3]
    s3_client.put_object(Bucket=bkt, Key=key, Body=data)

def _require_csv_prefix(fs: s3fs.S3FileSystem, prefix: str, label: str):
    paths = [f"s3://{p}" for p in fs.glob(prefix.replace("s3://","") + "/part-*.csv")]
    if not paths:
        raise FileNotFoundError(f"[DATA MISSING] Aucun CSV sous {prefix}/part-*.csv ({label})")
    return paths

def make_channels():
    return {
        "train": TrainingInput(TRN_DIR, content_type="text/csv", input_mode="File"),
        "train_weight": TrainingInput(TRW_DIR, content_type="text/csv", input_mode="File"),
        "validation": TrainingInput(VAL_DIR, content_type="text/csv", input_mode="File"),
        "validation_weight": TrainingInput(VAW_DIR, content_type="text/csv", input_mode="File"),
        "meta": TrainingInput(META_DIR, content_type="application/x-directory", input_mode="File"),
    }

def submit_job(hps: dict, session: sagemaker.Session) -> Estimator:
    src_dir = Path(__file__).parent
    entry = "train.py"
    if not (src_dir / entry).is_file():
        raise FileNotFoundError(f"train.py introuvable dans {src_dir}")

    est = Estimator(
        image_uri=XGB_IMAGE,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        hyperparameters=hps,
        output_path=MODEL_BASE_S3,
        sagemaker_session=session,
        enable_sagemaker_metrics=True,
        max_run=MAX_RUN_SEC,
        use_spot_instances=SPOT,
        max_wait=MAX_WAIT_SEC if SPOT else None,
        entry_point=entry,
        source_dir=str(src_dir),
        base_job_name="xgb-go",
    )

    job_name = f"xgb-go-{time.strftime('%Y%m%d-%H%M%S')}"
    print("Submitting job:", job_name, "| hps:", hps)
    est.fit(make_channels(), job_name=job_name, logs=True)
    return est

def _load_csv_dir(fs: s3fs.S3FileSystem, prefix: str) -> pd.DataFrame:
    paths = [f"s3://{p}" for p in fs.glob(prefix.replace("s3://","") + "/part-*.csv")]
    if not paths:
        raise FileNotFoundError(f"Aucun CSV trouvé sous {prefix}/part-*.csv")
    dfs = [pd.read_csv(p, header=None) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def _load_xy_by_columns(fs: s3fs.S3FileSystem, prefix: str, feature_names: List[str]):
    df = _load_csv_dir(fs, prefix)
    y = df.iloc[:, 0].astype(np.int32).to_numpy()
    n_feat = len(feature_names)
    if df.shape[1] != 1 + n_feat:
        raise RuntimeError(f"[{prefix}] bad X shape: got {df.shape[1]} expected {1+n_feat} (exact).")
    X = df.iloc[:, 1:1+n_feat].astype(np.float32).to_numpy()
    return X, y

def _load_audit_contract(fs: s3fs.S3FileSystem, audit_prefix: str, audit_cols: List[str]) -> Dict[str, np.ndarray]:
    df = _load_csv_dir(fs, audit_prefix)
    if int(df.shape[1]) != int(len(audit_cols)):
        raise RuntimeError(
            f"[{audit_prefix}] AUDIT shape mismatch: got {df.shape[1]} cols, expected {len(audit_cols)} cols={audit_cols}"
        )
    out: Dict[str, np.ndarray] = {}
    for j, name in enumerate(audit_cols):
        out[name] = df.iloc[:, j].astype(np.float64).to_numpy()
    return out

def _check_thr01(thr: np.ndarray, where: str, eps: float = 1e-12):
    thr = np.asarray(thr, dtype=np.float64)
    if not np.isfinite(thr).all():
        raise RuntimeError(f"[{where}] threshold contains NaN/inf.")
    if np.any(thr < -eps) or np.any(thr > 1.0 + eps):
        raise RuntimeError(f"[{where}] threshold out of [0,1].")

def _load_booster_from_tar(fs: s3fs.S3FileSystem, model_uri: str) -> xgb.Booster:
    with fs.open(model_uri, "rb") as f:
        raw = f.read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        member = None
        for m in tar.getmembers():
            if m.name.endswith(("model.json", "xgboost-model", "model.bin")):
                member = m
                break
        if member is None:
            raise RuntimeError("model.json / xgboost-model introuvable dans model.tar.gz")
        model_bytes = tar.extractfile(member).read()

    booster = xgb.Booster()
    try:
        booster.load_model(io.BytesIO(model_bytes))
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp.flush()
            booster.load_model(tmp.name)
    return booster

def _align_X_to_model(X: np.ndarray, n_model: int) -> Tuple[np.ndarray, dict]:
    """Permissif: ajuste X au nombre de features du modèle (truncate/pad)."""
    n = int(X.shape[1])
    if n == n_model:
        return X, {"action": "none", "before": n, "after": n}
    if n > n_model:
        return X[:, :n_model], {"action": "truncate", "before": n, "after": n_model, "dropped": n - n_model}
    pad = np.zeros((X.shape[0], n_model - n), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1), {"action": "pad", "before": n, "after": n_model, "added": n_model - n}

def _require_exact_feature_count(X: np.ndarray, n_model: int, where: str) -> dict:
    n = int(X.shape[1])
    if n != n_model:
        raise RuntimeError(f"[{where}] feature mismatch: data has {n} cols, model expects {n_model}. Refuse.")
    return {"action": "none", "before": n, "after": n}

# ========= TP/SL simulation =========

def _check_required_audit_cols(aud: Dict[str, np.ndarray], cols: List[str], where: str):
    missing = [c for c in cols if c not in aud]
    if missing:
        raise RuntimeError(f"[{where}] audit missing required cols: {missing}")

def _best_threshold_by_value(
    y: np.ndarray,
    p: np.ndarray,
    value: np.ndarray,
    audit_market_toxic: np.ndarray,
    audit_timeout: np.ndarray,
    audit_early_abort: np.ndarray,
    budget_thr: np.ndarray,
    eps: float,
    risk_z: float,
    min_allowed: int,
    min_ev_score: float,
    extra_gate: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Choose a decision threshold `thr` that maximizes a risk-adjusted score under hard gates.

    Implementation detail (important):
      - We sort samples by predicted probability `p` in descending order.
      - Each candidate threshold `thr` is taken as one of the unique values of `p`.
      - For a given `thr`, the predicted-positive set is the PREFIX of the sorted list:
            pred(thr) = { i : p_i >= thr }   (i.e., indices 0..end in sorted order)

    Hard gates (applied only within the predicted-positive prefix):
      - ok_flags  = (audit_market_toxic==0) & (audit_timeout==0) & (audit_early_abort==0)
      - ok_budget = (p >= budget_thr)
      - allowed(thr) = pred(thr) & ok_flags & ok_budget

     Objective (robust):
       - Compute netR stats on allowed(thr):
           EV_sum = sum(netR_allowed)
           mean   = mean(netR_allowed)
           std    = std(netR_allowed)
           score  = mean - risk_z * std / sqrt(n_allowed)
       - Maximize score (not EV_sum).

    Tie-break (in order):
      1) higher score
      2) higher TP (among allowed(thr))
      3) lower FP (among allowed(thr))
      4) higher thr
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)

    mt = np.asarray(audit_market_toxic, dtype=np.int64)
    to = np.asarray(audit_timeout, dtype=np.int64)
    ab = np.asarray(audit_early_abort, dtype=np.int64)
    thr0 = np.asarray(budget_thr, dtype=np.float64)

    if thr0.shape[0] != p.shape[0]:
        raise RuntimeError(f"[best_threshold] length mismatch: p={p.shape[0]} budget_thr={thr0.shape[0]}")

    n = int(y.shape[0])
    if n == 0:
        return {"decision_threshold": 1.0, "chosen_end": -1, "why": "empty", "EV": 0.0, "TP": 0, "FP": 0}

    _check_flags_binary(
        {"audit_market_toxic": mt, "audit_timeout": to, "audit_early_abort": ab},
        ["audit_market_toxic", "audit_timeout", "audit_early_abort"],
        "VALUE/flags",
    )
    
    _check_thr01(thr0, "VALUE/budget_thr", eps=eps)

    order = np.argsort(-p, kind="mergesort")
    ps = p[order]; ys = y[order]; vv = v[order]
    mt = mt[order]; to = to[order]; ab = ab[order]; th = thr0[order]

    tp_mask = (ys == 1)
    fp_mask = (ys == 0)
    ok_flags = (mt == 0) & (to == 0) & (ab == 0)
    ok_budget = (ps >= th)
    allowed = ok_flags & ok_budget
    if extra_gate is not None:
        allowed = allowed & np.asarray(extra_gate, dtype=bool)[order]

    pred_cum = np.arange(n, dtype=np.int64) + 1
    allowed_cum = np.cumsum(allowed.astype(np.int64))
    blocked_flags_cum  = np.cumsum((~ok_flags).astype(np.int64))
    blocked_budget_cum = np.cumsum((ok_flags & (~ok_budget)).astype(np.int64))

    EV_cum = np.cumsum(vv * allowed)
    TP_cum = np.cumsum((tp_mask & allowed).astype(np.int64))
    FP_cum = np.cumsum((fp_mask & allowed).astype(np.int64))

    ends = np.array([0], dtype=np.int64) if n == 1 else np.r_[np.where(ps[1:] != ps[:-1])[0], n - 1].astype(np.int64)

    # risk-adjusted running moments on allowed vv
    vv_allowed = vv * allowed.astype(np.float64)
    sum_v  = np.cumsum(vv_allowed)
    sum_v2 = np.cumsum((vv * vv) * allowed.astype(np.float64))

    best = None
    for end in ends:
        n_allowed = int(allowed_cum[end])
        if n_allowed < int(min_allowed):
            continue

        ev_sum = float(sum_v[end])
        mean = ev_sum / max(n_allowed, 1)
        ex2  = float(sum_v2[end]) / max(n_allowed, 1)
        var  = max(ex2 - mean * mean, 0.0)
        std  = math.sqrt(var)
        score = mean - float(risk_z) * std / math.sqrt(max(n_allowed, 1))

        if score < float(min_ev_score):
            continue

        cand = {
            "end": int(end),
            "thr": float(ps[end]),
            "EV": ev_sum,              # kept for backward compatibility
            "EV_sum": ev_sum,
            "EV_mean": float(mean),
            "EV_std": float(std),
            "EV_score": float(score),
            "TP": int(TP_cum[end]),
            "FP": int(FP_cum[end]),
            "n_pred": int(pred_cum[end]),
            "n_allowed": n_allowed,
            "n_blocked_flags": int(blocked_flags_cum[end]),
            "n_blocked_budget": int(blocked_budget_cum[end]),
        }
        if best is None:
            best = cand
        else:
            if (
                cand["EV_score"] > best["EV_score"] + eps or
                (abs(cand["EV_score"] - best["EV_score"]) <= eps and cand["TP"] > best["TP"]) or
                (abs(cand["EV_score"] - best["EV_score"]) <= eps and cand["TP"] == best["TP"] and cand["FP"] < best["FP"]) or
                (abs(cand["EV_score"] - best["EV_score"]) <= eps and cand["TP"] == best["TP"] and cand["FP"] == best["FP"] and cand["thr"] > best["thr"])
            ):
                best = cand

    if best is None:
        return {
            "decision_threshold": 1.0, "chosen_end": -1, "why": "no candidates (all filtered by min_allowed?)",
            "EV": 0.0, "EV_sum": 0.0, "EV_mean": 0.0, "EV_std": 0.0, "EV_score": -1e30,
            "TP": 0, "FP": 0, "n_pred": 0, "n_allowed": 0, "n_blocked_flags": 0, "n_blocked_budget": 0
        }

    return {
        "decision_threshold": float(best["thr"]),
        "chosen_end": int(best["end"]),
        "EV": float(best["EV"]),             # backward compatibility
        "EV_sum": float(best["EV_sum"]),
        "EV_mean": float(best["EV_mean"]),
        "EV_std": float(best["EV_std"]),
        "EV_score": float(best["EV_score"]),
        "TP": int(best["TP"]),
        "FP": int(best["FP"]),
        "n_pred": int(best["n_pred"]),
        "n_allowed": int(best["n_allowed"]),
        "n_blocked_flags": int(best["n_blocked_flags"]),
        "n_blocked_budget": int(best["n_blocked_budget"]),
        "why": "ok",
    }

def _eval_gated_metrics(
    y: np.ndarray,
    p: np.ndarray,
    thr: float,
    audit_market_toxic: np.ndarray,
    audit_timeout: np.ndarray,
    audit_early_abort: np.ndarray,
    budget_thr: np.ndarray,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    pred_pos = (p >= float(thr))

    mt = np.asarray(audit_market_toxic, dtype=np.int64)
    to = np.asarray(audit_timeout, dtype=np.int64)
    ab = np.asarray(audit_early_abort, dtype=np.int64)
    th = np.asarray(budget_thr, dtype=np.float64)
    if th.shape[0] != p.shape[0]:
        raise RuntimeError(f"[gated_metrics] length mismatch: p={p.shape[0]} budget_thr={th.shape[0]}")

    ok_flags = (mt == 0) & (to == 0) & (ab == 0)
    ok_budget = (p >= th)
    allowed = pred_pos & ok_flags & ok_budget

    tp = int(np.sum((y == 1) & allowed))
    fp = int(np.sum((y == 0) & allowed))
    tn = int(np.sum((y == 0) & (~allowed)))
    fn = int(np.sum((y == 1) & (~allowed)))

    n_neg = max(int(np.sum(y == 0)), 1)
    n_pos = max(int(np.sum(y == 1)), 1)

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "FPR": float(fp / n_neg),
        "TPR": float(tp / n_pos),
        "precision": float(tp / max(tp + fp, 1)),
        "n_allowed": int(np.sum(allowed)),
        "n_pred": int(np.sum(pred_pos)),
        "n_blocked_flags": int(np.sum(pred_pos & (~ok_flags))),
        "n_blocked_budget": int(np.sum(pred_pos & ok_flags & (~ok_budget))),
        "allowed_mask_sum": int(np.sum(allowed)),
    }

def _summarize_extra_audit(aud: Dict[str, np.ndarray], allowed_mask: np.ndarray) -> Dict[str, Any]:
    """
    Small diagnostics to help compare models (on allowed rows):
      - mean/quantiles of mfe/mae (if present)
      - first_touch value + first_touch_step stats (if present)
      - hit rates (if present)
    """
    out: Dict[str, Any] = {}
    m = np.asarray(allowed_mask, dtype=bool)

    if m.size == 0 or int(m.sum()) == 0:
        return {"note": "no allowed rows -> no extra audit summary"}

    def _q(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
        }

    # MFE/MAE (R)
    for k in ["audit_mfe_R_L", "audit_mae_R_L", "audit_mfe_R_S", "audit_mae_R_S"]:
        if k in aud:
            out[k] = _q(np.asarray(aud[k])[m])

    # First-touch value (signed R) + first-touch step
    for side in ["L", "S"]:
        k_val = f"audit_first_touch_{side}"
        k_stp = f"audit_first_touch_step_{side}"

        if k_val in aud:
            ft = np.asarray(aud[k_val], dtype=np.float64)[m]
            ft_f = ft[np.isfinite(ft)]
            base = _q(ft)

            # Fractions on finite values only (avoid NaN poisoning)
            if ft_f.size > 0:
                base.update({
                    "frac_pos":  float(np.mean(ft_f > 0)),
                    "frac_neg":  float(np.mean(ft_f < 0)),
                    "frac_zero": float(np.mean(ft_f == 0)),
                })
            else:
                base.update({"frac_pos": 0.0, "frac_neg": 0.0, "frac_zero": 0.0})

            out[k_val] = base

        if k_stp in aud:
            fs = np.asarray(aud[k_stp], dtype=np.float64)[m]
            out[k_stp] = _q(fs)

            # Optional: share how often "never touched" happens if you use -1 convention
            fs_f = fs[np.isfinite(fs)]
            if fs_f.size > 0:
                out[k_stp]["frac_never"] = float(np.mean(fs_f < 0))

    # Hit flags + hit steps (rates on allowed rows)
    hit_keys = [k for k in AUDIT_DIAG_OPTIONAL if k.startswith("audit_hit_") and k in aud]
    if hit_keys:
        hit_rates: Dict[str, float] = {}
        for k in hit_keys:
            v = np.asarray(aud[k], dtype=np.float64)[m]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            # Treat as 0/1 for flags; for steps this becomes "touched at all" (>=0) if needed.
            # Here we keep the original behavior: >=0.5 => "true" for flags.
            hit_rates[k] = float(np.mean(v >= 0.5))
        out["hit_rates_allowed"] = hit_rates

    return out

# ========= MAIN EVAL =========

def evaluate_ev_sweep(
    model_uri: str,
    val_prefix: str,
    tst_prefix: str,
    allow_feature_mismatch: bool = False,
    eps: float = 1e-12,
    tp_max: int = 5,
    sl_max: int = 3,
    emit_candidates: bool = False,
    candidates_topk: int = 10,
    risk_z: float = 1.0,
    min_allowed: int = 30,
    max_mae_over_mfe: float = 0.90, 
    min_mfe_R: float = 0.25,
    min_ev_score: float = -1e9,
) -> dict:
    fs = s3fs.S3FileSystem()

    booster = _load_booster_from_tar(fs, model_uri)
    label_name, feature_names, audit_cols, audit_paths = _load_meta_contract(fs, META_DIR)

    val_audit_prefix = f"{DATA_ROOT}/{audit_paths['val_audit'].split('/')[0]}"
    tst_audit_prefix = f"{DATA_ROOT}/{audit_paths['test_audit'].split('/')[0]}"

    # FAIL-HARD: audits must exist
    _require_csv_prefix(fs, val_audit_prefix, "val_audit")
    _require_csv_prefix(fs, tst_audit_prefix, "test_audit")

    AUDIT_REQUIRED_TPSL = _required_hit_step_cols(tp_max=int(tp_max), sl_max=int(sl_max))

    Xv, yv = _load_xy_by_columns(fs, val_prefix, feature_names)
    Xt, yt = _load_xy_by_columns(fs, tst_prefix, feature_names)

    for name, yy in [("VAL", yv), ("TEST", yt)]:
        u = np.unique(yy)
        if not np.all(np.isin(u, [0, 1])):
            raise RuntimeError(f"[{name}] y must be in {{0,1}}. got uniques={u[:10]}")

    aud_v = _load_audit_contract(fs, val_audit_prefix, audit_cols)
    aud_t = _load_audit_contract(fs, tst_audit_prefix, audit_cols)
    
    _validate_side_audit_contract(aud_v, "VAL")
    _validate_side_audit_contract(aud_t, "TEST")

    # Fail-hard: core + tp/sl fields must exist (since you said Stage4 must use new audit_*)
    _check_required_audit_cols(aud_v, AUDIT_REQUIRED_CORE, "VAL")
    _check_required_audit_cols(aud_t, AUDIT_REQUIRED_CORE, "TEST")
    _check_required_audit_cols(aud_v, AUDIT_REQUIRED_TPSL, "VAL")
    _check_required_audit_cols(aud_t, AUDIT_REQUIRED_TPSL, "TEST")
    _check_required_audit_cols(aud_v, AUDIT_REQUIRED_STAGE4, "VAL")
    _check_required_audit_cols(aud_t, AUDIT_REQUIRED_STAGE4, "TEST")

    # --- checks ---
    n_val = int(len(yv))
    for k in AUDIT_REQUIRED_CORE + AUDIT_REQUIRED_TPSL:
        if len(aud_v[k]) != n_val:
            raise RuntimeError(f"[VAL] audit length mismatch for {k}: y={n_val} got={len(aud_v[k])}")
    _check_flags_binary(aud_v, ["audit_market_toxic", "audit_timeout", "audit_early_abort"], "VAL")
    _check_thr01(aud_v["audit_p_thr_ev0"], "VAL/audit_p_thr_ev0", eps=float(eps))

    n_test = int(len(yt))
    for k in AUDIT_REQUIRED_CORE + AUDIT_REQUIRED_TPSL:
        if len(aud_t[k]) != n_test:
            raise RuntimeError(f"[TEST] audit length mismatch for {k}: y={n_test} got={len(aud_t[k])}")
    _check_flags_binary(aud_t, ["audit_market_toxic", "audit_timeout", "audit_early_abort"], "TEST")
    _check_thr01(aud_t["audit_p_thr_ev0"], "TEST/audit_p_thr_ev0", eps=float(eps))

    _assert_ev_inputs_finite(aud_v, "VAL", tp_max=int(tp_max), sl_max=int(sl_max))
    _assert_ev_inputs_finite(aud_t, "TEST", tp_max=int(tp_max), sl_max=int(sl_max))

    # --- align features ---
    n_model = int(booster.num_features())
    if allow_feature_mismatch:
        Xv, fix_v = _align_X_to_model(Xv, n_model)
        Xt, fix_t = _align_X_to_model(Xt, n_model)
    else:
        fix_v = _require_exact_feature_count(Xv, n_model, "VAL")
        fix_t = _require_exact_feature_count(Xt, n_model, "TEST")

    pv = booster.predict(xgb.DMatrix(Xv), output_margin=False)
    pt = booster.predict(xgb.DMatrix(Xt), output_margin=False)

    if not np.isfinite(pv).all(): raise RuntimeError("[VAL] pv NaN/inf")
    if not np.isfinite(pt).all(): raise RuntimeError("[TEST] pt NaN/inf")

    if np.min(pv) < -1e-6 or np.max(pv) > 1.0 + 1e-6:
        raise RuntimeError(f"Pred proba out of [0,1]: min={pv.min()} max={pv.max()}")

    # =========================
    # 1) Baseline (old) EV in bps on VAL for reference
    # =========================
    # NOTE: this baseline ignores new audit_* except gates.
    def _best_thr_baseline_pnl(aud: Dict[str, np.ndarray], y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
        # reuse _best_threshold_by_value with value = pnl_net_bps
        return _best_threshold_by_value(
            y=y, p=p,
            value=np.asarray(aud["audit_pnl_net_bps"], dtype=np.float64),
            audit_market_toxic=aud["audit_market_toxic"],
            audit_timeout=aud["audit_timeout"],
            audit_early_abort=aud["audit_early_abort"],
            budget_thr=np.asarray(aud["audit_p_thr_ev0"], dtype=np.float64),  # <-- FIX
            eps=float(eps),
            risk_z=float(risk_z),
            min_allowed=int(min_allowed),
            min_ev_score=float(min_ev_score)
        )

    baseline_sel = _best_thr_baseline_pnl(aud_v, yv, pv)

    def _thr_stats(thr_vec: np.ndarray) -> Dict[str, float]:
        x = np.asarray(thr_vec, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    # =========================
    # 2) NEW: TP/SL sweep using audit_hit_step_*
    #    We choose (tp_r, sl_r, side, thr) maximizing EV_R on VAL.
    # =========================
    best_combo = None
    all_combos: List[Dict[str, Any]] = []

    def _best_thr_pnl_with_budget(aud: Dict[str, np.ndarray], y: np.ndarray, p: np.ndarray, budget_thr: np.ndarray) -> Dict[str, Any]:
        return _best_threshold_by_value(
            y=y, p=p,
            value=np.asarray(aud["audit_pnl_net_bps"], dtype=np.float64),
            audit_market_toxic=aud["audit_market_toxic"],
            audit_timeout=aud["audit_timeout"],
            audit_early_abort=aud["audit_early_abort"],
            budget_thr=np.asarray(budget_thr, dtype=np.float64),
            eps=float(eps),
            risk_z=float(risk_z),
            min_allowed=int(min_allowed),
            min_ev_score=float(min_ev_score)
        )

    def _eval_pnl_at_thr(aud: Dict[str, np.ndarray], y: np.ndarray, p: np.ndarray, thr: float, budget_thr: np.ndarray) -> Dict[str, Any]:
        cls = _eval_gated_metrics(
            y=y, p=p, thr=float(thr),
            audit_market_toxic=aud["audit_market_toxic"],
            audit_timeout=aud["audit_timeout"],
            audit_early_abort=aud["audit_early_abort"],
            budget_thr=np.asarray(budget_thr, dtype=np.float64),
        )
        allowed = (
            (p >= float(thr)) &
            (np.asarray(aud["audit_market_toxic"], dtype=np.int64) == 0) &
            (np.asarray(aud["audit_timeout"], dtype=np.int64) == 0) &
            (np.asarray(aud["audit_early_abort"], dtype=np.int64) == 0) &
            (p >= np.asarray(budget_thr, dtype=np.float64))
        )
        pnl = float(np.sum(np.asarray(aud["audit_pnl_net_bps"], dtype=np.float64)[allowed])) if np.any(allowed) else 0.0
        return {**cls, "pnl_sum_bps_gated": pnl}

    for side in ["L", "S"]:
        step = np.asarray(aud_v[f"audit_first_touch_step_{side}"], dtype=np.float64)
        step_for_q = np.where(step < 0, np.nan, step)
        p95 = np.nanpercentile(step_for_q, 95)
        duration_ok_side = (step >= 0) & (step <= p95)

        for tp_r in range(1, int(tp_max) + 1):
            for sl_r in range(1, int(sl_max) + 1):

                value_r = _tp_sl_value_netR_from_hit_steps(
                    aud_v, side=side, tp_r=tp_r, sl_r=sl_r, cost_R=aud_v["audit_cost_R"]
                )

                mae_k = f"audit_mae_R_{side}"
                mfe_k = f"audit_mfe_R_{side}"
                if mae_k not in aud_v or mfe_k not in aud_v:
                    raise RuntimeError(f"Missing asymmetry cols for side={side}: need {mae_k},{mfe_k}")

                mae = np.asarray(aud_v[mae_k], dtype=np.float64)
                mfe = np.asarray(aud_v[mfe_k], dtype=np.float64)

                asym_ok = (
                    np.isfinite(mae) & np.isfinite(mfe) &
                    (mfe >= float(min_mfe_R)) &
                    (mae <= float(max_mae_over_mfe) * mfe)
                )

                extra_gate = asym_ok & duration_ok_side

                # optionnel: pas besoin de value_r = -1e6 si tu utilises extra_gate
                # value_r = np.where(asym_ok, value_r, -1e6)

                budget_thr = _p_thr_ev0_from_costR(aud_v["audit_cost_R"], tp_r=tp_r, sl_r=sl_r)

                sel = _best_threshold_by_value(
                    y=yv, p=pv,
                    value=value_r,
                    audit_market_toxic=aud_v["audit_market_toxic"],
                    audit_timeout=aud_v["audit_timeout"],
                    audit_early_abort=aud_v["audit_early_abort"],
                    budget_thr=budget_thr,
                    eps=float(eps),
                    risk_z=float(risk_z),
                    min_allowed=int(min_allowed),
                    min_ev_score=float(min_ev_score),
                    extra_gate=extra_gate,
                )

                cand = {
                    "side": side,
                    "tp_r": int(tp_r),
                    "sl_r": int(sl_r),
                    "decision_threshold": float(sel["decision_threshold"]),
                    "chosen_end": int(sel["chosen_end"]),
                    "EV_R_net": float(sel["EV"]),
                    "EV_R_sum": float(sel.get("EV_sum", sel["EV"])),
                    "EV_R_mean": float(sel.get("EV_mean", float("nan"))),
                    "EV_R_std": float(sel.get("EV_std", float("nan"))),
                    "EV_R_score": float(sel.get("EV_score", float("nan"))),
                    "TP": int(sel["TP"]),
                    "FP": int(sel["FP"]),
                    "n_pred": int(sel["n_pred"]),
                    "n_allowed": int(sel["n_allowed"]),
                    "n_blocked_flags": int(sel["n_blocked_flags"]),
                    "n_blocked_budget": int(sel["n_blocked_budget"]),
                }

                all_combos.append(cand)
                
                if best_combo is None:
                    best_combo = cand
                else:
                    score_diff = cand.get("EV_R_score", -1e30) - best_combo.get("EV_R_score", -1e30)
                    if (
                        cand.get("EV_R_score", -1e30) > best_combo.get("EV_R_score", -1e30) + eps or
                        (abs(cand.get("EV_R_score", -1e30) - best_combo.get("EV_R_score", -1e30)) <= eps and cand["TP"] > best_combo["TP"]) or
                        (abs(cand.get("EV_R_score", -1e30) - best_combo.get("EV_R_score", -1e30)) <= eps and cand["TP"] == best_combo["TP"] and cand["FP"] < best_combo["FP"]) or
                        (abs(cand.get("EV_R_score", -1e30) - best_combo.get("EV_R_score", -1e30)) <= eps and cand["TP"] == best_combo["TP"] and cand["FP"] == best_combo["FP"] and cand["decision_threshold"] > best_combo["decision_threshold"]) 
                        or (
                            abs(score_diff)<=eps and 
                            cand["TP"]==best_combo["TP"] and 
                            cand["FP"]==best_combo["FP"] and 
                            abs(cand["decision_threshold"]- best_combo["decision_threshold"])<=eps and
                            (
                                cand["tp_r"] < best_combo["tp_r"] or 
                                (cand["tp_r"] == best_combo["tp_r"] and cand["sl_r"] < best_combo["sl_r"])
                            )
                        )
                      ):
                        best_combo = cand

    if best_combo is None:
        raise RuntimeError("TP/SL sweep produced no candidate (unexpected).")

    # =========================
    # 3) Evaluate chosen combo on VAL + TEST (reporting)
    # =========================
    thr = float(best_combo["decision_threshold"])
    side = str(best_combo["side"])
    tp_r = int(best_combo["tp_r"])
    sl_r = int(best_combo["sl_r"])

    # after tp/sl chosen
    budget_val = _p_thr_ev0_from_costR(aud_v["audit_cost_R"], tp_r=tp_r, sl_r=sl_r)
    budget_tst = _p_thr_ev0_from_costR(aud_t["audit_cost_R"], tp_r=tp_r, sl_r=sl_r)
    budget_val_stats = _thr_stats(budget_val)
    budget_tst_stats = _thr_stats(budget_tst)

    # Baseline NEW gate: optimise thr sur pnl_net_bps sous le même budget gate (tp/sl/cost_R)
    baseline_new_sel = _best_thr_pnl_with_budget(aud_v, yv, pv, budget_val)

    baseline_new_val = _eval_pnl_at_thr(aud_v, yv, pv, baseline_new_sel["decision_threshold"], budget_val)
    baseline_new_tst = _eval_pnl_at_thr(aud_t, yt, pt, baseline_new_sel["decision_threshold"], budget_tst)

    # allowed masks to summarize extras
    def _allowed_mask(p: np.ndarray, 
                      thr: float, 
                      aud: Dict[str, np.ndarray], 
                      budget_thr: np.ndarray,
                      extra_gate: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        pred_pos = (p >= float(thr))
        ok_flags = (
            (np.asarray(aud["audit_market_toxic"], dtype=np.int64) == 0) &
            (np.asarray(aud["audit_timeout"], dtype=np.int64) == 0) &
            (np.asarray(aud["audit_early_abort"], dtype=np.int64) == 0)
        )
        ok_budget = (p >= np.asarray(budget_thr, dtype=np.float64))
        base = pred_pos & ok_flags & ok_budget
        if extra_gate is not None:
            base = base & np.asarray(extra_gate, dtype=bool)
        return base
    
    # value arrays
    v_val_r = _tp_sl_value_netR_from_hit_steps(aud_v, side=side, tp_r=tp_r, sl_r=sl_r, cost_R=aud_v["audit_cost_R"])
    v_tst_r = _tp_sl_value_netR_from_hit_steps(aud_t, side=side, tp_r=tp_r, sl_r=sl_r, cost_R=aud_t["audit_cost_R"])

    # --- asymmetry gate for reporting (must match selection intent) ---
    def _asym_gate(aud: Dict[str, np.ndarray], side: str) -> np.ndarray:
        mae = np.asarray(aud[f"audit_mae_R_{side}"], dtype=np.float64)
        mfe = np.asarray(aud[f"audit_mfe_R_{side}"], dtype=np.float64)
        g = (
            np.isfinite(mae) & np.isfinite(mfe) &
            (mfe >= float(min_mfe_R)) &
            (mae <= float(max_mae_over_mfe) * mfe)
        )
        return g
    
    def _duration_gate(aud: Dict[str, np.ndarray], side: str) -> np.ndarray:
        step = np.asarray(aud[f"audit_first_touch_step_{side}"], dtype=np.float64)
        step_for_q = np.where(step < 0, np.nan, step)  # ignore -1
        p95 = np.nanpercentile(step_for_q, 95)
        return (step >= 0) & (step <= p95)

    asym_val = _asym_gate(aud_v, side)
    asym_tst = _asym_gate(aud_t, side)

    duration_val = _duration_gate(aud_v, side)
    duration_tst = _duration_gate(aud_t, side)

    extra_val = asym_val & duration_val
    extra_tst = asym_tst & duration_tst

    m_val = _allowed_mask(pv, thr, aud_v, budget_val, extra_gate=extra_val)
    m_tst = _allowed_mask(pt, thr, aud_t, budget_tst, extra_gate=extra_tst)

    if not np.isfinite(v_val_r).all():
        raise RuntimeError("[VAL] v_val_r has NaN/inf after _tp_sl_value_from_first_touch()")
    if not np.isfinite(v_tst_r).all():
        raise RuntimeError("[TEST] v_tst_r has NaN/inf after _tp_sl_value_from_first_touch()")

    # sanity: allowed est bien sous-ensemble de pred_pos
    if np.any(m_val & (pv < thr)):
        raise RuntimeError("[VAL] allowed mask includes p<thr (mask bug)")
    if np.any(m_tst & (pt < thr)):
        raise RuntimeError("[TEST] allowed mask includes p<thr (mask bug)")

    # gated classification metrics
    val_cls = _eval_gated_metrics(
        y=yv, p=pv, thr=thr,
        audit_market_toxic=aud_v["audit_market_toxic"],
        audit_timeout=aud_v["audit_timeout"],
        audit_early_abort=aud_v["audit_early_abort"],
        budget_thr=budget_val,
    )
    tst_cls = _eval_gated_metrics(
        y=yt, p=pt, thr=thr,
        audit_market_toxic=aud_t["audit_market_toxic"],
        audit_timeout=aud_t["audit_timeout"],
        audit_early_abort=aud_t["audit_early_abort"],
        budget_thr=budget_tst,
    )

    # gated EVs
    val_ev_r = float(np.sum(v_val_r[m_val])) if m_val.any() else 0.0
    tst_ev_r = float(np.sum(v_tst_r[m_tst])) if m_tst.any() else 0.0

    def _risk_stats(v: np.ndarray, m: np.ndarray, z: float) -> Dict[str, float]:
        m = np.asarray(m, dtype=bool)
        vv = np.asarray(v, dtype=np.float64)[m]
        vv = vv[np.isfinite(vv)]
        n = int(vv.size)
        if n <= 0:
            return {"n": 0, "sum": 0.0, "mean": 0.0, "std": 0.0, "score": -1e30}
        s = float(np.sum(vv))
        mean = s / n
        ex2 = float(np.mean(vv * vv))
        var = max(ex2 - mean * mean, 0.0)
        std = math.sqrt(var)
        score = mean - float(z) * std / math.sqrt(max(n, 1))
        return {"n": n, "sum": s, "mean": float(mean), "std": float(std), "score": float(score)}
    
    val_rs = _risk_stats(v_val_r, m_val, risk_z)
    tst_rs = _risk_stats(v_tst_r, m_tst, risk_z)

    # baseline pnl EV (bps) evaluated at chosen thr too (so you can compare)
    val_pnl_bps = float(np.sum(np.asarray(aud_v["audit_pnl_net_bps"], dtype=np.float64)[m_val])) if m_val.any() else 0.0
    tst_pnl_bps = float(np.sum(np.asarray(aud_t["audit_pnl_net_bps"], dtype=np.float64)[m_tst])) if m_tst.any() else 0.0

    extra_val = _summarize_extra_audit(aud_v, m_val)
    extra_tst = _summarize_extra_audit(aud_t, m_tst)

    core_val = _allowed_mask(pv, thr, aud_v, budget_val, extra_gate=None)

    val_blocked_asym = int(np.sum(core_val & (~asym_val)))
    val_blocked_duration = int(np.sum(core_val & (~duration_val)))
    val_blocked_both = int(np.sum(core_val & (~asym_val) & (~duration_val)))

    val_allowed_core = int(np.sum(core_val))
    val_allowed_final = int(np.sum(m_val))

    core_tst = _allowed_mask(pt, thr, aud_t, budget_tst, extra_gate=None)

    tst_blocked_asym = int(np.sum(core_tst & (~asym_tst)))
    tst_blocked_duration = int(np.sum(core_tst & (~duration_tst)))
    tst_blocked_both = int(np.sum(core_tst & (~asym_tst) & (~duration_tst)))

    tst_allowed_core = int(np.sum(core_tst))
    tst_allowed_final = int(np.sum(m_tst))

    best = {
        "decision_threshold": thr,
        "tp_sl": {"side": side, "tp_r": tp_r, "sl_r": sl_r},
        "objective": "maximize risk-adjusted EV_R on VAL under hard gates (score=mean-z*std/sqrt(n))",
        "risk_adjusted": {"risk_z": float(risk_z), "min_allowed": int(min_allowed)},
        "val": {
            **val_cls,
            "EV_R_net": val_ev_r,
            "budget_thr_stats": budget_val_stats,
            "pnl_sum_bps_gated": val_pnl_bps,
            "extra_audit_summary": extra_val,
            "EV_R_sum":   float(val_rs["sum"]),
            "EV_R_mean":  float(val_rs["mean"]),
            "EV_R_std":   float(val_rs["std"]),
            "EV_R_score": float(val_rs["score"]),
            "n_allowed_core": int(val_allowed_core),
            "n_allowed_final": int(val_allowed_final),
            "n_blocked_asym": int(val_blocked_asym),
            "n_blocked_duration": int(val_blocked_duration),
            "n_blocked_both": int(val_blocked_both),
        },
        "test": {
            **tst_cls,
            "EV_R_net": tst_ev_r,
            "budget_thr_stats": budget_tst_stats,
            "pnl_sum_bps_gated": tst_pnl_bps,
            "extra_audit_summary": extra_tst,
            "EV_R_sum":   float(tst_rs["sum"]),
            "EV_R_mean":  float(tst_rs["mean"]),
            "EV_R_std":   float(tst_rs["std"]),
            "EV_R_score": float(tst_rs["score"]),
            "n_allowed_core": int(tst_allowed_core),
            "n_allowed_final": int(tst_allowed_final),
            "n_blocked_asym": int(tst_blocked_asym),
            "n_blocked_duration": int(tst_blocked_duration),
            "n_blocked_both": int(tst_blocked_both),
        },
        "chosen_combo": best_combo,
        "baseline_pnl_oldgate": {
            "gate": "old (audit_p_thr_ev0)",
            "val": {
                "decision_threshold": float(baseline_sel["decision_threshold"]),
                "EV_bps": float(baseline_sel["EV"]),
                "TP": int(baseline_sel["TP"]),
                "FP": int(baseline_sel["FP"]),
                "n_allowed": int(baseline_sel["n_allowed"]),
            },
        },
        "baseline_pnl_newgate": {
            "gate": "new (budget_thr(tp,sl,cost_R)) using chosen tp/sl",
            "val": {
                "decision_threshold": float(baseline_new_sel["decision_threshold"]),
                "pnl_sum_bps_gated": float(baseline_new_val["pnl_sum_bps_gated"]),
                "TP": int(baseline_new_val["TP"]),
                "FP": int(baseline_new_val["FP"]),
                "n_allowed": int(baseline_new_val["n_allowed"]),
            },
            "test": {
                "pnl_sum_bps_gated": float(baseline_new_tst["pnl_sum_bps_gated"]),
                "TP": int(baseline_new_tst["TP"]),
                "FP": int(baseline_new_tst["FP"]),
                "n_allowed": int(baseline_new_tst["n_allowed"]),
            },
        },
    }

    # A) candidates serialization (optional)
    if all_combos:
        all_combos_sorted = sorted(
            all_combos,
            key=lambda c: (
                float(c.get("EV_R_score", -1e30)),
                int(c.get("TP", 0)),
                -int(c.get("FP", 0)),
                float(c.get("decision_threshold", 0.0)),
            ),
            reverse=True,
        )
    else:
        all_combos_sorted = []

    out = {
        "feature_fix": {"val": fix_v, "test": fix_t},
        "best_by_val_only": best,
        "note": "Selection uses ONLY VAL (risk-adjusted) EV_R from TP/SL simulation via audit_hit_step_* (L/S) with cost_R. TEST is reporting only.",
        "audit_cols_seen": audit_cols,
    }

    if bool(emit_candidates):
        k = max(int(candidates_topk), 0)   # <-- important: allow 0
        out["candidates_topk"] = all_combos_sorted[:k]
        out["candidates_note"] = "Sorted by (EV_R_score, TP, -FP, thr). EV_R_score = mean - z*std/sqrt(n_allowed)."

    return out

def main():
    args = parse_args()
    fs = s3fs.S3FileSystem()

    _require_csv_prefix(fs, TRN_DIR, "train")
    _require_csv_prefix(fs, VAL_DIR, "validation")
    _require_csv_prefix(fs, TST_DIR, "test")
    _require_csv_prefix(fs, TRW_DIR, "train_weight")
    _require_csv_prefix(fs, VAW_DIR, "validation_weight")

    # meta contract FIRST (needed for audit paths)
    label_name, feature_names, audit_cols, audit_paths = _load_meta_contract(fs, META_DIR)

    val_audit_prefix  = f"{DATA_ROOT}/{audit_paths['val_audit'].split('/')[0]}"
    test_audit_prefix = f"{DATA_ROOT}/{audit_paths['test_audit'].split('/')[0]}"

    _require_csv_prefix(fs, val_audit_prefix, "val_audit")
    _require_csv_prefix(fs, test_audit_prefix, "test_audit")

    print(f"[meta] label_col={label_name} | n_features={len(feature_names)}")
    print(f"[meta] audit_cols(n={len(audit_cols)})={audit_cols}")
    print(f"[meta] audit_paths={audit_paths}")

    # fail-fast if audit_cols don't include new fields
    need = set(AUDIT_REQUIRED_CORE + AUDIT_REQUIRED_STAGE4)
    missing = [c for c in need if c not in set(audit_cols)]
    if missing:
        raise RuntimeError(f"Stage4 missing required audit cols in columns.json: {missing}")

    hp_base = dict(HP_BASE)

    sess = sagemaker.Session(boto3.Session(region_name=REGION))
    s3c = _boto3_client("s3", REGION)

    PRIORITY_COMBOS = [
        dict(),  # baseline
        dict(max_depth=3, min_child_weight=10, reg_lambda=10.0, reg_alpha=0.5, gamma=1.0),
        dict(max_depth=5, min_child_weight=5,  reg_lambda=5.0,  reg_alpha=0.2, gamma=0.5),
    ]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"{MODEL_BASE_S3}/grid/runs/{stamp}"
    results: List[Dict[str, Any]] = []

    def _checkpoint(name: str, row: Dict[str, Any]):
        uri = f"{run_dir}/{name}"
        _s3_put_bytes_uri(s3c, uri, _json_dumps_safe(row))
        print(f"[checkpoint] écrit: {uri}")

    if args.mode == "eval":
        if not args.model_uri:
            raise SystemExit("--mode eval requiert --model-uri s3://.../model.tar.gz")
        print(f"[run] MODE=eval | model_uri={args.model_uri}")
        row: Dict[str, Any] = {"mode": "eval", "model_uri": args.model_uri, "status": "ok"}

        sweep = evaluate_ev_sweep(
            model_uri=row["model_uri"],
            val_prefix=VAL_DIR,
            tst_prefix=TST_DIR,
            allow_feature_mismatch=bool(args.allow_feature_mismatch),
            eps=float(args.eps),
            tp_max=int(args.tp_max),
            sl_max=int(args.sl_max),
            emit_candidates=bool(args.emit_candidates),
            candidates_topk=int(args.candidates_topk),
            risk_z=float(args.risk_z),
            min_allowed=int(args.min_allowed),
            max_mae_over_mfe=float(args.max_mae_over_mfe),
            min_mfe_R=float(args.min_mfe_R),
            min_ev_score=float(args.min_ev_score),
        )
        row["sweep"] = _json_sanitize(sweep)
        row["best"]  = _json_sanitize(sweep["best_by_val_only"])
        _checkpoint("eval_only.json", row)
        results.append(row)
    else:
        print(f"[run] MODE=train | {len(PRIORITY_COMBOS)} runs | instance={INSTANCE_TYPE} | spot={bool(SPOT)}")
        for j, hp0 in enumerate(PRIORITY_COMBOS, start=1):
            hps = _merge_hps(hp_base, hp0)
            row: Dict[str, Any] = {"mode": "train", "hps": _json_sanitize(hps)}

            try:
                t0 = time.time()
                print(f"\n=== RUN {j}/{len(PRIORITY_COMBOS)} === hps={hps}")
                est = submit_job(hps, sess)
                row["train_seconds"] = float(round(time.time() - t0, 2))
                row["model_uri"] = est.model_data

                sweep = evaluate_ev_sweep(
                    model_uri=row["model_uri"],
                    val_prefix=VAL_DIR,
                    tst_prefix=TST_DIR,
                    allow_feature_mismatch=bool(args.allow_feature_mismatch),
                    eps=float(args.eps),
                    tp_max=int(args.tp_max),
                    sl_max=int(args.sl_max),
                    emit_candidates=bool(args.emit_candidates),
                    candidates_topk=int(args.candidates_topk),
                    risk_z=float(args.risk_z),
                    min_allowed=int(args.min_allowed),
                    max_mae_over_mfe=float(args.max_mae_over_mfe),
                    min_mfe_R=float(args.min_mfe_R),
                    min_ev_score=float(args.min_ev_score),
                )

                row["sweep"] = _json_sanitize(sweep)
                row["best"]  = _json_sanitize(sweep["best_by_val_only"])
                row["status"] = "ok"

                b = sweep["best_by_val_only"]
                print(
                    f"[BEST@VAL] thr={b['decision_threshold']:.6g} | "
                    f"TP/SL={b['tp_sl']['side']} tp={b['tp_sl']['tp_r']}R sl={b['tp_sl']['sl_r']}R | "
                    f"VAL EV_R_net={b['val'].get('EV_R_net', float('nan')):.6g} | "
                    f"VAL pnl_bps(gated)={b['val'].get('pnl_sum_bps_gated', float('nan')):.6g} | "
                    f"TP={b['val']['TP']} FP={b['val']['FP']}"
                )

            except Exception as e:
                tb = traceback.format_exc()
                row.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": tb[-2000:]})
                print(f"[ERROR] run {j} -> {e}", file=sys.stderr)

            _checkpoint(f"combo_{j:02d}.json", row)
            results.append(row)

    ok_rows = [r for r in results if r.get("status") == "ok"]
    best = None
    if ok_rows:
        # Select by VAL EV_R first, then TP, then -FP, then thr
        def _run_key(r):
            v = r["best"]["val"]
            return (
                v.get("EV_R_score", -1e30),                 # score robuste
                v.get("TP", 0),
                -v.get("FP", 0),
                r["best"].get("decision_threshold", 0.0),
            )

        best = max(ok_rows, key=_run_key)

    final_out = {
        "timestamp": stamp,
        "data_root": DATA_ROOT,
        "hp_base": _json_sanitize(hp_base),
        "selection": {
            "type": "EV_R_net",
            "note": "VAL-only maximize EV_R_net from TP/SL simulation via audit_hit_step_* with HARD filters: allowed=(p>=thr) & (p>=budget_thr(tp,sl,cost_R)) & (flags==0)",
            "tp_max": int(args.tp_max),
            "sl_max": int(args.sl_max),
        },
        "results": results,
        "best": best,
    }

    summary_uri = f"{MODEL_BASE_S3}/grid/{stamp}_summary.json"
    _s3_put_bytes_uri(s3c, summary_uri, _json_dumps_safe(final_out))
    print("\nRésumé écrit:", summary_uri)

    if best is None:
        print("\nAucun run OK (tous en échec). Consulte les JSON de run pour diagnostiquer.")
        return

    # Manifest release (threshold = best VAL-only)
    try:
        RELEASE_BASE = "s3://tradebot-config-tokyo/models/xgb/releases"
        release_name = f"go-{stamp}-{VERSION}"
        manifest_uri = f"{RELEASE_BASE}/{release_name}/manifest.json"

        scaler_stats_uri = f"{META_DIR}/scaler_stats.json"

        b = best["best"]

        manifest = {
            "model_uri": best["model_uri"],
            "version": f"{VERSION}-go",
            "timestamp": stamp,
            "side": "go",
            "data_root": DATA_ROOT,
            "metrics": {
                "selection": (
                    "VAL-only maximize risk-adjusted EV_R score via audit_hit_step_* "
                    "with HARD filters: allowed=(p>=thr) & (p>=budget_thr(tp,sl,audit_cost_R)) & (flags==0). "
                    "Score = mean(netR_allowed) - z*std/sqrt(n_allowed)."
                ),
                "risk_adjusted": {"risk_z": float(args.risk_z), "min_allowed": int(args.min_allowed)},
                "best": best["best"],
            },
            "inference": {
                "decision_threshold": float(b["decision_threshold"]),
                "proba_semantics": "p = P(Y=1)",
                "invert_output": False,
                "autoflip_allowed": False,
                "tp_sl": {
                    "side": b["tp_sl"]["side"],
                    "tp_r": int(b["tp_sl"]["tp_r"]),
                    "sl_r": int(b["tp_sl"]["sl_r"]),
                    "note": "Chosen on VAL via audit_hit_step_* TP/SL simulation. If your live system uses different TP/SL logic, align it.",
                },
                "budget_gate": {
                    "name": "p_thr_ev0_from_costR",
                    "formula": "(sl_r + audit_cost_R) / (tp_r + sl_r)",
                    "clip01": True,
                    "tp_r": int(b["tp_sl"]["tp_r"]),
                    "sl_r": int(b["tp_sl"]["sl_r"]),
                },
                "ev_fields": {
                    "audit_cost_R": "audit_cost_R",
                    "audit_market_toxic": "audit_market_toxic",
                    "audit_timeout": "audit_timeout",
                    "audit_early_abort": "audit_early_abort",
                    "audit_pnl_net_bps": "audit_pnl_net_bps",
                    "audit_p_thr_ev0": "audit_p_thr_ev0",
                    "audit_hit_step_p1R_L": "audit_hit_step_p1R_L",
                    "audit_hit_step_p1R_S": "audit_hit_step_p1R_S",
                    "budget_thr": "p_thr_ev0_from_costR(audit_cost_R,tp_r,sl_r)",
                },
            },
            "data_contract": {
                "label_col": label_name,
                "drop_cols": [],
                "features": feature_names,
                "scaler_stats_uri": scaler_stats_uri,
                "normalize_at_infer": False,
            },
        }

        _s3_put_bytes_uri(s3c, manifest_uri, _json_dumps_safe(manifest))
        print(f"[manifest] écrit: {manifest_uri}")
    except Exception as e:
        print(f"[manifest] WARN: impossible de générer le manifest: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()