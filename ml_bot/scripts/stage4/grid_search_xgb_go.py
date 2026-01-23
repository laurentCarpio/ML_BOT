# ml_bot/scripts/stage4/grid_search_xgb_go.py
import json
import time
import tarfile
import tempfile
import io, re
import sys
import traceback
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
#       allowed = (p>=thr) & (p>=audit_p_thr_ev0) & (flags==0)
#   - objective for selection:
#       maximize EV_R = sum(simulated_R[allowed])
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
                
def _assert_hit_flags_monotonic(aud: Dict[str, np.ndarray], side: str, where: str):
    # ex: hit_p5 => hit_p4 => ... => hit_p1
    for k in [5, 4, 3, 2]:
        a = aud.get(f"audit_hit_p{k}R_{side}")
        b = aud.get(f"audit_hit_p{k-1}R_{side}")
        if a is not None and b is not None:
            if np.any((a >= 0.5) & (b < 0.5)):
                raise RuntimeError(f"[{where}] monotonic violated: hit_p{k}R_{side} implies hit_p{k-1}R_{side}")

    for k in [3, 2]:
        a = aud.get(f"audit_hit_m{k}R_{side}")
        b = aud.get(f"audit_hit_m{k-1}R_{side}")
        if a is not None and b is not None:
            if np.any((a >= 0.5) & (b < 0.5)):
                raise RuntimeError(f"[{where}] monotonic violated: hit_m{k}R_{side} implies hit_m{k-1}R_{side}")

def _assert_first_touch_sane(aud: Dict[str, np.ndarray], side: str, where: str):
    ft = np.asarray(aud[f"audit_first_touch_{side}"], dtype=np.float64)
    if not np.isfinite(ft).all():
        raise RuntimeError(f"[{where}] audit_first_touch_{side} has NaN/inf")

    # si tu veux imposer un encodage quasi-integer (recommandé)
    frac = np.abs(ft - np.round(ft))
    if np.nanmax(frac) > 1e-6:
        raise RuntimeError(f"[{where}] audit_first_touch_{side} not integer-like (max frac={np.nanmax(frac)})")

    # check logique signes (optionnel mais utile)
    # si first_touch >0, ça devrait correspondre à une hit_p1R_* dans l'idéal
    hp1 = aud.get(f"audit_hit_p1R_{side}")
    if hp1 is not None:
        bad = (ft > 0) & (np.asarray(hp1, dtype=np.float64) < 0.5)
        # tolérance: ne pas fail si tu sais que hit flags ne sont pas fiables
        if np.any(bad):
            # mets juste un warning si tu préfères
            raise RuntimeError(f"[{where}] first_touch_{side}>0 but hit_p1R_{side}=0 on some rows (incoherent audit)")

    hm1 = aud.get(f"audit_hit_m1R_{side}")
    if hm1 is not None:
        bad = (ft < 0) & (np.asarray(hm1, dtype=np.float64) < 0.5)
        if np.any(bad):
            raise RuntimeError(f"[{where}] first_touch_{side}<0 but hit_m1R_{side}=0 on some rows (incoherent audit)")

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
        if f"audit_first_touch_{side}" in aud:
            _assert_first_touch_sane(aud, side, where)
            _assert_hit_flags_monotonic(aud, side, where)

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
    ap.add_argument("--tp-max", type=int, default=5, help="Max TP in R (inclusive).")
    ap.add_argument("--sl-max", type=int, default=3, help="Max SL in R (inclusive).")
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
        raise RuntimeError(f"[{where}] audit_p_thr_ev0 contains NaN/inf.")
    if np.any(thr < -eps) or np.any(thr > 1.0 + eps):
        raise RuntimeError(f"[{where}] audit_p_thr_ev0 out of [0,1].")

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

def _tp_sl_value_from_hit_steps(
    aud: Dict[str, np.ndarray],
    side: str,
    tp_r: int,
    sl_r: int,
) -> np.ndarray:
    """
    Uses hit STEPS to simulate TP/SL in R:
      t_tp = hit_step_p{tp_r}R_side
      t_sl = hit_step_m{sl_r}R_side

    Convention:
      - step >= 0 : touched at that step
      - step == -1 : never touched
    Decision:
      - if t_tp>=0 and (t_sl<0 or t_tp < t_sl): +tp_r
      - elif t_sl>=0: -sl_r
      - else: 0 (no touch / time)
    """
    t_tp = np.asarray(aud[f"audit_hit_step_p{tp_r}R_{side}"], dtype=np.int64)
    t_sl = np.asarray(aud[f"audit_hit_step_m{sl_r}R_{side}"], dtype=np.int64)

    # sanity: steps must be >= -1
    if np.any(t_tp < -1) or np.any(t_sl < -1):
        raise RuntimeError(f"Bad hit steps (<-1) for side={side} tp={tp_r} sl={sl_r}")

    out = np.zeros_like(t_tp, dtype=np.float64)

    tp_hit = (t_tp >= 0)
    sl_hit = (t_sl >= 0)

    tp_wins = tp_hit & (~sl_hit | (t_tp < t_sl))
    sl_wins = sl_hit & (~tp_hit | (t_sl < t_tp))

    out[tp_wins] = float(tp_r)
    out[sl_wins] = -float(sl_r)

    # tie case (t_tp == t_sl >=0) : decide policy
    # recommended: treat as SL (conservative) OR 0 (ignore). Pick one.
    ties = tp_hit & sl_hit & (t_tp == t_sl)
    # conservative:
    out[ties] = -float(sl_r)

    return out

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
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Choose a decision threshold `thr` that maximizes EV(value) under hard gates.

    Implementation detail (important):
      - We sort samples by predicted probability `p` in descending order.
      - Each candidate threshold `thr` is taken as one of the unique values of `p`.
      - For a given `thr`, the predicted-positive set is the PREFIX of the sorted list:
            pred(thr) = { i : p_i >= thr }   (i.e., indices 0..end in sorted order)

    Hard gates (applied only within the predicted-positive prefix):
      - ok_flags  = (audit_market_toxic==0) & (audit_timeout==0) & (audit_early_abort==0)
      - ok_budget = (p >= audit_p_thr_ev0)
      - allowed(thr) = pred(thr) & ok_flags & ok_budget

    Objective:
      - EV(thr) = sum(value[i] for i in allowed(thr))

    Tie-break (in order):
      1) higher EV
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

    pred_cum = np.arange(n, dtype=np.int64) + 1
    allowed_cum = np.cumsum(allowed.astype(np.int64))
    blocked_flags_cum  = np.cumsum((~ok_flags).astype(np.int64))
    blocked_budget_cum = np.cumsum((ok_flags & (~ok_budget)).astype(np.int64))

    EV_cum = np.cumsum(vv * allowed)
    TP_cum = np.cumsum((tp_mask & allowed).astype(np.int64))
    FP_cum = np.cumsum((fp_mask & allowed).astype(np.int64))

    ends = np.array([0], dtype=np.int64) if n == 1 else np.r_[np.where(ps[1:] != ps[:-1])[0], n - 1].astype(np.int64)

    best = None
    for end in ends:
        cand = {
            "end": int(end),
            "thr": float(ps[end]),
            "EV": float(EV_cum[end]),
            "TP": int(TP_cum[end]),
            "FP": int(FP_cum[end]),
            "n_pred": int(pred_cum[end]),
            "n_allowed": int(allowed_cum[end]),
            "n_blocked_flags": int(blocked_flags_cum[end]),
            "n_blocked_budget": int(blocked_budget_cum[end]),
        }
        if best is None:
            best = cand
        else:
            if (
                cand["EV"] > best["EV"] + eps or
                (abs(cand["EV"] - best["EV"]) <= eps and cand["TP"] > best["TP"]) or
                (abs(cand["EV"] - best["EV"]) <= eps and cand["TP"] == best["TP"] and cand["FP"] < best["FP"]) or
                (abs(cand["EV"] - best["EV"]) <= eps and cand["TP"] == best["TP"] and cand["FP"] == best["FP"] and cand["thr"] > best["thr"])
            ):
                best = cand

    if best is None:
        return {"decision_threshold": 1.0, "chosen_end": -1, "why": "no candidates", "EV": 0.0, "TP": 0, "FP": 0}

    return {
        "decision_threshold": float(best["thr"]),
        "chosen_end": int(best["end"]),
        "EV": float(best["EV"]),
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
    Small diagnostics to help compare models:
      - mean/quantiles of mfe/mae (if present)
      - hit rates (if present)
      - first_touch stats (if present)
    """
    out: Dict[str, Any] = {}
    m = np.asarray(allowed_mask, dtype=bool)
    if m.size == 0 or m.sum() == 0:
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

    for k in ["audit_mfe_R_L", "audit_mae_R_L", "audit_mfe_R_S", "audit_mae_R_S"]:
        if k in aud:
            out[k] = _q(aud[k][m])

    for k in ["audit_first_touch_L", "audit_first_touch_S"]:
        if k in aud:
            ft = aud[k][m]
            out[k] = {
                **_q(ft),
                "frac_pos": float(np.mean(ft > 0)) if np.isfinite(ft).any() else 0.0,
                "frac_neg": float(np.mean(ft < 0)) if np.isfinite(ft).any() else 0.0,
                "frac_zero": float(np.mean(ft == 0)) if np.isfinite(ft).any() else 0.0,
            }

    # hit flags
    hit_keys = [k for k in AUDIT_DIAG_OPTIONAL if k.startswith("audit_hit_") and k in aud]
    if hit_keys:
        hit_rates = {}
        for k in hit_keys:
            v = aud[k][m]
            v = np.asarray(v, dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            # treat as 0/1
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
            audit_p_thr_ev0=aud["audit_p_thr_ev0"],
            eps=float(eps),
        )

    baseline_sel = _best_thr_baseline_pnl(aud_v, yv, pv)

    # =========================
    # 2) NEW: TP/SL sweep using audit_hit_step_*
    #    We choose (tp_r, sl_r, side, thr) maximizing EV_R on VAL.
    # =========================
    best_combo = None

    for side in ["L", "S"]:
        for tp_r in range(1, int(tp_max) + 1):
            for sl_r in range(1, int(sl_max) + 1):
                value_r = _tp_sl_value_from_hit_steps(aud_v, side=side, tp_r=tp_r, sl_r=sl_r)
                budget_thr = _p_thr_ev0_from_costR(aud_v["audit_cost_R"], tp_r=tp_r, sl_r=sl_r)

                sel = _best_threshold_by_value(
                    y=yv, p=pv,
                    value=value_r,
                    audit_market_toxic=aud_v["audit_market_toxic"],
                    audit_timeout=aud_v["audit_timeout"],
                    audit_early_abort=aud_v["audit_early_abort"],
                    budget_thr=budget_thr,
                    eps=float(eps),
                )

                cand = {
                    "side": side,
                    "tp_r": int(tp_r),
                    "sl_r": int(sl_r),
                    "decision_threshold": float(sel["decision_threshold"]),
                    "chosen_end": int(sel["chosen_end"]),
                    "EV_R": float(sel["EV"]),
                    "TP": int(sel["TP"]),
                    "FP": int(sel["FP"]),
                    "n_pred": int(sel["n_pred"]),
                    "n_allowed": int(sel["n_allowed"]),
                    "n_blocked_flags": int(sel["n_blocked_flags"]),
                    "n_blocked_budget": int(sel["n_blocked_budget"]),
                }

                if best_combo is None:
                    best_combo = cand
                else:
                    # tie-break: EV_R, TP, -FP, thr
                    if (
                        cand["EV_R"] > best_combo["EV_R"] + eps or
                        (abs(cand["EV_R"] - best_combo["EV_R"]) <= eps and cand["TP"] > best_combo["TP"]) or
                        (abs(cand["EV_R"] - best_combo["EV_R"]) <= eps and cand["TP"] == best_combo["TP"] and cand["FP"] < best_combo["FP"]) or
                        (abs(cand["EV_R"] - best_combo["EV_R"]) <= eps and cand["TP"] == best_combo["TP"] and cand["FP"] == best_combo["FP"] and cand["decision_threshold"] > best_combo["decision_threshold"])
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

    # allowed masks to summarize extras
    def _allowed_mask(p: np.ndarray, thr: float, aud: Dict[str, np.ndarray]) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        pred_pos = (p >= float(thr))
        ok_flags = (
            (np.asarray(aud["audit_market_toxic"], dtype=np.int64) == 0) &
            (np.asarray(aud["audit_timeout"], dtype=np.int64) == 0) &
            (np.asarray(aud["audit_early_abort"], dtype=np.int64) == 0)
        )
        ok_budget = (p >= np.asarray(aud["audit_p_thr_ev0"], dtype=np.float64))
        return (pred_pos & ok_flags & ok_budget)
    
    # value arrays
    v_val_r = _tp_sl_value_from_hit_steps(aud_v, side=side, tp_r=tp_r, sl_r=sl_r)
    v_tst_r = _tp_sl_value_from_hit_steps(aud_t, side=side, tp_r=tp_r, sl_r=sl_r)

    m_val = _allowed_mask(pv, thr, aud_v)
    m_tst = _allowed_mask(pt, thr, aud_t)

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
        audit_p_thr_ev0=aud_v["audit_p_thr_ev0"],
    )
    tst_cls = _eval_gated_metrics(
        y=yt, p=pt, thr=thr,
        audit_market_toxic=aud_t["audit_market_toxic"],
        audit_timeout=aud_t["audit_timeout"],
        audit_early_abort=aud_t["audit_early_abort"],
        audit_p_thr_ev0=aud_t["audit_p_thr_ev0"],
    )

    # gated EVs
    val_ev_r = float(np.sum(v_val_r[m_val])) if m_val.any() else 0.0
    tst_ev_r = float(np.sum(v_tst_r[m_tst])) if m_tst.any() else 0.0

    # baseline pnl EV (bps) evaluated at chosen thr too (so you can compare)
    val_pnl_bps = float(np.sum(np.asarray(aud_v["audit_pnl_net_bps"], dtype=np.float64)[m_val])) if m_val.any() else 0.0
    tst_pnl_bps = float(np.sum(np.asarray(aud_t["audit_pnl_net_bps"], dtype=np.float64)[m_tst])) if m_tst.any() else 0.0

    extra_val = _summarize_extra_audit(aud_v, m_val)
    extra_tst = _summarize_extra_audit(aud_t, m_tst)

    best = {
        "decision_threshold": thr,
        "tp_sl": {"side": side, "tp_r": tp_r, "sl_r": sl_r},
        "objective": "maximize EV_R (VAL-only) under hard gates",
        "val": {
            **val_cls,
            "EV_R": val_ev_r,
            "pnl_sum_bps_gated": val_pnl_bps,
            "extra_audit_summary": extra_val,
        },
        "test": {
            **tst_cls,
            "EV_R": tst_ev_r,
            "pnl_sum_bps_gated": tst_pnl_bps,
            "extra_audit_summary": extra_tst,
        },
        "chosen_combo": best_combo,
        "baseline_val_best_thr_by_pnl_bps": {
            "decision_threshold": float(baseline_sel["decision_threshold"]),
            "EV_bps": float(baseline_sel["EV"]),  # here EV is pnl_bps because value=pnl_bps
            "TP": int(baseline_sel["TP"]),
            "FP": int(baseline_sel["FP"]),
            "n_allowed": int(baseline_sel["n_allowed"]),
        },
    }

    return {
        "feature_fix": {"val": fix_v, "test": fix_t},
        "best_by_val_only": best,
        "note": "Selection uses ONLY VAL EV_R from TP/SL simulation via audit_hit_step_* (L/S). TEST is reporting only.",
        "audit_cols_seen": audit_cols,
    }

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
    missing_needed = [c for c in AUDIT_REQUIRED_CORE if c not in set(audit_cols)]
    if missing_needed:
        raise RuntimeError(
            "Stage4 requires new audit_* but columns.json audit_format is missing required fields: "
            f"{missing_needed}"
        )

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
                )

                row["sweep"] = _json_sanitize(sweep)
                row["best"]  = _json_sanitize(sweep["best_by_val_only"])
                row["status"] = "ok"

                b = sweep["best_by_val_only"]
                print(
                    f"[BEST@VAL] thr={b['decision_threshold']:.6g} | "
                    f"TP/SL={b['tp_sl']['side']} tp={b['tp_sl']['tp_r']}R sl={b['tp_sl']['sl_r']}R | "
                    f"VAL EV_R={b['val'].get('EV_R', float('nan')):.6g} | "
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
        best = max(
            ok_rows,
            key=lambda r: (
                r["best"]["val"].get("EV_R", -1e30),
                r["best"]["val"].get("TP", 0),
                -r["best"]["val"].get("FP", 0),
                r["best"].get("decision_threshold", 0.0),
            ),
        )

    final_out = {
        "timestamp": stamp,
        "data_root": DATA_ROOT,
        "hp_base": _json_sanitize(hp_base),
        "selection": {
            "type": "EV_R",
            "note": "VAL-only maximize EV_R from TP/SL simulation via audit_hit_step_* with HARD filters: allowed=(p>=thr)&(p>=audit_p_thr_ev0)&(flags==0)",
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
                "selection": "VAL-only maximize EV_R via audit_hit_step_* with HARD filters: allowed=(p>=thr)&(p>=audit_p_thr_ev0)&(flags==0)",
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
                "ev_fields": {
                    # gating + baseline
                    "audit_pnl_net_bps": "audit_pnl_net_bps",
                    "audit_market_toxic": "audit_market_toxic",
                    "audit_timeout": "audit_timeout",
                    "audit_early_abort": "audit_early_abort",
                    "audit_p_thr_ev0": "audit_p_thr_ev0",
                    # new fields used for EV_R
                    "audit_hit_step_p1R_L": "audit_hit_step_p1R_L",
                    "audit_hit_step_p1R_S": "audit_hit_step_p1R_S",
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