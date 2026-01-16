# ml_bot/scripts/stage4/grid_search_xgb_go.py
import json
import time
import tarfile
import tempfile
import io
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

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

VAA_DIR = f"{DATA_ROOT}/val_audit"
TSA_DIR = f"{DATA_ROOT}/test_audit"

META_DIR = f"{DATA_ROOT}/_meta"

INSTANCE_TYPE   = "ml.c5.xlarge"
SPOT            = True
MAX_RUN_SEC     = 60 * 20
MAX_WAIT_SEC    = 60 * 40

# === Stage4: sweep budgets (VAL-only decision) ===

HP_BASE: Dict[str, Any] = dict(
    tree_method="hist",
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_round=600,              # un peu plus long pour compenser le modèle plus régularisé
    early_stopping_rounds=80,   # laisse respirer avant de stopper
    objective="binary:logistic",
    eval_metric="logloss",

    # regularisation + anti-suractivation
    reg_lambda=15.0,            # ↑ shrink les splits
    reg_alpha=1.0,              # ↑ pousse à des splits plus “utiles”
    gamma=2.0,                  # ↑ seuil minimum de gain => moins de micro-splits

    # structure plus simple (évite le plateau)
    max_depth=3,                # ↓ moins profond                    .....essayer 2  plus tard ?
    min_child_weight=30,        # ↑ empêche les feuilles “fragiles”  .....essayer 50 plus tard ?
    max_delta_step=1,
)

XGB_IMAGE = image_uri_retrieve("xgboost", REGION, version="1.7-1")

# ========= HELPERS =========
def _check_flags_binary(aud: Dict[str, np.ndarray], keys: List[str], where: str):
    for k in keys:
        if k not in aud:
            raise RuntimeError(f"[{where}] missing audit col: {k}")
        v = aud[k].astype(np.int64, copy=False)
        if not np.isin(v, [0, 1]).all():
            raise RuntimeError(f"[{where}] {k} must be 0/1.")
        
def _parse_list(s: str, cast=float) -> List:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_args():
    ap = argparse.ArgumentParser("Stage4 grid search + VAL-only budgeted threshold sweep (fp_cost_bps, audit_market_toxic).")
    ap.add_argument("--mode", choices=["train", "eval"], default="train",
                    help="train = lance des jobs SageMaker, eval = évalue un model_uri existant (sans retrain).")
    ap.add_argument("--model-uri", default="",
                    help="Requis en mode eval: s3://.../model.tar.gz")
        
    ap.add_argument("--audit-cols", default="fp_cost_bps,audit_early_abort,audit_timeout,audit_market_toxic,audit_p_thr_ev0",
                    help="Colonnes attendues dans *_audit (no header), ordre important."
    )
    
    ap.add_argument("--require-audit", action=argparse.BooleanOptionalAction, default=True,
                    help="Fail si val_audit n'existe pas.")
    ap.add_argument("--allow-feature-mismatch", action="store_true", default=False)

    # tie-break + safety
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Epsilon pour seuils si nécessaire.")
    return ap.parse_args()

def _read_json_s3(fs: s3fs.S3FileSystem, uri: str) -> dict:
    with fs.open(uri, "rb") as f:
        return json.load(f)

def _load_columns_meta(fs: s3fs.S3FileSystem, meta_dir: str):
    cols_path = f"{meta_dir}/columns.json"
    meta = _read_json_s3(fs, cols_path)
    label = str(meta.get("label", "is_tradeable"))
    feats = meta.get("features") or meta.get("feature_cols") or []
    if not isinstance(feats, list) or not feats:
        raise RuntimeError(f"columns.json invalide: features vide ({cols_path})")
    return label, [str(x) for x in feats]

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

def _load_audit_by_cols(fs: s3fs.S3FileSystem, prefix: str, audit_cols: List[str]) -> Dict[str, np.ndarray]:
    df = _load_csv_dir(fs, prefix)
    if df.shape[1] != len(audit_cols):
        raise RuntimeError(f"[{prefix}] audit shape mismatch: got {df.shape[1]} expected {len(audit_cols)} ({audit_cols})")
    out = {}
    for j, name in enumerate(audit_cols):
        out[name] = df.iloc[:, j].astype(np.float64).to_numpy()
    return out

def _check_thr01(thr: np.ndarray, where: str, eps: float = 1e-12):
    thr = np.asarray(thr, dtype=np.float64)
    if not np.isfinite(thr).all():
        raise RuntimeError(f"[{where}] audit_p_thr_ev0 contains NaN/inf.")
    if np.any(thr < -eps) or np.any(thr > 1.0 + eps):
        raise RuntimeError(f"[{where}] audit_p_thr_ev0 out of [0,1].")

def _best_threshold_by_ev(
    y: np.ndarray,
    p: np.ndarray,
    fp_cost_bps: np.ndarray,
    audit_market_toxic: np.ndarray,
    audit_timeout: np.ndarray,
    audit_early_abort: np.ndarray,
    audit_p_thr_ev0: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Choisit un seuil (sur VAL) qui maximise:
      EV = Σ_{TP} ( (1 - p_thr) * fp_cost_bps )  -  Σ_{FP} ( p_thr * fp_cost_bps )
           -  Σ(flag penalties)

    Flag penalties (en bps): on réutilise fp_cost_bps comme pénalité unitaire si flag==1.
      penalty_flags = Σ_{pred_pos & flag}( fp_cost_bps )

    Notes:
    - audit_p_thr_ev0 est obligatoire et déjà borné [0,1].
    - On évalue uniquement des seuils aux fins de "score groups" (p identiques).
    - Tie-break: max EV, puis max TP, puis min FP, puis thr plus haut.
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    cost = np.asarray(fp_cost_bps, dtype=np.float64)
    mt = np.asarray(audit_market_toxic, dtype=np.int64)
    to = np.asarray(audit_timeout, dtype=np.int64)
    ab = np.asarray(audit_early_abort, dtype=np.int64)
    thr0 = np.asarray(audit_p_thr_ev0, dtype=np.float64)

    n = int(y.shape[0])
    if n == 0:
        return {"decision_threshold": 1.0, "chosen_end": -1, "why": "empty", "EV_bps": 0.0, "TP": 0, "FP": 0}

    _check_flags_binary(
        {"audit_market_toxic": mt, "audit_timeout": to, "audit_early_abort": ab},
        ["audit_market_toxic", "audit_timeout", "audit_early_abort"],
        "EV/flags"
    )
    if np.any(cost < -eps):
        raise RuntimeError("[EV] fp_cost_bps has negative values (should be >= 0).")
    _check_thr01(thr0, "EV/audit_p_thr_ev0", eps=eps)

    # Sort par proba décroissante (stable)
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    cs = cost[order]
    mt = mt[order]
    to = to[order]
    ab = ab[order]
    th = thr0[order]

    pred_pos_prefix = np.ones(n, dtype=bool)  # pour "prefix" logique
    tp_mask = (ys == 1)
    fp_mask = (ys == 0)

    # --- EV components on prefix ---
    # TP reward: (1 - th) * cs
    tp_reward = ((1.0 - th) * cs) * tp_mask

    # FP penalty: th * cs
    fp_penalty = (th * cs) * fp_mask

    # Flags penalties (appliquées sur pred_pos uniquement)
    flag_pen = cs * (mt == 1) + cs * (to == 1) + cs * (ab == 1)

    tp_reward_cum = np.cumsum(tp_reward)
    fp_penalty_cum = np.cumsum(fp_penalty)
    flag_pen_cum = np.cumsum(flag_pen)

    EV_cum = tp_reward_cum - fp_penalty_cum - flag_pen_cum

    TP_cum = np.cumsum(tp_mask)
    FP_cum = np.cumsum(fp_mask)

    # fins de groupes de proba
    if n == 1:
        ends = np.array([0], dtype=np.int64)
    else:
        ends = np.r_[np.where(ps[1:] != ps[:-1])[0], n - 1].astype(np.int64)

    best = None
    for end in ends:
        ev = float(EV_cum[end])
        cand = {
            "end": int(end),
            "thr": float(ps[end]),
            "EV_bps": ev,
            "TP": int(TP_cum[end]),
            "FP": int(FP_cum[end]),
            "tp_reward_sum_bps": float(tp_reward_cum[end]),
            "fp_penalty_sum_bps": float(fp_penalty_cum[end]),
            "flag_penalty_sum_bps": float(flag_pen_cum[end]),
        }
        if best is None:
            best = cand
        else:
            if (
                cand["EV_bps"] > best["EV_bps"] + eps or
                (abs(cand["EV_bps"] - best["EV_bps"]) <= eps and cand["TP"] > best["TP"]) or
                (abs(cand["EV_bps"] - best["EV_bps"]) <= eps and cand["TP"] == best["TP"] and cand["FP"] < best["FP"]) or
                (abs(cand["EV_bps"] - best["EV_bps"]) <= eps and cand["TP"] == best["TP"] and cand["FP"] == best["FP"] and cand["thr"] > best["thr"])
            ):
                best = cand

    if best is None:
        # predict none
        return {"decision_threshold": 1.0, "chosen_end": -1, "why": "no candidates", "EV_bps": 0.0, "TP": 0, "FP": 0}

    return {
        "decision_threshold": float(best["thr"]),
        "chosen_end": int(best["end"]),
        "EV_bps": float(best["EV_bps"]),
        "TP": int(best["TP"]),
        "FP": int(best["FP"]),
        "tp_reward_sum_bps": float(best["tp_reward_sum_bps"]),
        "fp_penalty_sum_bps": float(best["fp_penalty_sum_bps"]),
        "flag_penalty_sum_bps": float(best["flag_penalty_sum_bps"]),
        "why": "ok",
    }

def _eval_ev_fields(y, p, thr, fp_cost_bps, audit_market_toxic, audit_timeout, audit_early_abort, audit_p_thr_ev0, eps: float = 1e-12):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    yb = (p >= float(thr)).astype(np.int64)
    pred_pos = (yb == 1)
    tp = pred_pos & (y == 1)
    fp = pred_pos & (y == 0)

    cost = np.asarray(fp_cost_bps, dtype=np.float64)
    th = np.asarray(audit_p_thr_ev0, dtype=np.float64)
    _check_thr01(th, "EVAL/audit_p_thr_ev0", eps=eps)

    tp_reward = float(np.sum(((1.0 - th) * cost)[tp])) if np.any(tp) else 0.0
    fp_penalty = float(np.sum((th * cost)[fp])) if np.any(fp) else 0.0

    mt = np.asarray(audit_market_toxic, dtype=np.int64)
    to = np.asarray(audit_timeout, dtype=np.int64)
    ab = np.asarray(audit_early_abort, dtype=np.int64)

    flag_pen = float(np.sum(cost[pred_pos & (mt == 1)])) + float(np.sum(cost[pred_pos & (to == 1)])) + float(np.sum(cost[pred_pos & (ab == 1)]))

    ev = tp_reward - fp_penalty - flag_pen

    return {
        "EV_bps": float(ev),
        "tp_reward_sum_bps": float(tp_reward),
        "fp_penalty_sum_bps": float(fp_penalty),
        "flag_penalty_sum_bps": float(flag_pen),
    }

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

def _align_X_to_model(X: np.ndarray, n_model: int) -> tuple[np.ndarray, dict]:
    """Permissif: ajuste X au nombre de features du modèle (truncate/pad)."""
    n = int(X.shape[1])
    if n == n_model:
        return X, {"action": "none", "before": n, "after": n}
    if n > n_model:
        return X[:, :n_model], {"action": "truncate", "before": n, "after": n_model, "dropped": n - n_model}
    pad = np.zeros((X.shape[0], n_model - n), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1), {"action": "pad", "before": n, "after": n_model, "added": n_model - n}

def _require_exact_feature_count(X: np.ndarray, n_model: int, where: str) -> dict:
    """Strict: refuse si mismatch. Retourne juste un petit meta dict."""
    n = int(X.shape[1])
    if n != n_model:
        raise RuntimeError(f"[{where}] feature mismatch: data has {n} cols, model expects {n_model}. Refuse.")
    return {"action": "none", "before": n, "after": n}

def _eval_split_from_scores(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = y.astype(int)
    yb = (p >= thr).astype(int)
    tp = int(np.sum((y == 1) & (yb == 1)))
    fp = int(np.sum((y == 0) & (yb == 1)))
    tn = int(np.sum((y == 0) & (yb == 0)))
    fn = int(np.sum((y == 1) & (yb == 0)))
    n_neg = max(int(np.sum(y == 0)), 1)
    n_pos = max(int(np.sum(y == 1)), 1)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "FPR": float(fp / n_neg),
        "TPR": float(tp / n_pos),
        "precision": float(tp / max(tp + fp, 1)),
    }

def evaluate_ev_sweep(model_uri: str,
                          val_prefix: str, tst_prefix: str,
                          val_audit_prefix: str,
                          tst_audit_prefix: str,
                          audit_cols: List[str],
                          require_audit: bool,
                          allow_feature_mismatch: bool = False,
                          eps: float = 1e-12) -> dict:
    fs = s3fs.S3FileSystem()
    booster = _load_booster_from_tar(fs, model_uri)
    _, feature_names = _load_columns_meta(fs, META_DIR)

    Xv, yv = _load_xy_by_columns(fs, val_prefix, feature_names)
    Xt, yt = _load_xy_by_columns(fs, tst_prefix, feature_names)

    if require_audit:
        _require_csv_prefix(fs, val_audit_prefix, "val_audit")

    aud_cols = [c.strip() for c in audit_cols if c.strip()]
    aud_v = _load_audit_by_cols(fs, val_audit_prefix, aud_cols)

    # --- VAL checks (clean) ---
    n_val = int(len(yv))
    required = ["fp_cost_bps", "audit_market_toxic", "audit_timeout", "audit_early_abort", "audit_p_thr_ev0"]

    missing = [c for c in required if c not in aud_v]
    if missing:
        raise RuntimeError(f"val_audit missing cols: {missing} (audit_cols={aud_cols})")

    for k in required:
        if len(aud_v[k]) != n_val:
            raise RuntimeError(f"[VAL] audit length mismatch for {k}: y={n_val} got={len(aud_v[k])}")

    _check_flags_binary(aud_v, ["audit_market_toxic", "audit_timeout", "audit_early_abort"], "VAL")

    if np.any(aud_v["fp_cost_bps"] < -eps):    
        raise RuntimeError("[VAL] fp_cost_bps has negative values (should be >= 0).")

    # --- TEST audit (optional reporting only) ---
    has_test_audit = True
    try:
        _require_csv_prefix(fs, tst_audit_prefix, "test_audit")
        aud_t = _load_audit_by_cols(fs, tst_audit_prefix, aud_cols)

        # same checks as VAL
        n_test = int(len(yt))
        missing_t = [c for c in required if c not in aud_t]
        if missing_t:
            raise RuntimeError(f"test_audit missing cols: {missing_t} (audit_cols={aud_cols})")

        for k in required:
            if len(aud_t[k]) != n_test:
                raise RuntimeError(f"[TEST] audit length mismatch for {k}: y={n_test} got={len(aud_t[k])}")

        _check_flags_binary(aud_t, ["audit_market_toxic", "audit_timeout", "audit_early_abort"], "TEST")

        if np.any(aud_t["fp_cost_bps"] < -eps):
            raise RuntimeError("[TEST] fp_cost_bps has negative values (should be >= 0).")

    except FileNotFoundError:
        has_test_audit = False
        aud_t = {}
    except Exception as e:
        print(f"[WARN] test_audit invalide, budgets TEST désactivés: {e}", file=sys.stderr)
        has_test_audit = False
        aud_t = {}

    n_model = int(booster.num_features())

    if allow_feature_mismatch:
        Xv, fix_v = _align_X_to_model(Xv, n_model)
        Xt, fix_t = _align_X_to_model(Xt, n_model)
    else:
        fix_v = _require_exact_feature_count(Xv, n_model, "VAL")
        fix_t = _require_exact_feature_count(Xt, n_model, "TEST")

    pv = booster.predict(xgb.DMatrix(Xv))
    pt = booster.predict(xgb.DMatrix(Xt))

    sel = _best_threshold_by_ev(
        y=yv, p=pv,
        fp_cost_bps=aud_v["fp_cost_bps"],
        audit_market_toxic=aud_v["audit_market_toxic"],
        audit_timeout=aud_v["audit_timeout"],
        audit_early_abort=aud_v["audit_early_abort"],
        audit_p_thr_ev0=aud_v["audit_p_thr_ev0"],
        eps=float(eps),
    )
    thr = float(sel["decision_threshold"])

    val_m = _eval_split_from_scores(yv, pv, thr)
    val_ev = _eval_ev_fields(
        yv, pv, thr,
        aud_v["fp_cost_bps"],
        aud_v["audit_market_toxic"],
        aud_v["audit_timeout"],
        aud_v["audit_early_abort"],
        aud_v["audit_p_thr_ev0"],
        eps=float(eps),
    )

    tst_m = _eval_split_from_scores(yt, pt, thr)

    best = {
        "decision_threshold": thr,
        "val": {**val_m, **val_ev},
        "test": tst_m,
        "chosen": sel,
    }
    if has_test_audit:
        tst_ev = _eval_ev_fields(
            yt, pt, thr,
            aud_t["fp_cost_bps"],
            aud_t["audit_market_toxic"],
            aud_t["audit_timeout"],
            aud_t["audit_early_abort"],
            aud_t["audit_p_thr_ev0"],
            eps=float(eps),
        )
        best["test"].update(tst_ev)
    else:
        best["test"].update({"EV_bps": None, "note": "test_audit missing/incomplete => EV not reported on test"})

    return {
        "feature_fix": {"val": fix_v, "test": fix_t},
        "best_by_val_only": best,
        "note": "Selection uses ONLY VAL EV. TEST is reporting only.",
    }

def main():
    args = parse_args()
    fs = s3fs.S3FileSystem()
    _require_csv_prefix(fs, TRN_DIR, "train")
    _require_csv_prefix(fs, VAL_DIR, "validation")
    _require_csv_prefix(fs, TST_DIR, "test")
    if args.require_audit:
        _require_csv_prefix(fs, VAA_DIR, "val_audit")

    label_name, feature_names = _load_columns_meta(fs, META_DIR)

    hp_base = dict(HP_BASE)
    print(f"[meta] label_col = {label_name} | n_features={len(feature_names)}")

    sess = sagemaker.Session(boto3.Session(region_name=REGION))
    s3c = _boto3_client("s3", REGION)

    audit_cols = [c.strip() for c in str(args.audit_cols).split(",") if c.strip()]

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
            val_prefix=VAL_DIR, tst_prefix=TST_DIR,
            val_audit_prefix=VAA_DIR, tst_audit_prefix=TSA_DIR,
            audit_cols=audit_cols,
            require_audit=bool(args.require_audit),
            allow_feature_mismatch=bool(args.allow_feature_mismatch),
            eps=float(args.eps),
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
                    val_prefix=VAL_DIR, tst_prefix=TST_DIR,
                    val_audit_prefix=VAA_DIR, tst_audit_prefix=TSA_DIR,
                    audit_cols=audit_cols,
                    require_audit=bool(args.require_audit),
                    allow_feature_mismatch=bool(args.allow_feature_mismatch),
                    eps=float(args.eps),
                )

                row["sweep"] = _json_sanitize(sweep)
                row["best"]  = _json_sanitize(sweep["best_by_val_only"])
                row["status"] = "ok"

                b = sweep["best_by_val_only"]
                print(
                    f"[BEST@VAL] thr={b['decision_threshold']:.6g} | "
                    f"VAL EV_bps={b['val'].get('EV_bps', float('nan')):.6g} "
                    f"TP={b['val']['TP']} FP={b['val']['FP']} | "
                    f"TEST(report) TP={b['test']['TP']} FP={b['test']['FP']}"
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
        best = max(
            ok_rows,
            key=lambda r: (
                r["best"]["val"].get("EV_bps", -1e30),
                r["best"]["val"]["TP"],
                -r["best"]["val"]["FP"],
                r["best"]["decision_threshold"],
            ),
        )

    final_out = {
        "timestamp": stamp,
        "data_root": DATA_ROOT,
        "hp_base": _json_sanitize(hp_base),
        "selection": {
            "type": "EV",
            "note": "VAL-only maximize EV (TP reward - FP penalty - flag penalties) using audit_p_thr_ev0",
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
                "selection": "VAL-only maximize EV (TP reward - FP penalty - flag penalties) using audit_p_thr_ev0",
                "best": best["best"],
            },
            "inference": {
                "decision_threshold": float(b["decision_threshold"]),
                "proba_semantics": "p = P(Y=1)",
                "invert_output": False,
                "autoflip_allowed": False,
                "ev_fields": {
                    "fp_cost_bps": "fp_cost_bps",
                    "audit_market_toxic": "audit_market_toxic",
                    "audit_timeout": "audit_timeout",
                    "audit_early_abort": "audit_early_abort",
                    "audit_p_thr_ev0": "audit_p_thr_ev0",
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