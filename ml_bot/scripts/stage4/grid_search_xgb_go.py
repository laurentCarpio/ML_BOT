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
TSW_DIR = f"{DATA_ROOT}/test_weight"

TRA_DIR = f"{DATA_ROOT}/train_audit"
VAA_DIR = f"{DATA_ROOT}/val_audit"
TSA_DIR = f"{DATA_ROOT}/test_audit"

META_DIR = f"{DATA_ROOT}/_meta"

INSTANCE_TYPE   = "ml.c5.xlarge"
SPOT            = True
MAX_RUN_SEC     = 60 * 20
MAX_WAIT_SEC    = 60 * 40

# === Stage4: sweep budgets (VAL-only decision) ===
# B = budget en bps (somme fp_cost_bps sur les faux positifs prédits)
DEFAULT_B_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]

DEFAULT_K_TOXIC_SWEEP   = [0, 1]
DEFAULT_K_TIMEOUT_SWEEP = [10, 50, 200]   # exemple, ajuste
DEFAULT_K_ABORT_SWEEP   = [10, 50, 200]   # exemple, ajuste

HP_BASE: Dict[str, Any] = dict(
    tree_method="hist",
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_round=400,
    early_stopping_rounds=50,
    objective="binary:logistic",
    eval_metric="logloss",
    reg_lambda=5.0,
    reg_alpha=0.2,
    gamma=0.5,
    max_depth=4,
    min_child_weight=5,
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

    ap.add_argument("--b-sweep", default=",".join(str(x) for x in DEFAULT_B_SWEEP),
                    help="Liste de budgets B (bps) séparés par virgules. Ex: 0,0.5,1,2,5")
    
    ap.add_argument("--k-toxic-sweep", default=",".join(str(x) for x in DEFAULT_K_TOXIC_SWEEP))
    ap.add_argument("--k-timeout-sweep", default=",".join(str(x) for x in DEFAULT_K_TIMEOUT_SWEEP))
    ap.add_argument("--k-abort-sweep", default=",".join(str(x) for x in DEFAULT_K_ABORT_SWEEP))
    
    ap.add_argument("--audit-cols", default="fp_cost_bps,audit_early_abort,audit_timeout,audit_market_toxic",
                    help="Colonnes attendues dans *_audit (no header), ordre important.")
    
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

def _load_train_balance(fs: s3fs.S3FileSystem, meta_dir: str):
    path = f"{meta_dir}/train_class_balance.json"
    try:
        bal = _read_json_s3(fs, path)
        spw = bal.get("scale_pos_weight_raw", None)
        return bal, float(spw) if spw is not None else None
    except Exception:
        return None, None

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

def _best_threshold_under_budgets(
    y: np.ndarray,
    p: np.ndarray,
    fp_cost_bps: np.ndarray,
    audit_market_toxic: np.ndarray,
    audit_timeout: np.ndarray,
    audit_early_abort: np.ndarray,
    B: float,
    K_toxic: int,
    K_timeout: int,
    K_abort: int,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Sélectionne un seuil sur VAL (VAL-only) qui maximise TP sous:
      sum(fp_cost_bps sur FP prédits) <= B
      count(audit_market_toxic==1 sur FP prédits) <= K_toxic
    Gestion des ties: on évalue par "score group" (même proba).
    """
    y = y.astype(np.int64, copy=False)
    p = p.astype(np.float64, copy=False)
    fp_cost_bps = fp_cost_bps.astype(np.float64, copy=False)
    mt = audit_market_toxic.astype(np.float64, copy=False)
    to = audit_timeout.astype(np.float64, copy=False)
    ab = audit_early_abort.astype(np.float64, copy=False)

    n = int(len(y))
    if n == 0:
        return {"decision_threshold": 1.0, "chosen_end": -1, "why": "empty"}

    order = np.argsort(-p, kind="mergesort")  # stable
    ps = p[order]
    ys = y[order]
    cs = fp_cost_bps[order]
    mt = mt[order]
    to = to[order]
    ab = ab[order]

    fp_mask = (ys == 0)
    tp_cum = np.cumsum(ys == 1)
    fp_cost_cum = np.cumsum(cs * fp_mask)

    tox_cum  = np.cumsum(fp_mask & (mt.astype(np.int64, copy=False) == 1))
    tout_cum = np.cumsum(fp_mask & (to.astype(np.int64, copy=False) == 1))
    abrt_cum = np.cumsum(fp_mask & (ab.astype(np.int64, copy=False) == 1))
    neg_cum = np.cumsum(fp_mask)  # here prefix is predicted positive => neg_cum == FP_cum

    # group ends where score changes
    change = np.ones_like(ps, dtype=bool)
    change[1:] = (ps[1:] != ps[:-1])
    group_ends = np.where(change)[0]
    # group_ends are group starts; we need ends:
    # start indices: group_ends; end indices: next_start-1, last=n-1
    starts = group_ends
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = n - 1

    best = None
    for end in ends:
        cost = float(fp_cost_cum[end])
        tox  = int(tox_cum[end])
        tout = int(tout_cum[end])
        abrt = int(abrt_cum[end])
        if (cost <= B + eps) and (tox <= K_toxic) and (tout <= K_timeout) and (abrt <= K_abort):
            cand = {
                "end": int(end),
                "thr": float(ps[end]),
                "TP": int(tp_cum[end]),
                "FP": int(neg_cum[end]),
                "fp_cost_sum_bps": cost,
                "fp_toxic": tox,
                "fp_timeout": tout,
                "fp_abort": abrt,
            }
            if best is None:
                best = cand
            else:
                # objective: max TP, then min cost, then min fp_toxic, then min FP, then higher threshold
                if (
                    cand["TP"] > best["TP"] or
                    (cand["TP"] == best["TP"] and cand["fp_cost_sum_bps"] < best["fp_cost_sum_bps"] - eps) or
                    (cand["TP"] == best["TP"] and abs(cand["fp_cost_sum_bps"] - best["fp_cost_sum_bps"]) <= eps and cand["fp_toxic"] < best["fp_toxic"]) or
                    (cand["TP"] == best["TP"] and abs(cand["fp_cost_sum_bps"] - best["fp_cost_sum_bps"]) <= eps and cand["fp_toxic"] == best["fp_toxic"] and cand["fp_timeout"] < best["fp_timeout"]) or
                    (cand["TP"] == best["TP"] and abs(cand["fp_cost_sum_bps"] - best["fp_cost_sum_bps"]) <= eps and cand["fp_toxic"] == best["fp_toxic"] and cand["fp_timeout"] == best["fp_timeout"] and cand["fp_abort"] < best["fp_abort"]) or
                    (cand["TP"] == best["TP"] and abs(cand["fp_cost_sum_bps"] - best["fp_cost_sum_bps"]) <= eps and cand["fp_toxic"] == best["fp_toxic"] and cand["fp_timeout"] == best["fp_timeout"] and cand["fp_abort"] == best["fp_abort"] and cand["FP"] < best["FP"]) or
                    (cand["TP"] == best["TP"] and abs(cand["fp_cost_sum_bps"] - best["fp_cost_sum_bps"]) <= eps and cand["fp_toxic"] == best["fp_toxic"] and cand["fp_timeout"] == best["fp_timeout"] and cand["fp_abort"] == best["fp_abort"] and cand["FP"] == best["FP"] and cand["thr"] > best["thr"])
                ):
                    best = cand

    if best is None:
        # Aucun seuil non-trivial ne respecte le budget -> seuil = 1.0 => predict none
        return {
            "decision_threshold": 1.0,
            "chosen_end": -1,
            "TP": 0,
            "FP": 0,
            "fp_cost_sum_bps": 0.0,
            "fp_toxic": 0,
            "fp_timeout": 0,
            "fp_abort": 0,
            "why": "no feasible threshold under budgets",
        }

    return {
        "decision_threshold": float(best["thr"]),
        "chosen_end": int(best["end"]),
        "TP": int(best["TP"]),
        "FP": int(best["FP"]),
        "fp_cost_sum_bps": float(best["fp_cost_sum_bps"]),
        "fp_toxic": int(best["fp_toxic"]),
        "fp_timeout": int(best["fp_timeout"]),
        "fp_abort": int(best["fp_abort"]),
        "why": "ok",
    }

def _eval_budget_fields(y, p, thr, fp_cost_bps, audit_market_toxic, audit_timeout, audit_early_abort):
    y = y.astype(np.int64, copy=False)
    yb = (p >= thr).astype(np.int64)
    fp = (y == 0) & (yb == 1)

    cost = float(np.sum(fp_cost_bps[fp])) if fp.any() else 0.0
    tox  = int(np.sum(audit_market_toxic[fp].astype(np.int64, copy=False) == 1)) if fp.any() else 0
    tout = int(np.sum(audit_timeout[fp].astype(np.int64, copy=False) == 1)) if fp.any() else 0
    abrt = int(np.sum(audit_early_abort[fp].astype(np.int64, copy=False) == 1)) if fp.any() else 0

    return {"fp_cost_sum_bps": cost, "fp_toxic": tox, "fp_timeout": tout, "fp_abort": abrt}

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

def evaluate_budget_sweep(model_uri: str,
                          val_prefix: str, tst_prefix: str,
                          val_audit_prefix: str,
                          tst_audit_prefix: str,
                          audit_cols: List[str],
                          B_sweep: List[float],
                          K_toxic_sweep: List[int],
                          K_timeout_sweep: List[int],
                          K_abort_sweep: List[int],
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
    required = ["fp_cost_bps", "audit_market_toxic", "audit_timeout", "audit_early_abort"]

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

    rows = []
    for B in B_sweep:
        for Kt in K_toxic_sweep:
            for Kto in K_timeout_sweep:
                for Ka in K_abort_sweep:
                    sel = _best_threshold_under_budgets(
                        y=yv, p=pv,
                        fp_cost_bps=aud_v["fp_cost_bps"],
                        audit_market_toxic=aud_v["audit_market_toxic"],
                        audit_timeout=aud_v["audit_timeout"],
                        audit_early_abort=aud_v["audit_early_abort"],
                        B=float(B),
                        K_toxic=int(Kt),
                        K_timeout=int(Kto),
                        K_abort=int(Ka),
                        eps=float(eps),
                    )
                    thr = float(sel["decision_threshold"])
                    val_m = _eval_split_from_scores(yv, pv, thr)
                    val_budget = _eval_budget_fields(
                        yv, pv, thr,
                        aud_v["fp_cost_bps"],
                        aud_v["audit_market_toxic"],
                        aud_v["audit_timeout"],
                        aud_v["audit_early_abort"],
                    )
                    tst_m = _eval_split_from_scores(yt, pt, thr)

                    row = {
                        "B_bps": float(B),
                        "K_toxic": int(Kt),
                        "K_timeout": int(Kto),
                        "K_abort": int(Ka),
                        "decision_threshold": float(thr),
                        "val": {**val_m, **val_budget},
                        "test": tst_m,
                    }
                    if has_test_audit:
                        tst_budget = _eval_budget_fields(
                            yt, pt, thr,
                            aud_t["fp_cost_bps"],
                            aud_t["audit_market_toxic"],
                            aud_t["audit_timeout"],
                            aud_t["audit_early_abort"],
                        )
                        row["test"].update(tst_budget)
                    else:
                        row["test"].update({"fp_cost_sum_bps": None, "fp_toxic": None, "fp_timeout": None, "fp_abort": None,
                                            "note": "test_audit missing/incomplete => budgets not reported on test"})
                    rows.append(row)

    # Sélection VAL-only: max TP, puis min fp_cost_sum_bps, puis min fp_toxic, puis min FP, puis thr plus haut

    if not rows:
        raise RuntimeError("Budget sweep vide: vérifie --b-sweep / --k-*-sweep")

    best = min(rows, key=lambda r: (
        -r["val"]["TP"],
        r["val"]["fp_cost_sum_bps"],
        r["val"]["fp_toxic"],
        r["val"]["fp_timeout"],
        r["val"]["fp_abort"],
        r["val"]["FP"],
        -r["decision_threshold"],
    ))

    return {
        "feature_fix": {"val": fix_v, "test": fix_t},
        "sweep": rows,
        "best_by_val_only": best,
        "note": "Selection uses ONLY VAL constraints/metrics. TEST is reporting only.",
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
    _, spw = _load_train_balance(fs, META_DIR)

    hp_base = dict(HP_BASE)
    if spw is not None:
        hp_base["scale_pos_weight"] = float(spw)
        print(f"[meta] scale_pos_weight = {spw:.6g} (from train_class_balance.json)")
    print(f"[meta] label_col = {label_name} | n_features={len(feature_names)}")

    sess = sagemaker.Session(boto3.Session(region_name=REGION))
    s3c = _boto3_client("s3", REGION)

    B_sweep = _parse_list(args.b_sweep, float)
    K_toxic_sweep   = _parse_list(args.k_toxic_sweep, int)
    K_timeout_sweep = _parse_list(args.k_timeout_sweep, int)
    K_abort_sweep   = _parse_list(args.k_abort_sweep, int)

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

        sweep = evaluate_budget_sweep(
            model_uri=row["model_uri"],
            val_prefix=VAL_DIR, tst_prefix=TST_DIR,
            val_audit_prefix=VAA_DIR, tst_audit_prefix=TSA_DIR,
            audit_cols=audit_cols,
            B_sweep=B_sweep,
            K_toxic_sweep=K_toxic_sweep,
            K_timeout_sweep=K_timeout_sweep,
            K_abort_sweep=K_abort_sweep,
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

                sweep = evaluate_budget_sweep(
                    model_uri=row["model_uri"],
                    val_prefix=VAL_DIR, tst_prefix=TST_DIR,
                    val_audit_prefix=VAA_DIR, tst_audit_prefix=TSA_DIR,
                    audit_cols=audit_cols,
                    B_sweep=B_sweep,
                    K_toxic_sweep=K_toxic_sweep,
                    K_timeout_sweep=K_timeout_sweep,
                    K_abort_sweep=K_abort_sweep,
                    require_audit=bool(args.require_audit),
                    allow_feature_mismatch=bool(args.allow_feature_mismatch),
                    eps=float(args.eps)
                )

                row["sweep"] = _json_sanitize(sweep)
                row["best"]  = _json_sanitize(sweep["best_by_val_only"])
                row["status"] = "ok"

                b = sweep["best_by_val_only"]
                print(
                    f"[BEST@VAL] B={b['B_bps']} K={b['K_toxic']} thr={b['decision_threshold']:.6g} | "
                    f"VAL TP={b['val']['TP']} FP={b['val']['FP']} "
                    f"fp_cost_sum_bps={b['val']['fp_cost_sum_bps']:.6g} fp_toxic={b['val']['fp_toxic']} | "
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
        best = min(
        ok_rows,
        key=lambda r: (
            -r["best"]["val"]["TP"],              # max TP
            r["best"]["val"]["fp_cost_sum_bps"],  # min cost
            r["best"]["val"]["fp_toxic"],         # min toxic
            r["best"]["val"]["fp_timeout"],       # min timeout
            r["best"]["val"]["fp_abort"],         # min abort
            r["best"]["val"]["FP"],               # min FP
            -r["best"]["decision_threshold"],     # max thr
        )
    )

    final_out = {
        "timestamp": stamp,
        "data_root": DATA_ROOT,
        "hp_base": _json_sanitize(hp_base),
        "budget_sweep": {
            "B_bps": B_sweep,
            "K_toxic": K_toxic_sweep,
            "K_timeout": K_timeout_sweep,
            "K_abort": K_abort_sweep,
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
                "selection": "VAL-only budgeted sweep (sum(fp_cost_bps on FP) <= B, FP flags caps)",
                "best": best["best"],
                "sweep": best["sweep"]["sweep"],
            },
            "inference": {
                "decision_threshold": float(b["decision_threshold"]),
                "proba_semantics": "p = P(Y=1)",
                "invert_output": False,
                "autoflip_allowed": False,
                "budget_constraints": {
                "B_bps": float(b["B_bps"]),
                "K_toxic": int(b["K_toxic"]),
                "K_timeout": int(b["K_timeout"]),
                "K_abort": int(b["K_abort"]),
                "fields": {
                    "fp_cost_bps": "fp_cost_bps",
                    "audit_market_toxic": "audit_market_toxic",
                    "audit_timeout": "audit_timeout",
                    "audit_early_abort": "audit_early_abort",
                },
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