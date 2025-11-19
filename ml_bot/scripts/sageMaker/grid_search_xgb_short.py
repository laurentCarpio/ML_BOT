# ml_bot/scripts/sageMaker/grid_search_xgb_short.py
import os, json, time, tarfile, tempfile, io, sys, traceback
from itertools import product
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import xgboost as xgb
import s3fs
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve as image_uri_retrieve
from sagemaker.estimator import Estimator

# ========= PARAMS =========
REGION    = "ap-northeast-1"
ROLE_ARN  = "arn:aws:iam::174175447862:role/AmazonSageMaker-ExecutionRole"

# ✅ nouveau dataset v44, short-only
DATA_ROOT       = "s3://tradebot-config-tokyo/data/stage3/v44-shortonly"
MODEL_BASE_S3   = "s3://tradebot-config-tokyo/models/xgb-v44-shortonly"

INSTANCE_TYPE   = "ml.c5.xlarge"
SPOT            = True
MAX_RUN_SEC     = 60 * 20
MAX_WAIT_SEC    = 60 * 40   # > MAX_RUN_SEC (Spot)

HP_BASE = dict(
    tree_method="hist",
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_round=800,
    early_stopping_rounds=100,
    objective="binary:logistic",
    eval_metric="aucpr",
)

HP_BASE.update(dict(
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
))


# ========= PATHS =========
TRN_DIR = f"{DATA_ROOT}/train"
VAL_DIR = f"{DATA_ROOT}/validation"
TRW_DIR = f"{DATA_ROOT}/train_weight"
VAW_DIR = f"{DATA_ROOT}/validation_weight"
META_DIR= f"{DATA_ROOT}/_meta"

# ========= IMAGE =========
XGB_IMAGE = image_uri_retrieve("xgboost", REGION, version="1.7-1")

# ========= HELPERS =========

def _grid_dicts(grid: dict):
    """Produit des dicts hparams à partir de GRID en conservant l’ordre des clés."""
    from itertools import product
    keys = list(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        yield {k: v for k, v in zip(keys, vals)}

def _hps_tag(hps: dict):
    keys = ["scale_pos_weight","max_depth","min_child_weight","max_delta_step",
            "reg_lambda","reg_alpha","gamma","subsample","colsample_bytree"]
    return ",".join([f"{k}={hps[k]}" for k in keys if k in hps])

def _merge_hps(base: dict, overlay: dict):
    h = base.copy()
    h.update(overlay)
    # cast sûrs pour l’estimator
    h["scale_pos_weight"] = float(h["scale_pos_weight"])
    h["max_depth"]        = int(h["max_depth"])
    h["min_child_weight"] = int(h["min_child_weight"])
    h["max_delta_step"]   = int(h["max_delta_step"])
    h["reg_lambda"]       = float(h["reg_lambda"])
    h["reg_alpha"]        = float(h["reg_alpha"])
    h["gamma"]            = float(h["gamma"])
    h["subsample"]        = float(h["subsample"])
    h["colsample_bytree"] = float(h["colsample_bytree"])
    return h

def _json_sanitize(obj):
    """Convertit récursivement np.* et ndarray en types JSON natifs."""
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
    """Toujours sérialisable (fallback sur default=item())."""
    payload = _json_sanitize(payload)
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

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

def _require_csv_prefix(fs: s3fs.S3FileSystem, prefix: str, label: str):
    paths = [f"s3://{p}" for p in fs.glob(prefix.replace("s3://","") + "/part-*.csv")]
    if not paths:
        raise FileNotFoundError(f"[DATA MISSING] Aucun CSV sous {prefix}/part-*.csv ({label})")
    return paths

def read_scale_pos_weight_from_meta():
    fs = s3fs.S3FileSystem()
    meta_path = f"{META_DIR}/train_pos_weight.json"
    try:
        with fs.open(meta_path, "rb") as f:
            return float(json.load(f)["pos_weight"])
    except Exception:
        return None

def make_channels():
    return {
        "train": TrainingInput(TRN_DIR, content_type="text/csv", input_mode="File"),
        "train_weight": TrainingInput(TRW_DIR, content_type="text/csv", input_mode="File"),
        "validation": TrainingInput(VAL_DIR, content_type="text/csv", input_mode="File"),
        "validation_weight": TrainingInput(VAW_DIR, content_type="text/csv", input_mode="File"),
        "meta": TrainingInput(META_DIR, content_type="application/x-directory", input_mode="File"),
    }

def submit_job(hps: dict, session: sagemaker.Session) -> Estimator:
    SRC_DIR  = Path(__file__).parent
    ENTRY    = "train.py"  # doit exister dans SRC_DIR
    if not (SRC_DIR / ENTRY).is_file():
        raise FileNotFoundError(f"train.py introuvable dans {SRC_DIR}")

    est = Estimator(
        image_uri=XGB_IMAGE,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        hyperparameters=hps,
        output_path=f"{MODEL_BASE_S3}",
        sagemaker_session=session,
        enable_sagemaker_metrics=True,
        max_run=MAX_RUN_SEC,
        use_spot_instances=SPOT,
        max_wait=MAX_WAIT_SEC if SPOT else None,
        entry_point=ENTRY,
        source_dir=str(SRC_DIR),
        base_job_name="xgb-short",
        tags=[
            {"Key": "project", "Value": "ml_bot"},
            {"Key": "stage",   "Value": "stage3-xgb-shortonly"},
            {"Key": "side",    "Value": "short"},
        ],
    )
    job_name = f"xgb-short-{time.strftime('%Y%m%d-%H%M%S')}"
    print("Submitting job:", job_name, "| hps:", hps)
    est.fit(make_channels(), job_name=job_name, logs=True)
    return est

def _load_csv_dir(fs: s3fs.S3FileSystem, prefix: str) -> pd.DataFrame:
    paths = [f"s3://{p}" for p in fs.glob(prefix.replace("s3://","") + "/part-*.csv")]
    if not paths:
        raise FileNotFoundError(f"Aucun CSV trouvé sous {prefix}/part-*.csv")
    dfs = [pd.read_csv(p, header=None) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def _maybe_load_weights(fs: s3fs.S3FileSystem, weight_prefix: str):
    try:
        w = _load_csv_dir(fs, weight_prefix).iloc[:, 0].astype(float).to_numpy()
        return w
    except Exception:
        return None

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
        booster.load_model(io.BytesIO(model_bytes))  # xgboost>=2.0
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_bytes); tmp.flush()
            booster.load_model(tmp.name)
    return booster

def evaluate_model(model_uri: str, val_prefix: str, valw_prefix: str):
    fs = s3fs.S3FileSystem()

    # validation des données AVANT toute charge lourde
    _require_csv_prefix(fs, val_prefix, "validation")
    # chargement
    val = _load_csv_dir(fs, val_prefix).to_numpy()
    y = val[:, 0].astype(np.float32)
    X = val[:, 1:].astype(np.float32)
    sample_weight = _maybe_load_weights(fs, valw_prefix)

    booster = _load_booster_from_tar(fs, model_uri)
    dval = xgb.DMatrix(X, label=y)
    yhat = booster.predict(dval)

    # métriques robustes
    try:
        auprc = float(average_precision_score(y, yhat, sample_weight=sample_weight))
    except Exception:
        # fallback sans poids si souci inattendu
        auprc = float(average_precision_score(y, yhat))

    P, R, thr = precision_recall_curve(y, yhat, sample_weight=sample_weight)
    f1s = 2 * P * R / np.maximum(P + R, 1e-12)
    # indices valides (évite nanargmax si tout NaN)
    if np.all(~np.isfinite(f1s)):
        best_thr = 0.5
        f1 = 0.0
    else:
        best_idx = int(np.nanargmax(f1s))
        # mapping indices PR→thr: len(thr) = len(P) - 1
        if best_idx == 0:
            best_thr = float(thr[0]) if len(thr) > 0 else 0.5
        elif best_idx >= len(thr):
            best_thr = float(thr[-1]) if len(thr) > 0 else 0.5
        else:
            best_thr = float(thr[best_idx - 1])
        ybin = (yhat >= best_thr).astype(int)
        f1 = float(f1_score(y, ybin, sample_weight=sample_weight) if sample_weight is not None
                   else f1_score(y, ybin))

    ybin = (yhat >= best_thr).astype(int)
    tp = int(np.sum((y == 1) & (ybin == 1)))
    fp = int(np.sum((y == 0) & (ybin == 1)))
    tn = int(np.sum((y == 0) & (ybin == 0)))
    fn = int(np.sum((y == 1) & (ybin == 0)))
    precision = float(tp / max(tp + fp, 1))
    recall    = float(tp / max(tp + fn, 1))

    pos_rate = float(np.mean(y))
    return dict(
        AUPRC=auprc,
        best_threshold=best_thr,
        F1=f1,
        precision=precision,
        recall=recall,
        TP=tp, FP=fp, TN=tn, FN=fn,
        n=int(len(y)), pos_rate=pos_rate,
    )

def _s3_put_bytes(s3_client, s3_uri: str, data: bytes):
    bkt = s3_uri.split("/", 3)[2]
    key = s3_uri.split("/", 3)[3]
    s3_client.put_object(Bucket=bkt, Key=key, Body=data)

def main():
    # === pré-validation datasets ===
    fs = s3fs.S3FileSystem()
    _require_csv_prefix(fs, TRN_DIR, "train")
    _require_csv_prefix(fs, VAL_DIR, "validation")
    # poids non obligatoires

    # === sessions ===
    sess = sagemaker.Session(boto3.Session(region_name=REGION))
    s3c  = _boto3_client("s3", REGION)

    # === meta pos_weight ===

    SPW0 = float(read_scale_pos_weight_from_meta() or 24.0)
    spw_grid = sorted({round(SPW0 * r, 1) for r in (0.9, 1.0, 1.1)})
    print(f"[meta] scale_pos_weight par défaut: {SPW0} | grille: {spw_grid}")
    
    # GRID pour passe structurelle
    GRID = {
        "scale_pos_weight": spw_grid,        # lu depuis meta ou fallback
        "max_depth":        [7],
        "min_child_weight": [2],
        "max_delta_step":   [2],             # <-- structurelle : 2
        "reg_lambda":       [1.0, 2.0],
        "reg_alpha":        [0.5],           # <-- structurelle : 0.5
        "gamma":            [0.0],
        "subsample":        [0.8],
        "colsample_bytree": [0.8],
    }

    # === Run prioritaire (ancre "structure anti-FP" pour short-only) ===
    PRIORITY_COMBOS = [
        dict(
            scale_pos_weight=SPW0,  # lu depuis DATA_ROOT/_meta/train_pos_weight.json
            max_depth=7,
            min_child_weight=2,
            max_delta_step=2,       # 👈 anti-FP
            reg_lambda=1.0,
            reg_alpha=0.5,          # 👈 L1 pour sparsifier/discipliner
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
    ]

    # === combos ===
    all_grid_dicts = list(_grid_dicts(GRID))
    total = len(all_grid_dicts)
    stamp  = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"{MODEL_BASE_S3}/grid/runs/{stamp}"
    results = []

    print(f"[run] {total} combinaisons à lancer | instance={INSTANCE_TYPE} | spot={bool(SPOT)}")

    # --- 0) Run prioritaire (ancre best) ---
    for hp0 in PRIORITY_COMBOS:
        hps = _merge_hps(HP_BASE, hp0)
        combo_tag = "PRIORITY:" + _hps_tag(hps)
        row = dict(hps=_json_sanitize(hps))
        try:
            t0 = time.time()
            print(f"\n=== {combo_tag} ===")
            print("[fit] soumission du job SageMaker…")
            est = submit_job(hps, sess)
            train_secs = time.time() - t0
            model_uri = est.model_data
            row["model_uri"] = model_uri
            row["train_seconds"] = float(round(train_secs, 2))
            print(f"[fit] terminé en {train_secs:.1f}s | model_uri={model_uri}")

            t1 = time.time()
            print("[eval] lecture des CSV validation + poids (si présents)…")
            metrics = evaluate_model(model_uri=model_uri, val_prefix=VAL_DIR, valw_prefix=VAW_DIR)
            eval_secs = time.time() - t1
            metrics = {k: (float(v) if isinstance(v, (np.floating,))
                           else int(v) if isinstance(v, (np.integer,)) else v)
                       for k, v in metrics.items()}
            row.update(metrics)
            row["eval_seconds"] = float(round(eval_secs, 2))
            row["status"] = "ok"

            print(f"[RESULT] {combo_tag} | "
                  f"AUPRC={metrics['AUPRC']:.5f} | F1={metrics['F1']:.4f} | "
                  f"thr={metrics['best_threshold']:.4f} | "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} | "
                  f"fit={train_secs:.1f}s eval={eval_secs:.1f}s")
        except Exception as e:
            tb = traceback.format_exc()
            row.update({
                "status": "failed",
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": tb[-2000:],
            })
            print(f"[ERROR] {combo_tag} -> {e}", file=sys.stderr)

        # checkpoint
        part_key = f"{run_dir}/combo_priority.json"
        _s3_put_bytes(s3c, part_key, _json_dumps_safe(row))
        print(f"[checkpoint] écrit: s3://{part_key.split('://',1)[1]}")
        results.append(row)

    # --- 1) Boucle grille resserrée ---
    for i, hp0 in enumerate(all_grid_dicts, start=1):
        hps = _merge_hps(HP_BASE, hp0)
        combo_tag = _hps_tag(hps)
        print(f"\n=== [{i}/{total}] {combo_tag} ===")
        row = dict(hps=_json_sanitize(hps))

        try:
            t0 = time.time()
            print("[fit] soumission du job SageMaker…")
            est = submit_job(hps, sess)
            train_secs = time.time() - t0
            model_uri = est.model_data
            row["model_uri"] = model_uri
            row["train_seconds"] = float(round(train_secs, 2))
            print(f"[fit] terminé en {train_secs:.1f}s | model_uri={model_uri}")

            t1 = time.time()
            print("[eval] lecture des CSV validation + poids (si présents)…")
            metrics = evaluate_model(model_uri=model_uri, val_prefix=VAL_DIR, valw_prefix=VAW_DIR)
            eval_secs = time.time() - t1
            metrics = {k: (float(v) if isinstance(v, (np.floating,))
                           else int(v) if isinstance(v, (np.integer,)) else v)
                       for k, v in metrics.items()}
            row.update(metrics)
            row["eval_seconds"] = float(round(eval_secs, 2))
            row["status"] = "ok"

            print(
                f"[RESULT] {combo_tag} | "
                f"AUPRC={metrics['AUPRC']:.5f} | F1={metrics['F1']:.4f} | "
                f"thr={metrics['best_threshold']:.4f} | "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} | "
                f"fit={train_secs:.1f}s eval={eval_secs:.1f}s"
            )

        except Exception as e:
            tb = traceback.format_exc()
            row.update({
                "status": "failed",
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": tb[-2000:],
            })
            print(f"[ERROR] {combo_tag} -> {e}", file=sys.stderr)

        # checkpoint S3 à chaque combo
        part_key = f"{run_dir}/combo_{i:02d}.json"
        _s3_put_bytes(s3c, part_key, _json_dumps_safe(row))
        print(f"[checkpoint] écrit: s3://{part_key.split('://',1)[1]}")
        results.append(row)

    # === résumé final (classement AUPRC sur status=ok) ===
    ok_rows = [r for r in results if r.get("status") == "ok"]
    if ok_rows:
        df = pd.DataFrame(ok_rows).sort_values("AUPRC", ascending=False)
        best = _json_sanitize(df.iloc[0].to_dict())
    else:
        best = None

    final_out = {
        "timestamp": stamp,
        "data_root": DATA_ROOT,
        "grid": _json_sanitize(GRID),
        "hp_base": _json_sanitize(HP_BASE),
        "instance_type": INSTANCE_TYPE,
        "spot": bool(SPOT),
        "results": results,
        "best": best,
    }
    summary_uri = f"{MODEL_BASE_S3}/grid/{stamp}_summary.json"
    _s3_put_bytes(s3c, summary_uri, _json_dumps_safe(final_out))
    print("\nRésumé écrit:", summary_uri)

    if ok_rows:
        show = pd.DataFrame(ok_rows)[["AUPRC","F1","hps","model_uri"]].sort_values("AUPRC", ascending=False).head(10)
        with pd.option_context("display.max_colwidth", 200):
            print("\n=== TOP COMBOS (par AUPRC) ===")
            print(show.to_string(index=False))
    else:
        print("\nAucun combo valide (tous en échec). Consulte les JSON de run pour diagnostiquer.")
                
if __name__ == "__main__":
    main()

def evaluate_split(model_uri, prefix_features, prefix_weights):
    fs = s3fs.S3FileSystem()
    val = _load_csv_dir(fs, prefix_features).to_numpy()
    y = val[:, 0].astype(np.float32)
    X = val[:, 1:].astype(np.float32)
    w = _maybe_load_weights(fs, prefix_weights)
    booster = _load_booster_from_tar(fs, model_uri)
    yhat = booster.predict(xgb.DMatrix(X, label=y))
    return y, yhat, w

def _make_thr_json_from_reports(rpt_df: pd.DataFrame, thr_df: pd.DataFrame,
                                auprc_floor: float = 0.10):
    """Construit le mapping seuils par poche et filtre les poches faibles."""
    ok_keys = set(
        rpt_df.loc[rpt_df["AUPRC"] >= auprc_floor, ["tf", "side_num"]]
              .apply(lambda s: f"{s['tf']}|{int(s['side_num'])}", axis=1)
              .tolist()
    )
    thr_map = {
        f"{row['tf']}|{int(row['side_num'])}": float(row["thr_f1"])
        for _, row in thr_df.iterrows()
        if f"{row['tf']}|{int(row['side_num'])}" in ok_keys
    }
    return thr_map, ok_keys

def eval_test_and_breakdown(best_model_uri: str):
    fs = s3fs.S3FileSystem()
    TEST_DIR = f"{DATA_ROOT}/test"
    TSTW_DIR = f"{DATA_ROOT}/test_weight"

    # 1️⃣ — Évalue globalement
    y, yhat, w = evaluate_split(best_model_uri, TEST_DIR, TSTW_DIR)
    from sklearn.metrics import average_precision_score
    auprc = float(average_precision_score(y, yhat, sample_weight=w)
                  if w is not None else average_precision_score(y, yhat))
    print(f"[TEST] AUPRC={auprc:.5f} (global)")

    # 2️⃣ — Breakdown (AUPRC par tf × side)
    meta_tf = f"{DATA_ROOT}/_meta/splits_parquet/test"
    tables = []
    for p in fs.glob(meta_tf.replace("s3://", "") + "/*.parquet"):
        with fs.open(f"s3://{p}", "rb") as f:
            import pyarrow.parquet as pq
            tables.append(pq.read_table(f))
    mdf = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    mdf = mdf.assign(y=y, yhat=yhat)

    def _summ(g):
        from sklearn.metrics import average_precision_score
        return pd.Series({
            "n": len(g),
            "pos": int(g["y"].sum()),
            "pos_rate": float(g["y"].mean()),
            "AUPRC": float(average_precision_score(g["y"], g["yhat"]))
        })

    rpt = (mdf.groupby(["tf", "side_num"], as_index=False)
             .apply(_summ, include_groups=False)
             .reset_index(drop=True))

    print("\n[TEST] breakdown par (tf, side_num):")
    print(rpt.sort_values("AUPRC", ascending=False).to_string(index=False))

    # 3️⃣ — Calcul des seuils F1 par poche
    from sklearn.metrics import precision_recall_curve, f1_score
    thr_rows = []
    for (tf, side_num), g in mdf.groupby(["tf", "side_num"]):
        P, R, thr = precision_recall_curve(g["y"], g["yhat"])
        f1s = 2 * P * R / np.maximum(P + R, 1e-12)
        if np.all(~np.isfinite(f1s)):
            continue
        best_idx = int(np.nanargmax(f1s))
        if best_idx == 0:
            best_thr = float(thr[0]) if len(thr) > 0 else 0.5
        elif best_idx >= len(thr):
            best_thr = float(thr[-1]) if len(thr) > 0 else 0.5
        else:
            best_thr = float(thr[best_idx - 1])
        f1 = float(f1s[best_idx])
        thr_rows.append(dict(tf=tf, side_num=int(side_num),
                             thr_f1=best_thr, F1=f1))

    thr_df = pd.DataFrame(thr_rows)
    print("\n[TEST] seuils par poche (F1-max):")
    print(thr_df.sort_values("F1", ascending=False).to_string(index=False))

    # 4️⃣ — Construire le JSON de seuils et le pousser sur S3
    thr_map, ok_keys = _make_thr_json_from_reports(rpt, thr_df, auprc_floor=0.10)

    payload = {
        "model_uri": best_model_uri,
        "created_at": int(time.time()),
        "policy": {"auprc_floor": 0.10, "criterion": "thr_f1"},
        "enabled_pockets": sorted(list(ok_keys)),
        "thresholds": thr_map
    }

    out_uri = f"{MODEL_BASE_S3}/deploy/thresholds_{int(time.time())}.json"
    _s3_put_bytes(_boto3_client("s3", REGION), out_uri, _json_dumps_safe(payload))
    print(f"[deploy] thresholds map écrit: {out_uri}")

# === NEW: seuils par poche -> JSON de déploiement ============================
def _per_pocket_reports(model_uri: str):
    """
    Recalcule le breakdown TEST (AUPRC par (tf, side)) + seuil F1 par poche.
    On réutilise les chemins globaux DATA_ROOT et META parquet.
    """
    import pyarrow.parquet as pq
    import s3fs

    fs = s3fs.S3FileSystem()

    # 1) Charger TEST (features+poids) et prédire
    TEST_DIR = f"{DATA_ROOT}/test"
    TSTW_DIR = f"{DATA_ROOT}/test_weight"
    y, yhat, w = evaluate_split(model_uri, TEST_DIR, TSTW_DIR)

    # 2) Lire le meta parquet TEST pour récupérer tf/side_num alignés
    meta_tf = f"{DATA_ROOT}/_meta/splits_parquet/test"
    tables = []
    for p in fs.glob(meta_tf.replace("s3://","") + "/*.parquet"):
        with fs.open(f"s3://{p}", "rb") as f:
            tables.append(pq.read_table(f))
    mdf = pd.concat([t.to_pandas() for t in tables], ignore_index=True)

    # 3) Assembler y/yhat et calculer AUPRC par poche
    mdf = mdf.assign(y=y, yhat=yhat)
    from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

    def _summ(g: pd.DataFrame) -> pd.Series:
        # AUPRC
        auprc = float(average_precision_score(g["y"], g["yhat"]))
        # seuil F1-max
        P, R, thr = precision_recall_curve(g["y"].to_numpy(), g["yhat"].to_numpy())
        f1s = 2 * P * R / np.maximum(P + R, 1e-12)
        if np.all(~np.isfinite(f1s)):
            thr_f1 = 0.5
            f1_best = 0.0
        else:
            k = int(np.nanargmax(f1s))
            if k == 0:
                thr_f1 = float(thr[0]) if len(thr) > 0 else 0.5
            elif k >= len(thr):
                thr_f1 = float(thr[-1]) if len(thr) > 0 else 0.5
            else:
                thr_f1 = float(thr[k - 1])
            ybin = (g["yhat"].to_numpy() >= thr_f1).astype(int)
            f1_best = float(f1_score(g["y"].to_numpy(), ybin))
        return pd.Series({
            "n": int(len(g)),
            "pos": int(g["y"].sum()),
            "pos_rate": float(g["y"].mean()),
            "AUPRC": auprc,
            "thr_f1": thr_f1,
            "F1": f1_best,
        })

    # ⚠️ évite le FutureWarning: sélectionne explicitement les colonnes
    grp = (mdf[["tf", "side_num", "y", "yhat"]]
           .groupby(["tf", "side_num"], as_index=False)
           .apply(_summ, include_groups=False))
    rpt_df = grp.reset_index(drop=True)  # colonnes: tf, side_num, n,pos,pos_rate,AUPRC,thr_f1,F1

    # thr_df = uniquement les colonnes nécessaires au mapping de seuils
    thr_df = rpt_df[["tf","side_num","thr_f1"]].copy()
    return rpt_df, thr_df

def make_thresholds_json_from_breakdown(
    model_uri: str,
    auprc_floor: float = 0.12,
    policy: str = "thr_f1",       # "thr_f1" ou "prec_floor" (à ajouter plus tard)
    prec_floor: float | None = None
) -> str:
    """
    Calcule les rapports TEST par poche, filtre les poches faibles (AUPRC < floor),
    puis écrit un JSON de déploiement {enabled_pockets, thresholds{tf|side: thr}} sous:
      {MODEL_BASE_S3}/deploy/thresholds_{unix}.json

    Retourne l’URI S3 créé.
    """
    import time
    s3c = _boto3_client("s3", REGION)

    rpt_df, thr_df = _per_pocket_reports(model_uri)
    thr_map, ok_keys = _make_thr_json_from_reports(rpt_df, thr_df, auprc_floor=auprc_floor)

    now = int(time.time())
    payload = {
        "model_uri": model_uri,
        "created_at": now,
        "policy": {
            "auprc_floor": auprc_floor,
            "criterion": policy,
            **({"prec_floor": float(prec_floor)} if (policy == "prec_floor" and prec_floor is not None) else {})
        },
        "enabled_pockets": sorted(list(ok_keys)),
        "thresholds": {k: float(v) for k, v in sorted(thr_map.items())},
    }

    out_uri = f"{MODEL_BASE_S3}/deploy/thresholds_{now}.json"
    _s3_put_bytes(s3c, out_uri, _json_dumps_safe(payload))
    print(f"[deploy] thresholds écrit: {out_uri}")
    # petit aperçu
    if payload["enabled_pockets"]:
        print("[deploy] enabled_pockets:", ", ".join(payload["enabled_pockets"]))
    else:
        print("[deploy] ⚠️ aucune poche activée (AUPRC < floor partout).")
    return out_uri