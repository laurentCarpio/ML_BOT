# ml_bot/scripts/sageMaker/train.py
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import multiprocessing
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import xgboost as xgb


def _list_parts(dir_path: str) -> List[str]:
    parts = sorted(glob.glob(os.path.join(dir_path, "part-*.csv")))
    if not parts:
        raise RuntimeError(f"Aucun fichier part-*.csv dans {dir_path}")
    return parts


def _check_y_binary(y: np.ndarray, name: str):
    # tolère float32 mais impose valeurs {0,1}
    u = np.unique(y)
    if not np.all(np.isin(u, [0.0, 1.0])):
        # affiche un aperçu sans spammer
        preview = u[:10]
        raise RuntimeError(f"{name} labels must be 0/1, got unique={preview} (n_unique={len(u)})")


def _load_csv_matrix(dir_path: str, delim: str = ",") -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage3_GO:
      - CSVs sans header
      - colonnes: [Y, f0, f1, ...]
    Lecture via pandas (plus robuste que np.loadtxt pour gros fichiers).
    """
    parts = _list_parts(dir_path)
    xs, ys = [], []

    for p in parts:
        # engine="c" = rapide; header=None car pas d'entête
        # dtype float32 réduit la RAM; low_memory=False évite des surprises de dtype
        df = pd.read_csv(
            p,
            header=None,
            sep=delim,
            engine="c",
            dtype=np.float32,
            low_memory=False,
        )

        if df.shape[1] < 2:
            raise RuntimeError(f"CSV invalide (attendu [Y]+features) : {p} shape={df.shape}")

        y = df.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        X = df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)

        ys.append(y)
        xs.append(X)

    X_all = np.vstack(xs).astype(np.float32, copy=False)
    y_all = np.concatenate(ys).astype(np.float32, copy=False)
    return X_all, y_all


def _load_weights(dir_path: str, delim: str = ",") -> Optional[np.ndarray]:
    """
    Charge un channel de poids (part-*.csv). Doit être 1 colonne.
    Retourne None si absent.
    """
    if not dir_path or not os.path.isdir(dir_path):
        return None

    parts = sorted(glob.glob(os.path.join(dir_path, "part-*.csv")))
    if not parts:
        return None

    ws = []
    for p in parts:
        df = pd.read_csv(
            p,
            header=None,
            sep=delim,
            engine="c",
            dtype=np.float32,
            low_memory=False,
        )
        if df.shape[1] != 1:
            raise RuntimeError(f"Poids invalide (attendu 1 colonne) : {p} shape={df.shape}")
        w = df.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        ws.append(w)

    w_all = np.concatenate(ws).astype(np.float32, copy=False)
    return w_all


def parse_args():
    p = argparse.ArgumentParser()

    # core
    p.add_argument("--objective", type=str, default="binary:logistic")
    # ⚠️ cohérent avec ton grid_search (HP_BASE.eval_metric="logloss")
    p.add_argument("--eval_metric", type=str, default="logloss")
    p.add_argument("--tree_method", type=str, default="hist")
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--num_round", type=int, default=800)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--max_delta_step", type=float, default=0.0)

    # regularization
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)

    args, unknown = p.parse_known_args()

    # refuse explicitement scale_pos_weight (évite le "silent ignore")
    banned = {"--scale_pos_weight", "--scale-pos-weight"}
    if any(u.split("=", 1)[0] in banned for u in unknown):
        raise SystemExit("Argument interdit: --scale_pos_weight. Utiliser uniquement les sample weights (train_weight).")

    return args


def _ensure_weight_len(w: Optional[np.ndarray], n: int, name: str) -> Optional[np.ndarray]:
    if w is None:
        return None
    if len(w) != n:
        raise RuntimeError(f"{name} length mismatch: len(w)={len(w)} vs n_rows={n}")
    return w


if __name__ == "__main__":
    args = parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    trw_dir   = os.environ.get("SM_CHANNEL_TRAIN_WEIGHT", "/opt/ml/input/data/train_weight")
    vaw_dir   = os.environ.get("SM_CHANNEL_VALIDATION_WEIGHT", "/opt/ml/input/data/validation_weight")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # --- load data ---
    Xtr, ytr = _load_csv_matrix(train_dir, delim=",")
    Xva, yva = _load_csv_matrix(val_dir,   delim=",")

    # sanity checks labels
    _check_y_binary(ytr, "train")
    _check_y_binary(yva, "validation")

    wtr = _load_weights(trw_dir)
    wva = _load_weights(vaw_dir)
    wtr = _ensure_weight_len(wtr, len(ytr), "train_weight")
    wva = _ensure_weight_len(wva, len(yva), "validation_weight")

    print(f"[data] Xtr={Xtr.shape} ytr={ytr.shape} wtr={'None' if wtr is None else wtr.shape}")
    print(f"[data] Xva={Xva.shape} yva={yva.shape} wva={'None' if wva is None else wva.shape}")
    print(f"[data] n_features={Xtr.shape[1]}")

    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)

    params = {
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "tree_method": args.tree_method,
        "eta": float(args.eta),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "max_delta_step": float(args.max_delta_step),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
        "gamma": float(args.gamma),
    }
    
    # Threads: SageMaker expose souvent OMP_NUM_THREADS
    nthread = int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count() or 4))
    params["nthread"] = max(1, nthread)
    print(f"[threads] nthread={params['nthread']} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")

    effective = dict(
        **params,
        num_round=int(args.num_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    print("[HP] effectifs:", json.dumps(effective, indent=2))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(model_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(effective, f, indent=2)

    evals = [(dtrain, "train"), (dvalid, "validation")]
    evals_result = {}

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.num_round),
        evals=evals,
        early_stopping_rounds=int(args.early_stopping_rounds),
        evals_result=evals_result,
        verbose_eval=True,
    )

    meta = {
        "best_iteration": int(getattr(bst, "best_iteration", 0)) if getattr(bst, "best_iteration", None) is not None else None,
        "best_score": float(getattr(bst, "best_score", 0.0)) if getattr(bst, "best_score", None) is not None else None,
        "evals_result": evals_result,
    }
    with open(os.path.join(model_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    bst.save_model(os.path.join(model_dir, "model.json"))
    print("[save] model.json écrit dans", model_dir)