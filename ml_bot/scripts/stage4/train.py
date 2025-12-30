# ml_bot/scripts/sageMaker/train.py
import os, glob, json, argparse, sys
from pathlib import Path
import numpy as np
import xgboost as xgb

def _load_csv_matrix(dir_path, delim=","):
    parts = sorted(glob.glob(os.path.join(dir_path, "part-*.csv")))
    if not parts:
        raise RuntimeError(f"Aucun fichier part-*.csv dans {dir_path}")
    xs, ys = [], []
    for p in parts:
        arr = np.loadtxt(p, delimiter=delim)
        # colonnes GO stage3: [Y, feat1, feat2, ...]  (poids à part)
        y = arr[:, 0]
        X = arr[:, 1:]
        ys.append(y); xs.append(X)
    X = np.vstack(xs).astype("float32")
    y = np.concatenate(ys).astype("float32")
    return X, y

def _load_weights(dir_path):
    if not dir_path or not os.path.isdir(dir_path):
        return None
    parts = sorted(glob.glob(os.path.join(dir_path, "part-*.csv")))
    if not parts:
        return None
    ws = []
    for p in parts:
        w = np.loadtxt(p, delimiter=",")
        if w.ndim == 2 and w.shape[1] == 1:
            w = w.ravel()
        ws.append(w.astype("float32"))
    w_all = np.concatenate(ws)
    return w_all

def parse_args():
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--objective", type=str, default="binary:logistic")
    p.add_argument("--eval_metric", type=str, default="aucpr")
    p.add_argument("--tree_method", type=str, default="hist")
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--scale_pos_weight", type=float, default=1.0)
    p.add_argument("--num_round", type=int, default=800)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--max_delta_step", type=float, default=0.0)
    # regularization (the ones that were missing)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)

    # allow future extra args without crashing
    args, _unknown = p.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    # weights come as separate SageMaker channels, not subfolders under train/validation
    trw_dir   = os.environ.get("SM_CHANNEL_TRAIN_WEIGHT", "/opt/ml/input/data/train_weight")
    vaw_dir   = os.environ.get("SM_CHANNEL_VALIDATION_WEIGHT", "/opt/ml/input/data/validation_weight")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    Xtr, ytr = _load_csv_matrix(train_dir, delim=",")
    Xva, yva = _load_csv_matrix(val_dir,   delim=",")

    wtr = _load_weights(trw_dir)
    wva = _load_weights(vaw_dir)

    print(f"[data] Xtr={Xtr.shape} ytr={ytr.shape} wtr={'None' if wtr is None else wtr.shape}")
    print(f"[data] Xva={Xva.shape} yva={yva.shape} wva={'None' if wva is None else wva.shape}")

    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)

    params = {
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "tree_method": args.tree_method,
        "eta": args.eta,
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "scale_pos_weight": float(args.scale_pos_weight),
        "max_delta_step": float(args.max_delta_step),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
        "gamma": float(args.gamma),
    }

    effective = dict(
        **params,
        num_round=int(args.num_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    print("[HP] effectifs:", json.dumps(effective, indent=2))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(model_dir, "hparams.json"), "w") as f:
        json.dump(effective, f, indent=2)

    evals = [(dtrain, "train"), (dvalid, "validation")]
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.num_round),
        evals=evals,
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=True,
    )

    # trace best iteration / score
    best_iter = getattr(bst, "best_iteration", None)
    best_score = getattr(bst, "best_score", None)
    meta = dict(best_iteration=best_iter, best_score=best_score)
    try:
        with open(os.path.join(model_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[warn] impossible d'écrire training_meta.json: {e}")

    # save model
    bst.save_model(os.path.join(model_dir, "model.json"))
    print("[save] model.json écrit dans", model_dir)