# ml_bot/scripts/sageMaker/post_eval_xgb.py
# -*- coding: utf-8 -*-
import argparse, json, os, tarfile, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import fsspec
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")

def open_any(uri: str, mode="rb"):
    fs, _, paths = fsspec.get_fs_token_paths(uri)
    return fs, fs.open(paths[0], mode)

def load_csv(uri: str, header=None):
    if is_s3(uri):
        with fsspec.open(uri, "rb") as f:
            return pd.read_csv(f, header=header)
    else:
        return pd.read_csv(uri, header=header)

def ensure_parent_exists(uri: str):
    if is_s3(uri):
        # s3: nothing to mkdir
        return
    else:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)

def save_json(obj: dict, uri: str):
    ensure_parent_exists(uri)
    data = (json.dumps(obj, indent=2) + "\n").encode("utf-8")
    if is_s3(uri):
        with fsspec.open(uri, "wb") as f:
            f.write(data)
    else:
        with open(uri, "wb") as f:
            f.write(data)

def maybe_extract_model_tar(model_uri: str) -> str:
    """
    Si model_uri est un .tar.gz (artefact SageMaker), on l'extrait en temp
    et on retourne le chemin du fichier modèle (model.json / xgboost_model / model.bin).
    Sinon on retourne model_uri tel quel.
    """
    if not model_uri.endswith(".tar.gz"):
        return model_uri

    with fsspec.open(model_uri, "rb") as f_in, tempfile.TemporaryDirectory() as td:
        tmp_tar = Path(td) / "model.tar.gz"
        with open(tmp_tar, "wb") as f_out:
            f_out.write(f_in.read())
        with tarfile.open(tmp_tar, "r:gz") as tar:
            tar.extractall(td)
        # fichiers possibles selon scripts/containers
        candidates = [
            "model.json",
            "xgboost_model",          # nom par défaut du conteneur built-in
            "model.bin",
        ]
        # cherche récursivement
        found = None
        for root, _, files in os.walk(td):
            for nm in files:
                if nm in candidates:
                    found = Path(root) / nm
                    break
            if found:
                break
        if not found:
            raise RuntimeError("Aucun fichier modèle reconnu trouvé dans le tar.gz")
        # on recopie dans un vrai fichier temp persistant hors context manager
        fd, final_path = tempfile.mkstemp(prefix="xgb_model_", suffix=found.suffix or "")
        os.close(fd)
        with open(found, "rb") as src, open(final_path, "wb") as dst:
            dst.write(src.read())
        return final_path  # chemin local

def load_booster(model_uri: str) -> xgb.Booster:
    model_path = maybe_extract_model_tar(model_uri)
    booster = xgb.Booster()
    # xgboost sait charger json, bin, et "xgboost_model"
    booster.load_model(model_path)
    return booster

def choose_threshold_by_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # thr a une taille = len(prec)-1 ; alignons sur les points valides
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    i = np.nanargmax(f1)
    return {
        "threshold": float(thr[i]),
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
        "prec_curve": prec.tolist(),
        "rec_curve": rec.tolist(),
        "thr_curve": thr.tolist(),
    }

def choose_threshold_by_profit(y_true, y_prob, tp_gain=1.0, fp_cost=1.0):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # Pour chaque seuil t, on reconstruit TP/FP approx :
    # P = precision = TP / (TP+FP) ; R = recall = TP / Pos
    Pos = np.sum(y_true == 1)
    Neg = np.sum(y_true == 0)
    # On ne calcule que sur thr (len-1)
    P = prec[:-1]; R = rec[:-1]
    TP = R * Pos
    FP = (TP / (P + 1e-12)) - TP
    profit = tp_gain * TP - fp_cost * FP
    i = np.nanargmax(profit)
    return {
        "threshold": float(thr[i]),
        "precision": float(P[i]),
        "recall": float(R[i]),
        "profit": float(profit[i]),
        "tp_gain": float(tp_gain),
        "fp_cost": float(fp_cost),
        "prec_curve": prec.tolist(),
        "rec_curve": rec.tolist(),
        "thr_curve": thr.tolist(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv-uri", required=True,
                    help="CSV validation, colonnes: [y, features...] (S3 ou local)")
    ap.add_argument("--model-uri", required=True,
                    help="Chemin modèle (model.json/bin/xgboost_model) ou artefact .tar.gz (S3 ou local)")
    ap.add_argument("--weights-csv-uri", default=None,
                    help="Optionnel: CSV de poids par ligne (S3 ou local), 1 poids par ligne")
    ap.add_argument("--out-json-uri", required=True,
                    help="Où écrire le JSON récap (S3 ou local)")
    ap.add_argument("--optimize", choices=["f1", "profit"], default="f1",
                    help="Critère pour choisir le seuil")
    ap.add_argument("--tp-gain", type=float, default=1.0,
                    help="Gain par vrai positif (si optimize=profit)")
    ap.add_argument("--fp-cost", type=float, default=1.0,
                    help="Coût par faux positif (si optimize=profit)")
    ap.add_argument("--delimiter", default=",",
                    help="Délimiteur CSV (par défaut ,)")
    args = ap.parse_args()

    # 1) Charger validation
    df = load_csv(args.val_csv_uri, header=None)
    if args.delimiter != ",":
        # si besoin d'un autre séparateur
        df = pd.read_csv(fsspec.open(args.val_csv_uri, "rb") if is_s3(args.val_csv_uri) else args.val_csv_uri,
                         header=None, sep=args.delimiter)

    y = df.iloc[:, 0].to_numpy().astype(int)
    X = df.iloc[:, 2:].to_numpy()   # skip col 1 = side_num

    # (optionnel) poids
    weights = None
    if args.weights_csv_uri:
        wdf = load_csv(args.weights_csv_uri, header=None)
        weights = wdf.iloc[:, 0].to_numpy().astype(float)
        if len(weights) != len(y):
            raise ValueError(f"weights ({len(weights)}) != n_samples ({len(y)})")

    # 2) Charger modèle + prédire
    booster = load_booster(args.model_uri)
    dval = xgb.DMatrix(X, label=y, weight=weights)
    y_prob = booster.predict(dval)
    auprc = float(average_precision_score(y, y_prob, sample_weight=weights))

    # 3) Choix du seuil
    if args.optimize == "f1":
        chosen = choose_threshold_by_f1(y, y_prob)
        crit_name = "f1"
        crit_value = chosen["f1"]
    else:
        chosen = choose_threshold_by_profit(y, y_prob, args.tp_gain, args.fp_cost)
        crit_name = "profit"
        crit_value = chosen["profit"]

    thr = chosen["threshold"]
    y_hat = (y_prob >= thr).astype(int)

    # 4) Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0,1]).ravel()
    precision = float(tp / (tp + fp + 1e-12))
    recall    = float(tp / (tp + fn + 1e-12))

    # 5) Résumé JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "val_csv_uri": args.val_csv_uri,
        "weights_csv_uri": args.weights_csv_uri,
        "model_uri": args.model_uri,
        "optimize": args.optimize,
        "criterion_value": float(crit_value),
        "best_threshold": float(thr),
        "metrics": {
            "auprc": auprc,
            "precision_at_thr": precision,
            "recall_at_thr": recall,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "n": int(len(y)),
            "pos_rate": float(np.mean(y))
        },
        "curves": {
            "precision": chosen["prec_curve"][:2000],  # évite JSON trop gros
            "recall": chosen["rec_curve"][:2000],
            "thresholds": chosen["thr_curve"][:2000],
        }
    }

    # 6) Affichage court + sauvegarde
    print(f"AUPRC={auprc:.5f} | best_thr={thr:.4f} ({args.optimize}={crit_value:.4f}) "
          f"| P={precision:.3f} R={recall:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}")
    save_json(summary, args.out_json_uri)
    print(f"➡️  Écrit: {args.out_json_uri}")

if __name__ == "__main__":
    main()


#python ml_bot/scripts/sageMaker/post_eval_xgb.py \
#  --val-csv-uri s3://tradebot-config-tokyo/data/stage3-xgb/v1-go/validation/part-00000.csv \
#  --model-uri   s3://tradebot-config-tokyo/models/xgb/xgb-go-20251106-141046/output/model.tar.gz \
#  --out-json-uri s3://tradebot-config-tokyo/models/xgb/xgb-go-20251106-141046/metrics/val_summary.json