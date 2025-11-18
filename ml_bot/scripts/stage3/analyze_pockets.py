# ml_bot/scripts/stage3/analyze_pockets.py
import os, io, json, time, argparse, tarfile, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import s3fs
import boto3
from botocore.config import Config as BotoConfig

from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

import pyarrow.parquet as pq
import xgboost as xgb


# -----------------------
# Helpers S3 / I/O
# -----------------------
def _boto3_client(service, region):
    return boto3.client(
        service,
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}, connect_timeout=5, read_timeout=60),
    )

def _s3_put_bytes(s3_client, s3_uri: str, data: bytes):
    bucket = s3_uri.split("/", 3)[2]
    key    = s3_uri.split("/", 3)[3]
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)

def _glob_s3(fs: s3fs.S3FileSystem, prefix: str, pattern="part-*.csv"):
    base = prefix.replace("s3://", "")
    return [f"s3://{p}" for p in fs.glob(f"{base}/{pattern}")]

def _load_stage3_csv(fs: s3fs.S3FileSystem, prefix: str) -> pd.DataFrame:
    """
    Stage3 CSV: colonnes = [Y, side_num, feat1, feat2, ...]
    On ne charge que Y et side_num pour l'analyse structurelle.
    """
    paths = _glob_s3(fs, prefix)
    if not paths:
        raise FileNotFoundError(f"Aucun CSV sous {prefix}/part-*.csv")
    dfs = []
    for p in paths:
        # header=None → 0:Y, 1:side_num
        df = pd.read_csv(p, header=None, usecols=[0, 1])
        df.columns = ["y", "side_num"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def _load_meta_tf(fs: s3fs.S3FileSystem, meta_split_dir: str) -> pd.Series:
    """
    Lit les meta parquet (~ _meta/splits_parquet/<split>/*.parquet)
    On n'a besoin que de 'tf'. On suppose l'ordre d'écriture conservé (même concat).
    """
    meta_base = meta_split_dir.replace("s3://", "")
    files = fs.glob(f"{meta_base}/*.parquet")
    if not files:
        raise FileNotFoundError(f"Aucun parquet sous {meta_split_dir}")
    tables = []
    for p in files:
        with fs.open(f"s3://{p}", "rb") as f:
            tables.append(pq.read_table(f))
    meta_df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    if "tf" not in meta_df.columns:
        raise KeyError("Colonne 'tf' absente des meta parquet")
    return meta_df["tf"].reset_index(drop=True)

def _safe_concat_features(df_core: pd.DataFrame, tf_series: pd.Series) -> pd.DataFrame:
    if len(df_core) != len(tf_series):
        raise RuntimeError(f"Longueurs différentes entre CSV ({len(df_core)}) et meta tf ({len(tf_series)})")
    out = df_core.copy()
    out["tf"] = tf_series
    # sécurité : side_num en int {-1, 1}
    out["side_num"] = out["side_num"].astype(int)
    return out

# -----------------------
# XGBoost (optionnel) pour AUPRC par poche
# -----------------------
def _load_booster_from_s3(fs: s3fs.S3FileSystem, model_uri: str) -> xgb.Booster:
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
        # fallback via fichier temp
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_bytes); tmp.flush()
            booster.load_model(tmp.name)
    return booster

def _load_full_matrix(fs: s3fs.S3FileSystem, prefix: str):
    """Charge toutes les colonnes (pour inférence) → ndarray X, y, side_num, tf (depuis meta)."""
    # 1) CSV complet
    paths = _glob_s3(fs, prefix)
    if not paths:
        raise FileNotFoundError(f"Aucun CSV sous {prefix}/part-*.csv")
    dfs = [pd.read_csv(p, header=None) for p in paths]
    full = pd.concat(dfs, ignore_index=True)

    y         = full.iloc[:, 0].astype(np.float32).to_numpy()
    side_num  = full.iloc[:, 1].astype(int).to_numpy()
    X         = full.iloc[:, 1:].astype(np.float32).to_numpy()  # inclut side_num comme 1ère colonne des features

    return y, side_num, X

def _predict_split(fs: s3fs.S3FileSystem, model_uri: str, split_prefix: str, meta_split_dir: str):
    y, side_num, X = _load_full_matrix(fs, split_prefix)
    tf_series = _load_meta_tf(fs, meta_split_dir)
    if len(tf_series) != len(y):
        raise RuntimeError("Incohérence de taille entre meta tf et matrice X/y")

    booster = _load_booster_from_s3(fs, model_uri)
    yhat = booster.predict(xgb.DMatrix(X, label=y))

    # DataFrame pour groupby
    df = pd.DataFrame({
        "y": y.astype(np.int32),
        "side_num": side_num.astype(np.int32),
        "tf": tf_series.values,
        "yhat": yhat.astype(np.float32),
    })
    return df

def _auprc_pr(y, yhat, w=None):
    try:
        return float(average_precision_score(y, yhat, sample_weight=w))
    except Exception:
        return float(average_precision_score(y, yhat))

# -----------------------
# Analyse principale
# -----------------------
def analyze(data_root: str,
            aws_region: str,
            out_s3_prefix: str = None,
            model_uri: str = None):
    """
    - Compte global et par side_num sur TRAIN/VAL/TEST
    - Compte par poche (tf × side_num) sur TRAIN/VAL/TEST
    - Si model_uri fourni: AUPRC global et par poche sur VALIDATION & TEST
    - Donne des heuristiques pour décider 'split per side' vs 'modèle global'
    """
    fs = s3fs.S3FileSystem()
    s3c = _boto3_client("s3", aws_region)

    TR = f"{data_root}/train"
    VA = f"{data_root}/validation"
    TE = f"{data_root}/test"
    MR = f"{data_root}/_meta/splits_parquet"

    # 1) LOAD CORE (Y, side_num) + TF
    def _load_split(split_name, prefix):
        df_core = _load_stage3_csv(fs, prefix)   # y, side_num
        tf = _load_meta_tf(fs, f"{MR}/{split_name}")
        return _safe_concat_features(df_core, tf)  # y, side_num, tf

    train = _load_split("train", TR)
    valid = _load_split("validation", VA)
    test  = _load_split("test", TE)

    # 2) Comptes globaux par side
    def _agg_side(df):
        g = df.groupby("side_num")["y"].agg(count="count", pos="sum", pos_rate="mean").reset_index()
        g["pos"] = g["pos"].astype(int)
        return g.sort_values("side_num")

    agg_train_side = _agg_side(train)
    agg_valid_side = _agg_side(valid)
    agg_test_side  = _agg_side(test)

    # 3) Poches (tf × side_num)
    def _agg_pocket(df, label):
        g = (df.groupby(["tf", "side_num"])["y"]
               .agg(count="count", pos="sum", pos_rate="mean")
               .reset_index())
        g["pos"] = g["pos"].astype(int)
        g["split"] = label
        return g.sort_values(["tf", "side_num"])

    pockets_train = _agg_pocket(train, "train")
    pockets_valid = _agg_pocket(valid, "validation")
    pockets_test  = _agg_pocket(test,  "test")

    # 4) (Optionnel) Evaluation modèle actuel (VALIDATION + TEST)
    eval_valid = None
    eval_test  = None
    if model_uri:
        # VALIDATION
        dfv = _predict_split(fs, model_uri, VA, f"{MR}/validation")  # y, side_num, tf, yhat
        auprc_val_global = _auprc_pr(dfv["y"].to_numpy(), dfv["yhat"].to_numpy())
        by_pocket_val = (dfv.groupby(["tf", "side_num"])
                           .apply(lambda g: pd.Series({
                               "n": len(g),
                               "pos": int(g["y"].sum()),
                               "pos_rate": float(g["y"].mean()),
                               "AUPRC": _auprc_pr(g["y"].to_numpy(), g["yhat"].to_numpy())
                           }))
                           .reset_index()
                           .sort_values("AUPRC", ascending=False))

        # TEST
        dft = _predict_split(fs, model_uri, TE, f"{MR}/test")
        auprc_tst_global = _auprc_pr(dft["y"].to_numpy(), dft["yhat"].to_numpy())
        by_pocket_tst = (dft.groupby(["tf", "side_num"])
                           .apply(lambda g: pd.Series({
                               "n": len(g),
                               "pos": int(g["y"].sum()),
                               "pos_rate": float(g["y"].mean()),
                               "AUPRC": _auprc_pr(g["y"].to_numpy(), g["yhat"].to_numpy())
                           }))
                           .reset_index()
                           .sort_values("AUPRC", ascending=False))

        eval_valid = {
            "global_AUPRC": auprc_val_global,
            "by_pocket": by_pocket_val.to_dict(orient="records"),
        }
        eval_test = {
            "global_AUPRC": auprc_tst_global,
            "by_pocket": by_pocket_tst.to_dict(orient="records"),
        }

    # 5) Heuristique décision split
    #    - côté +1 (long) : nb_pos_total, et par poches dominantes
    #    - côté -1 (short): idem
    def _side_snapshot(df, side):
        sub = df[df["side_num"] == side]
        return {
            "rows": int(len(sub)),
            "pos": int(sub["y"].sum()),
            "pos_rate": float(sub["y"].mean())
        }

    snapshot = {
        "train": {
            "short(-1)": _side_snapshot(train, -1),
            "long(+1)":  _side_snapshot(train,  1),
        },
        "validation": {
            "short(-1)": _side_snapshot(valid, -1),
            "long(+1)":  _side_snapshot(valid,  1),
        },
        "test": {
            "short(-1)": _side_snapshot(test, -1),
            "long(+1)":  _side_snapshot(test,  1),
        }
    }

    # Règles simples:
    # - OK split si :
    #   (a) pos_long_train >= 250   (sinon le long-only risque d'être trop rare)
    #   (b) pos_short_train >= 250  (sinon inutile de split)
    #   (c) et si (eval_test.by_pocket montre une vraie asymétrie d'AUPRC en faveur d'un side)
    pos_long = snapshot["train"]["long(+1)"]["pos"]
    pos_short= snapshot["train"]["short(-1)"]["pos"]
    split_suggested = (pos_long >= 250 and pos_short >= 250)

    # 6) Assemblage rapport
    report = {
        "data_root": data_root,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts_by_side": {
            "train": agg_train_side.to_dict(orient="records"),
            "validation": agg_valid_side.to_dict(orient="records"),
            "test": agg_test_side.to_dict(orient="records"),
        },
        "pockets": {
            "train": pockets_train.to_dict(orient="records"),
            "validation": pockets_valid.to_dict(orient="records"),
            "test": pockets_test.to_dict(orient="records"),
        },
        "model_uri": model_uri,
        "eval_validation": eval_valid,
        "eval_test": eval_test,
        "decision_guidance": {
            "pos_long_train": pos_long,
            "pos_short_train": pos_short,
            "split_suggested_by_counts": split_suggested,
            "rules": {
                "min_pos_per_side_train": 250,
                "rationale": "en dessous, le modèle par side devient fragile; au-dessus, XGB tient généralement.",
                "next_steps_if_split": [
                    "entraîner XGB_long (filtre side_num=+1) & XGB_short (side_num=-1)",
                    "grille spw côté long plus agressive (pos bien plus rares)",
                    "calibration Isotonic par poche, par side"
                ]
            }
        }
    }

    # 7) Impression lisible
    def _print_side_table(lbl, gdf):
        print(f"\n[{lbl}] Comptes par side_num")
        print(gdf.to_string(index=False))

    _print_side_table("TRAIN", agg_train_side)
    _print_side_table("VALIDATION", agg_valid_side)
    _print_side_table("TEST", agg_test_side)

    print("\n[Pockets] exemples (VALIDATION):")
    print(pockets_valid.sort_values(["tf","side_num","count"], ascending=[True, True, False]).head(12).to_string(index=False))

    if eval_valid:
        print(f"\n[VALIDATION] AUPRC global = {eval_valid['global_AUPRC']:.5f}")
        evv = pd.DataFrame(eval_valid["by_pocket"]).sort_values("AUPRC", ascending=False)
        print(evv.to_string(index=False))

    if eval_test:
        print(f"\n[TEST] AUPRC global = {eval_test['global_AUPRC']:.5f}")
        evt = pd.DataFrame(eval_test["by_pocket"]).sort_values("AUPRC", ascending=False)
        print(evt.to_string(index=False))

    print("\n[Decision] Snapshot counts:", json.dumps(snapshot, indent=2))
    print("\n[Decision] Split suggested by counts?:", split_suggested)

    # 8) Sauvegardes (optionnel)
    if out_s3_prefix:
        stamp = int(time.time())
        out_uri = f"{out_s3_prefix.rstrip('/')}/pocket_analysis_{stamp}.json"
        _s3_put_bytes(s3c, out_uri, json.dumps(report, indent=2).encode("utf-8"))
        print(f"[saved] {out_uri}")


# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Analyse globale et par poche (tf × side_num) pour décision split.")
    ap.add_argument("--data-root", required=True, help="ex: s3://tradebot-config-tokyo/data/stage3-xgb/v3-go")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--model-uri", default=None, help="(optionnel) s3://.../model.tar.gz pour AUPRC par poche")
    ap.add_argument("--out-s3-prefix", default=None, help="(optionnel) s3://... où écrire le rapport JSON")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyze(
        data_root=args.data_root,
        aws_region=args.aws_region,
        out_s3_prefix=args.out_s3_prefix,
        model_uri=args.model_uri
    )