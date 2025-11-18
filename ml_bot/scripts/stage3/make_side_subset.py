# ml_bot/scripts/stage3/make_side_subset.py
import argparse, io, json, os, tarfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq

def _glob(fs: s3fs.S3FileSystem, prefix: str, pattern: str):
    base = prefix.replace("s3://","")
    return [f"s3://{p}" for p in fs.glob(f"{base}/{pattern}")]

def _load_csv_concat(fs: s3fs.S3FileSystem, prefix: str) -> pd.DataFrame:
    paths = _glob(fs, prefix, "part-*.csv")
    if not paths:
        raise FileNotFoundError(f"Aucun CSV sous {prefix}/part-*.csv")
    dfs = [pd.read_csv(p, header=None) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def _maybe_load_weights(fs: s3fs.S3FileSystem, weight_prefix: str):
    paths = _glob(fs, weight_prefix, "part-*.csv")
    if not paths:
        return None
    dfs = [pd.read_csv(p, header=None) for p in paths]
    w = pd.concat(dfs, ignore_index=True)
    # poids: 1 seule colonne
    col = w.columns[0]
    return w[col].astype(float).to_numpy()

def _load_meta_table(fs: s3fs.S3FileSystem, meta_split_dir: str) -> pa.Table:
    base = meta_split_dir.replace("s3://","")
    files = fs.glob(f"{base}/*.parquet")
    if not files:
        raise FileNotFoundError(f"Aucun parquet sous {meta_split_dir}")
    tabs = []
    for p in files:
        with fs.open(f"s3://{p}", "rb") as f:
            tabs.append(pq.read_table(f))
    return pa.concat_tables(tabs, promote=True)

def _write_single_csv(fs: s3fs.S3FileSystem, dst_prefix: str, df: pd.DataFrame):
    out = dst_prefix.rstrip("/") + "/part-00000.csv"
    with fs.open(out, "wb") as f:
        df.to_csv(f, header=False, index=False)

def _write_single_weights(fs: s3fs.S3FileSystem, dst_prefix: str, w: np.ndarray):
    out = dst_prefix.rstrip("/") + "/part-00000.csv"
    with fs.open(out, "wb") as f:
        pd.DataFrame(w).to_csv(f, header=False, index=False)

def _write_single_parquet(fs: s3fs.S3FileSystem, dst_dir: str, table: pa.Table):
    path = dst_dir.rstrip("/") + "/part-00000.parquet"
    with fs.open(path, "wb") as f:
        pq.write_table(table, f, compression="zstd")

def _ensure_dir(fs: s3fs.S3FileSystem, s3_uri: str):
    # s3fs crée à l’écriture; rien à faire ici
    return

def _copy_if_exists(fs: s3fs.S3FileSystem, src: str, dst: str):
    try:
        with fs.open(src, "rb") as fsrc, fs.open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
    except FileNotFoundError:
        pass

def make_subset(src_root: str, dst_root: str, side: int):
    fs = s3fs.S3FileSystem()
    assert side in (-1, 1), "side doit être -1 ou 1"

    splits = ["train","validation","test"]
    weights_dirs = {
        "train": f"{src_root}/train_weight",
        "validation": f"{src_root}/validation_weight",
        "test": f"{src_root}/test_weight",
    }

    meta_splits = f"{src_root}/_meta/splits_parquet"
    dst_meta_splits = f"{dst_root}/_meta/splits_parquet"

    stats = {}
    for split in splits:
        feat_prefix = f"{src_root}/{split}"
        df = _load_csv_concat(fs, feat_prefix)   # [Y, side_num, feat1, ...]
        if df.shape[1] < 2:
            raise RuntimeError("CSV Stage3 attendus: 2 premières colonnes = Y, side_num")

        mask = (df.iloc[:,1].astype(int) == side)
        df_sub = df.loc[mask].reset_index(drop=True)

        # weights
        w_src_dir = weights_dirs[split]
        w = _maybe_load_weights(fs, w_src_dir)
        if w is not None:
            if len(w) != len(df):
                raise RuntimeError(f"weights != features ({len(w)} vs {len(df)}) pour {split}")
            w_sub = w[mask.to_numpy()]
        else:
            w_sub = None

        # meta parquet
        meta_dir = f"{meta_splits}/{split}"
        tab = _load_meta_table(fs, meta_dir)
        if tab.num_rows != len(df):
            raise RuntimeError(f"meta rows != features ({tab.num_rows} vs {len(df)}) pour {split}")
        idx = np.nonzero(mask.to_numpy())[0]
        tab_sub = tab.take(pa.array(idx))

        # write
        _write_single_csv(fs, f"{dst_root}/{split}", df_sub)
        if w_sub is not None:
            _write_single_weights(fs, f"{dst_root}/{split}_weight", w_sub)
        _ensure_dir(fs, f"{dst_meta_splits}/{split}")
        _write_single_parquet(fs, f"{dst_meta_splits}/{split}", tab_sub)

        stats[split] = {
            "rows_total": int(len(df)),
            "rows_kept": int(len(df_sub)),
            "pos_kept":  int(df_sub.iloc[:,0].sum()),
            "pos_rate_kept": float(df_sub.iloc[:,0].mean()),
        }

    # copier columns.json et scaler_stats.json (pooled z-score OK pour subset)
    _copy_if_exists(fs, f"{src_root}/_meta/columns.json",      f"{dst_root}/_meta/columns.json")
    _copy_if_exists(fs, f"{src_root}/_meta/scaler_stats.json", f"{dst_root}/_meta/scaler_stats.json")

    # écrire train_pos_weight.json recalculé (neg/pos)
    y_train = _load_csv_concat(fs, f"{dst_root}/train").iloc[:,0].to_numpy()
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    spw = float(neg / max(pos, 1.0))
    meta = {"label_mode": "go", "pos_weight": spw, "side": side, "created_at": int(time.time())}
    with fs.open(f"{dst_root}/_meta/train_pos_weight.json", "wb") as f:
        f.write(json.dumps(meta, indent=2).encode("utf-8"))

    print(json.dumps({
        "dst_root": dst_root,
        "side": side,
        "train_pos_weight": spw,
        "stats": stats
    }, indent=2))
    return spw

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="ex: s3://.../data/stage3-xgb/v3-go")
    ap.add_argument("--dst-root", required=True, help="ex: s3://.../data/stage3-xgb/v3-go-longonly")
    ap.add_argument("--side", type=int, required=True, choices=[-1,1], help="-1=short, 1=long")
    args = ap.parse_args()
    make_subset(args.src_root, args.dst_root, args.side)

# 1) Construire les subsets
#python ml_bot/scripts/stage3/make_side_subset.py \
#  --src-root s3://tradebot-config-tokyo/data/stage3-xgb/v3-go \
#  --dst-root s3://tradebot-config-tokyo/data/stage3-xgb/v3-go-longonly \
#  --side 1

#python ml_bot/scripts/stage3/make_side_subset.py \
#  --src-root s3://tradebot-config-tokyo/data/stage3-xgb/v3-go \
#  --dst-root s3://tradebot-config-tokyo/data/stage3-xgb/v3-go-shortonly \
#  --side -1

# 2) QA rapide
#python ml_bot/scripts/stage3/validate_stage3_csv.py \
#  --root s3://tradebot-config-tokyo/data/stage3-xgb/v3-go-longonly \
#  --split all --aws-region ap-northeast-1 --check-meta-parquet

#python ml_bot/scripts/stage3/validate_stage3_csv.py \
#  --root s3://tradebot-config-tokyo/data/stage3-xgb/v3-go-shortonly \
#  --split all --aws-region ap-northeast-1 --check-meta-parquet
    