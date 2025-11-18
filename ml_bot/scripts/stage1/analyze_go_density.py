#!/usr/bin/env python3
# analyze_go_density.py — comptage par (symbol, tf) sur 2024–2025 pour le modèle GO

from __future__ import annotations
import argparse, sys
from typing import Optional, Dict
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

# ---------------------------------------------------------------------------
def _fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _strip_s3(uri: str) -> str:
    assert uri.startswith("s3://"), "Only s3:// supported"
    return uri[len("s3://"):]

def _dataset_from_split(fs: pafs.S3FileSystem, root: str, split: str) -> ds.Dataset:
    base = _strip_s3(f"{root.rstrip('/')}/{split}")
    return ds.dataset(base, filesystem=fs, format="parquet", partitioning="hive")

# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Compter rows/GO+ par (symbol, tf) sur 2024–2025.")
    ap.add_argument("--src-root", required=True, help="s3://…/data/stage2")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--splits", nargs="*", default=["train","val","test"],
                    help="Liste des splits stage2 à analyser (défaut: train,val,test)")
    ap.add_argument("--out", default="go_density_2024_2025.csv",
                    help="Nom du CSV de sortie (local ou s3://…)")
    return ap.parse_args()

# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    fs = _fs(args.aws_region)
    all_rows = []

    # Limites temporelles
    t_min = pd.Timestamp("2024-01-01", tz="UTC")
    t_max = pd.Timestamp("2025-10-31 23:59:59.999999", tz="UTC")

    for split in args.splits:
        try:
            dataset = _dataset_from_split(fs, args.src_root, split)
        except Exception as e:
            print(f"⚠️  Split {split} introuvable ou illisible: {e}")
            continue

        print(f"🔍 Scan {split} …")
        cols = [c for c in ("t","symbol","tf","Y","side") if c in dataset.schema.names]
        filt = (ds.field("t") >= pd.Timestamp(t_min)) & (ds.field("t") <= pd.Timestamp(t_max))
        scanner = dataset.scanner(columns=cols, filter=filt, batch_size=250_000)

        for batch in scanner.to_batches():
            df = batch.to_pandas(types_mapper=pd.ArrowDtype)
            if df.empty:
                continue

            # force types
            df["symbol"] = df["symbol"].astype(str)
            df["tf"] = df["tf"].astype(str)
            df["Y"] = pd.to_numeric(df["Y"], errors="coerce").fillna(0).astype("int8")
            df["side"] = df.get("side", "buy").astype(str)
            df["side_num"] = df["side"].map({"buy":1, "sell":-1}).fillna(0).astype("int8")

            # convertir Y tri-class en GO=1/NO-GO=0 relatif à la side
            y_tri = df["Y"].astype("int8")
            sgn = df["side_num"].astype("int8")
            y_dir = (y_tri * sgn).astype("int8")
            df["Y_go"] = (y_dir == 1).astype("int8")

            g = df.groupby(["symbol","tf"], sort=False)["Y_go"].agg(["count","sum"]).reset_index()
            g["split"] = split
            all_rows.append(g)

    if not all_rows:
        print("⛔ Aucun lot lu.")
        sys.exit(2)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = df_all.groupby(["symbol","tf"], as_index=False).agg(
        rows_24_25=("count","sum"),
        go_pos_24_25=("sum","sum")
    )
    df_all["go_ratio"] = (df_all["go_pos_24_25"] / df_all["rows_24_25"]).round(4)

    df_all = df_all.sort_values(["tf","symbol"]).reset_index(drop=True)
    print("\nAperçu:")
    print(df_all.head(40).to_string(index=False))

    # --- export CSV ---
    if args.out.startswith("s3://"):
        with fs.open_output_stream(_strip_s3(args.out)) as f:
            f.write(df_all.to_csv(index=False).encode("utf-8"))
        print(f"✅ Résumé écrit sur {args.out}")
    else:
        df_all.to_csv(args.out, index=False)
        print(f"✅ Résumé écrit localement: {args.out}")

if __name__ == "__main__":
    main()