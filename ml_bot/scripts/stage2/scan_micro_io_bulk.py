#!/usr/bin/env python3
# scan_micro_io_bulk.py — lance validate_micro_io-like checks en bulk et produit un rapport CSV
from __future__ import annotations
import argparse, itertools
import numpy as np
import pandas as pd
import pyarrow.fs as pafs

BOOK_LEVELS = 15

# ─────────────────────────── Helpers ────────────────────────────
def _s3fs(region): 
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _exists(fs, s3uri):
    obj = s3uri[len("s3://"):] if s3uri.startswith("s3://") else s3uri
    try:
        return fs.get_file_info([obj])[0].type != pafs.FileType.NotFound
    except Exception:
        return False

def _fmt(x):
    try:
        return pd.Timestamp(x).isoformat()
    except Exception:
        return str(x)

def _read_parquet(fs, s3uri, cols=None):
    import pyarrow.dataset as ds
    path = s3uri[len("s3://"):]
    dataset = ds.dataset(path, filesystem=fs, format="parquet")
    tbl = dataset.to_table(columns=cols)
    return tbl.to_pandas(types_mapper=pd.ArrowDtype)

# ────────────────────── Core scan par fichier ──────────────────────
def _one_month_probe(fs, symbol, ym, book_root, trade_root) -> dict:
    y, m = map(int, ym.split("-"))
    book = f"{book_root.rstrip('/')}/{symbol}/{y}-{m:02d}.parquet"
    tr   = f"{trade_root.rstrip('/')}/{symbol}/{y}-{m:02d}.parquet" if trade_root else None

    out = {
        "symbol":symbol, "ym":ym,
        "has_book":False, "has_trades":False,
        "rows_book":0, "rows_trades":0,
        "ts_min":None, "ts_max":None,
        "rt_min":None, "rt_max":None,
        "gap_ratio_1s":np.nan, "crossed_count":np.nan,
        "seq_decreases":np.nan,
        "spread_p10":np.nan,"spread_p50":np.nan,"spread_p90":np.nan,
        "lat_ms_p50":np.nan,"lat_ms_p90":np.nan,"lat_ms_p99":np.nan,"lat_neg":np.nan
    }

    if not _exists(fs, book):
        return out

    out["has_book"] = True
    cols = ["timestamp","received_time","seq","bid_0_price","ask_0_price","bid_0_size","ask_0_size"]
    try:
        df = _read_parquet(fs, book, cols)
    except Exception as e:
        out["error_book"] = f"{type(e).__name__}: {e}"
        return out

    if df.empty:
        return out

    out["rows_book"] = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "received_time" in df.columns:
        df["received_time"] = pd.to_datetime(df["received_time"], utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    out["ts_min"], out["ts_max"] = _fmt(df["timestamp"].min()), _fmt(df["timestamp"].max())

    # Latence
    if "received_time" in df.columns:
        ok = df["timestamp"].notna() & df["received_time"].notna()
        if ok.any():
            dms = (df.loc[ok,"received_time"] - df.loc[ok,"timestamp"]).dt.total_seconds() * 1000.0
            out["lat_neg"]   = int((dms < 0).sum())
            p = np.nanpercentile(dms, [50,90,99])
            out["lat_ms_p50"], out["lat_ms_p90"], out["lat_ms_p99"] = float(p[0]), float(p[1]), float(p[2])
            out["rt_min"], out["rt_max"] = _fmt(df.loc[ok,"received_time"].min()), _fmt(df.loc[ok,"received_time"].max())

    # Crossed book
    chk = df[["bid_0_price","ask_0_price"]].dropna()
    out["crossed_count"] = int((chk["bid_0_price"] >= chk["ask_0_price"]).sum())

    # Gap ratio
    rs = df.set_index("timestamp")[["bid_0_price"]].resample("1s").last()
    out["gap_ratio_1s"] = float(rs["bid_0_price"].isna().mean()) if len(rs)>0 else np.nan

    # Seq monotonic
    if "seq" in df.columns:
        seq = pd.to_numeric(df["seq"], errors="coerce")
        out["seq_decreases"] = int((seq.diff() < 0).fillna(False).sum())

    # Spread
    sp = ((df["ask_0_price"] - df["bid_0_price"]) /
          ((df["ask_0_price"] + df["bid_0_price"])/2.0))*1e4
    sp = sp.replace([np.inf,-np.inf], np.nan)
    if sp.notna().any():
        p = np.nanpercentile(sp.dropna().values, [10,50,90])
        out["spread_p10"], out["spread_p50"], out["spread_p90"] = float(p[0]), float(p[1]), float(p[2])

    # Trades
    if tr and _exists(fs, tr):
        out["has_trades"] = True
        try:
            trv = _read_parquet(fs, tr, cols=["timestamp","received_time","price","qty"])
            out["rows_trades"] = len(trv)
        except Exception:
            pass

    return out

# ────────────────────── CLI + boucle principale ──────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aws-region", default="ap-northeast-1")
    p.add_argument("--book-root", required=True, help="s3://…/book15")
    p.add_argument("--trade-root", default=None, help="s3://…/trades (optional)")
    p.add_argument("--symbols", nargs="+", required=True, help="Liste des symboles (ex: BTCUSDT ETHUSDT ...)")
    p.add_argument("--years", nargs="+", type=int, required=True, help="Liste des années (ex: 2023 2024 2025)")
    p.add_argument("--out-csv", required=True, help="s3://…/qa/microio_scan.csv")

    # ── Seuils RELAXÉS (configurables depuis la CLI) ──
    p.add_argument("--gap-max", type=float, default=0.80, help="Seuil gap_ratio_1s au-dessus duquel on lève flag_gap_hi (défaut: 0.80)")
    p.add_argument("--crossed-max", type=int, default=1_000, help="Seuil crossed_count au-dessus duquel on lève flag_crossing (défaut: 1000)")
    p.add_argument("--seq-decr-max", type=int, default=10_000, help="Seuil seq_decreases au-dessus duquel on lève flag_seq_back (défaut: 10000)")
    p.add_argument("--lat-neg-max", type=int, default=100, help="Nombre de latences négatives tolérées avant flag_lat_neg (défaut: 100)")
    return p.parse_args()

def main():
    args = parse_args()
    fs = _s3fs(args.aws_region)

    symbols = args.symbols
    years   = args.years
    months = [f"{y}-{m:02d}" for y in years for m in range(1, 13)]

    rows = []
    for sym, ym in itertools.product(symbols, months):
        rows.append(_one_month_probe(fs, sym, ym, args.book_root, args.trade_root))
        print(f"… {sym} {ym} done")

    df = pd.DataFrame(rows)

    # ── Flags RELAXÉS (au lieu de >0 stricts) ──
    df["flag_gap_hi"]   = df["gap_ratio_1s"].fillna(1.0) > args.gap_max
    df["flag_crossing"] = df["crossed_count"].fillna(0).astype(float) > float(args.crossed_max)
    df["flag_seq_back"] = df["seq_decreases"].fillna(0).astype(float) > float(args.seq_decr_max)
    df["flag_lat_neg"]  = df["lat_neg"].fillna(0).astype(float) > float(args.lat_neg_max)

    # Écrit rapport CSV sur S3
    path = args.out_csv
    with fs.open_output_stream(path[len("s3://"):]) as out:
        out.write(df.to_csv(index=False).encode("utf-8"))
    print(f"✅ report written to {path}")
    print(
        f"ℹ️ seuils: gap_max={args.gap_max}, crossed_max={args.crossed_max}, "
        f"seq_decr_max={args.seq_decr_max}, lat_neg_max={args.lat_neg_max}"
    )

if __name__ == "__main__":
    main()