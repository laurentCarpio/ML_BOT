#!/usr/bin/env python3
# lake_to_s3.py (RAM-safe, streaming monthly Parquet, fixed schema) — supports book (15 levels)
from __future__ import annotations
import os, sys, time, argparse, datetime as dt, gc
from collections import defaultdict

import boto3
import pandas as pd
import lakeapi
from lakeapi.exceptions import NoFilesFound

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_object_dtype,
)

# =======================
# Config
# =======================

BOOK_LEVELS = 15
MAX_ROWS_PER_CHUNK = 200_000
CHUNK_WRITE_ROWS   = 64_000

REQ_BY_TABLE = {
    "level_1": {"origin_time","received_time","bid_0_price","bid_0_size","ask_0_price","ask_0_size"},
    "trades" : {"origin_time","received_time","price","quantity","side","trade_id"},
    "book"   : (
        {"origin_time","received_time","sequence_number"} |
        {f"bid_{i}_price" for i in range(BOOK_LEVELS)} |
        {f"bid_{i}_size"  for i in range(BOOK_LEVELS)} |
        {f"ask_{i}_price" for i in range(BOOK_LEVELS)} |
        {f"ask_{i}_size"  for i in range(BOOK_LEVELS)}
    ),
}

DEFAULT_S3_PREFIX = {
    "level_1": "s3://tradebot-config-tokyo/data/level_1",
    "trades" : "s3://tradebot-config-tokyo/data/trade",
    "book"   : "s3://tradebot-config-tokyo/data/book",
}
def arrow_schema_for(table: str) -> pa.Schema:
    if table == "level_1":
        return pa.schema([
            pa.field("timestamp", pa.timestamp("ns", tz="UTC")),
            pa.field("best_bid",  pa.float64()),
            pa.field("best_ask",  pa.float64()),
            pa.field("bid_qty",   pa.float64()),
            pa.field("ask_qty",   pa.float64()),
            pa.field("spread",    pa.float64()),
        ])
    if table == "trades":
        return pa.schema([
            pa.field("timestamp",     pa.timestamp("ns", tz="UTC")),
            pa.field("received_time", pa.timestamp("ns", tz="UTC")),
            pa.field("price",         pa.float64()),
            pa.field("qty",           pa.float64()),
            pa.field("is_aggr_buy",   pa.bool_()),
            pa.field("trade_id",      pa.int64()),
            pa.field("side",          pa.utf8()),
        ])
    if table == "book":
        fields = [
            pa.field("timestamp",     pa.timestamp("ns", tz="UTC")),
            pa.field("received_time", pa.timestamp("ns", tz="UTC")),
            pa.field("seq",           pa.int64())
        ]
        for i in range(BOOK_LEVELS):
            fields.append(pa.field(f"bid_{i}_price", pa.float64()))
            fields.append(pa.field(f"bid_{i}_size",  pa.float32()))
        for i in range(BOOK_LEVELS):
            fields.append(pa.field(f"ask_{i}_price", pa.float64()))
            fields.append(pa.field(f"ask_{i}_size",  pa.float32()))
        fields.append(pa.field("spread_top", pa.float64()))
        return pa.schema(fields)
    raise ValueError(f"Unknown table: {table}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="book", choices=["level_1","trades","book"])
    ap.add_argument("--exchange", default="BINANCE_FUTURES")
    ap.add_argument("--symbols", nargs="+", required=True,
                    help="One or more lake symbols (e.g., BTC-USDT-PERP ETH-USDT-PERP)")
    ap.add_argument("--lake-bucket", default="qnt.data")
    ap.add_argument("--lake-region", default="eu-west-1")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end",  default="2024-01-01")
    ap.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX["level_1"])
    ap.add_argument("--s3-region", default="ap-northeast-1")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=1.6)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()

def daterange(a: dt.date, b: dt.date):
    d = a
    while d < b:
        yield d
        d += dt.timedelta(days=1)

def _ensure_utc(x):
    ser = x if isinstance(x, pd.Series) else pd.Series(x)
    return _coerce_to_timestamp_utc(ser)

# =======================
# Normalisations
# =======================
def _coerce_to_timestamp_utc(s: pd.Series) -> pd.Series:
    # tz-aware datetime64
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        # already tz-aware → normalize to UTC
        return s.dt.tz_convert("UTC")

    # naive datetime64
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s.dt.tz_localize("UTC")

    # numeric / object → interpret as milliseconds since epoch (your files use ms)
    if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype):
        return pd.to_datetime(s, errors="coerce", utc=True, unit="ms")

    # fallback
    return pd.to_datetime(s, errors="coerce", utc=True)

def df_to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    # 1) complétion colonnes manquantes selon le schema
    for name in schema.names:
        if name not in df.columns:
            t = schema.field(name).type
            if pa.types.is_boolean(t):
                df[name] = pd.Series([pd.NA] * len(df), dtype="boolean")
            elif pa.types.is_integer(t):
                df[name] = pd.Series([pd.NA] * len(df), dtype="Int64")
            elif pa.types.is_floating(t):
                df[name] = pd.Series([float("nan")] * len(df), dtype="float64")
            elif pa.types.is_timestamp(t):
                df[name] = pd.Series([pd.NaT] * len(df), dtype="datetime64[ns]")
                df[name] = df[name].dt.tz_localize("UTC")
            else:
                df[name] = pd.Series([pd.NA] * len(df), dtype="string")

    # 2) coercions ciblées pour les colonnes timestamp déjà présentes
    for f in schema:
        if pa.types.is_timestamp(f.type):
            df[f.name] = _coerce_to_timestamp_utc(df[f.name])

    # 3) ordonner & conversion pandas -> Arrow
    df = df[schema.names]
    tbl = pa.Table.from_pandas(df, preserve_index=False)

    # 4) aligner le schema final (au cas où)
    if tbl.schema != schema:
        tbl = tbl.cast(schema, safe=False)
    return tbl

def normalize_l1(df_src: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "timestamp": _ensure_utc(df_src["origin_time"]),
        "best_bid":  pd.to_numeric(df_src["bid_0_price"], errors="coerce").astype("float64"),
        "best_ask":  pd.to_numeric(df_src["ask_0_price"], errors="coerce").astype("float64"),
        "bid_qty":   pd.to_numeric(df_src["bid_0_size"],  errors="coerce").astype("float64"),
        "ask_qty":   pd.to_numeric(df_src["ask_0_size"],  errors="coerce").astype("float64"),
    })
    out["spread"] = (out["best_ask"] - out["best_bid"]).astype("float64")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out

def normalize_trades(df_src: pd.DataFrame) -> pd.DataFrame:
    side_str = df_src["side"].astype("string").str.lower()
    is_buy = side_str.eq("buy").fillna(False).astype("bool")
    trade_id = pd.to_numeric(df_src["trade_id"], errors="coerce").astype("Int64")
    out = pd.DataFrame({
        "timestamp":     _ensure_utc(df_src["origin_time"]),
        "received_time": _ensure_utc(df_src["received_time"]),
        "price":         pd.to_numeric(df_src["price"],    errors="coerce").astype("float64"),
        "qty":           pd.to_numeric(df_src["quantity"], errors="coerce").astype("float64"),
        "is_aggr_buy":   is_buy,
        "trade_id":      trade_id,
        "side":          side_str.astype("string"),
    })
    out = out.dropna(subset=["timestamp","price","qty"]).sort_values("timestamp", kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out

def normalize_book(df_src: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "timestamp":     _ensure_utc(df_src["origin_time"]),
        "received_time": _ensure_utc(df_src["received_time"]),
        "seq":           pd.to_numeric(df_src.get("sequence_number"), errors="coerce").astype("Int64"),
    })
    for i in range(BOOK_LEVELS):
        out[f"bid_{i}_price"] = pd.to_numeric(df_src.get(f"bid_{i}_price"), errors="coerce").astype("float64")
        out[f"bid_{i}_size"]  = pd.to_numeric(df_src.get(f"bid_{i}_size"),  errors="coerce").astype("float32")
    for i in range(BOOK_LEVELS):
        out[f"ask_{i}_price"] = pd.to_numeric(df_src.get(f"ask_{i}_price"), errors="coerce").astype("float64")
        out[f"ask_{i}_size"]  = pd.to_numeric(df_src.get(f"ask_{i}_size"),  errors="coerce").astype("float32")
    out["spread_top"] = (out["ask_0_price"] - out["bid_0_price"]).astype("float64")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    out.reset_index(drop=True, inplace=True)
    out["seq"] = out["seq"].fillna(-1).astype("int64")
    return out

# =======================
# Arrow / S3 helpers
# =======================

def _fs_and_path_from_uri(uri: str, region: str | None = None):
    if region:
        fs = pafs.S3FileSystem(region=region)
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got {uri}")
        path = uri[len("s3://"):]
        return fs, path
    fs, path = pafs.FileSystem.from_uri(uri)
    return fs, path

def _write_table_chunked(writer: pq.ParquetWriter, tbl: pa.Table):
    n = tbl.num_rows
    for i in range(0, n, CHUNK_WRITE_ROWS):
        writer.write_table(tbl.slice(i, min(CHUNK_WRITE_ROWS, n - i)))

# =======================
# Main
# =======================

def main():
    args = parse_args()

    # S3 prefix par défaut selon table
    if args.s3_prefix == DEFAULT_S3_PREFIX["level_1"]:
        args.s3_prefix = DEFAULT_S3_PREFIX[args.table]

    # Sessions / creds
    lake_key = os.getenv("LAKEAPI_ACCESS_KEY_ID")
    lake_sec = os.getenv("LAKEAPI_SECRET_ACCESS_KEY")
    if not lake_key or not lake_sec:
        print("⛔ LAKEAPI_ACCESS_KEY_ID / LAKEAPI_SECRET_ACCESS_KEY non présents", file=sys.stderr)
        sys.exit(2)

    lake_session = boto3.Session(
        aws_access_key_id=lake_key,
        aws_secret_access_key=lake_sec,
        region_name=args.lake_region,
    )

    req_schema = REQ_BY_TABLE[args.table]
    out_schema = arrow_schema_for(args.table)
    mapping_str = {
        "level_1": "origin_time→timestamp | bid_0_price→best_bid | ask_0_price→best_ask | bid_0_size→bid_qty | ask_0_size→ask_qty | spread=ask-bid",
        "trades":  "origin_time→timestamp | received_time→received_time | price→price | quantity→qty | side→is_aggr_buy/side | trade_id→trade_id",
        "book":    "origin_time→timestamp | received_time→received_time | sequence_number→seq | bid/ask_{0..14}_{price,size} | spread_top=ask_0_price-bid_0_price",
    }[args.table]

    if not args.s3_prefix.startswith("s3://"):
        print("⚠️ s3-prefix does not start with s3://")
    print(f"════════ INGEST {args.table} → monthly parquet (RAM-safe, fixed schema) ════════")
    print(f"Lake table : {args.table}")
    print(f"Lake exch  : {args.exchange}")
    print(f"Lake symbols: {', '.join(args.symbols)}")
    print(f"Lake bucket: {args.lake_bucket}")
    print(f"Lake region: {args.lake_region}")
    print(f"Date range : {args.start} → {args.end} (daily)")
    print(f"S3 out     : {args.s3_prefix.rstrip('/')}/<SYMBOL>/YYYY-MM.parquet")
    print(f"S3 region  : {args.s3_region}")
    print("Expect     :", sorted(req_schema))
    print("Mapping    :", mapping_str)
    print("────────────────────────────────────────────")

    # STS sanity (non bloquant)
    try:
        print(f"[lake] STS caller: {lake_session.client('sts').get_caller_identity().get('Arn')}")
    except Exception as e:
        print(f"[lake] STS check failed: {type(e).__name__}: {e}")
    try:
        print(f"[s3-write] STS caller: {boto3.client('sts').get_caller_identity().get('Arn')}")
    except Exception as e:
        print(f"[s3-write] STS check failed: {type(e).__name__}: {e}")

    start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date   = dt.datetime.strptime(args.end,   "%Y-%m-%d").date()

    writers: dict[str, pq.ParquetWriter] = {}
    current_key: dict[str, tuple[int,int]] = {}
    approx_rows_in_month = defaultdict(int)
    first_logged_for_symbol: dict[str, bool] = {}

    def _open_new_writer(symbol: str, y: int, m: int):
        s3_uri = f"{args.s3_prefix.rstrip('/')}/{symbol}/{y}-{m:02d}.parquet"
        fs, path = _fs_and_path_from_uri(s3_uri, region=args.s3_region)
        w = None
        if not args.dry_run:
            w = pq.ParquetWriter(
                path,
                out_schema,
                filesystem=fs,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        writers[symbol] = w
        current_key[symbol] = (y, m)
        approx_rows_in_month[symbol] = 0
        if args.dry_run:
            print(f"[dry-run] would open writer {s3_uri}")

    def _close_writer(symbol: str):
        w = writers.pop(symbol, None)
        if w is not None:
            w.close()
        current_key.pop(symbol, None)
        approx_rows_in_month.pop(symbol, None)

    try:
        for symbol in args.symbols:
            print(f"──────── Processing symbol: {symbol} ────────")
            first_logged_for_symbol[symbol] = False

            for day in daterange(start_date, end_date):
                day_start = dt.datetime.combine(day, dt.time.min)
                day_end   = day_start + dt.timedelta(days=1)
                day_str   = day.strftime("%Y-%m-%d")

                tries, df = 0, None
                while tries < args.max_retries:
                    tries += 1
                    try:
                        df = lakeapi.load_data(
                            table=args.table,
                            start=day_start,
                            end=day_end,
                            symbols=[symbol],  # IMPORTANT: un seul symbole
                            exchanges=[args.exchange],
                            boto3_session=lake_session,
                            bucket=args.lake_bucket,
                            cached=False,
                        )
                        break
                    except NoFilesFound:
                        df = None
                        break
                    except Exception as e:
                        if tries >= args.max_retries:
                            print(f"❌ load_data failed {day_str} [{symbol}] after {tries} tries: {e}")
                        else:
                            print(f"↻ retry {tries}/{args.max_retries} {day_str} [{symbol}]: {e}")
                            time.sleep(args.backoff ** tries)

                if df is None or len(df) == 0:
                    print(f"⚠️ empty/missing for {day_str} [{symbol}]")
                    gc.collect()
                    continue

                need = sorted(REQ_BY_TABLE[args.table])
                try:
                    df = df[need]
                except Exception as e:
                    print(f"❌ select columns failed {day_str} [{symbol}]: {e} → SKIP")
                    gc.collect()
                    continue

                if not first_logged_for_symbol[symbol]:
                    print(f"✅ first day {day_str} [{symbol}] cols: {list(df.columns)}")
                    miss = sorted(list(req_schema - set(df.columns)))
                    if miss:
                        print(f"⛔ schema mismatch for {symbol} — missing: {miss} → SKIP SYMBOL")
                        # on passe au symbole suivant
                        break
                    first_logged_for_symbol[symbol] = True

                if not req_schema.issubset(df.columns):
                    print(f"❌ {day_str} [{symbol}] schema mismatch — missing {sorted(req_schema - set(df.columns))} → SKIP DAY")
                    gc.collect()
                    continue

                # Normalisation
                if args.table == "level_1":
                    df_norm = normalize_l1(df)
                elif args.table == "trades":
                    df_norm = normalize_trades(df)
                else:
                    df_norm = normalize_book(df)

                if df_norm.empty:
                    print(f"⚠️ normalized empty {day_str} [{symbol}] → SKIP")
                    del df_norm, df
                    gc.collect()
                    continue

                y = int(df_norm["timestamp"].dt.year.iat[0])
                m = int(df_norm["timestamp"].dt.month.iat[0])

                if symbol not in writers:
                    _open_new_writer(symbol, y, m)
                elif (y, m) != current_key[symbol]:
                    prev_y, prev_m = current_key[symbol]
                    print(f"🌀 month rollover [{symbol}]: closing {prev_y}-{prev_m:02d}, opening {y}-{m:02d}")
                    _close_writer(symbol)
                    _open_new_writer(symbol, y, m)

                # Écriture RAM-safe
                if len(df_norm) > MAX_ROWS_PER_CHUNK:
                    df_norm["_hour"] = df_norm["timestamp"].dt.floor("h")
                    for _, df_h in df_norm.groupby("_hour", sort=True):
                        tbl_h = df_to_table(df_h.drop(columns=["_hour"]), out_schema)
                        if writers[symbol] is not None:
                            _write_table_chunked(writers[symbol], tbl_h)
                        approx_rows_in_month[symbol] += len(df_h)
                        del df_h, tbl_h
                        gc.collect()
                else:
                    tbl = df_to_table(df_norm, out_schema)
                    if writers[symbol] is not None:
                        _write_table_chunked(writers[symbol], tbl)
                    approx_rows_in_month[symbol] += len(df_norm)
                    del tbl

                del df_norm, df
                gc.collect()

            _close_writer(symbol)
        print("Done.")
    finally:
        # ensure all writers are closed on any exception
        for s in list(writers.keys()):
            _close_writer(s)

if __name__ == "__main__":
    main()