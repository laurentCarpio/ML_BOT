# next_bot/scripts/validate_micro_io.py
from __future__ import annotations

import argparse, random, re
from typing import Optional, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

BOOK_LEVELS = 15
# Pour TRADES (latence incluse si dispo)
TRADE_COLS_MIN = {"timestamp","received_time","price","qty"}

# ─────────────────────────── FS helpers (S3 only) ────────────────────────────

def _s3_fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _s3_object_path(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError("This validator only supports s3:// paths")
    return uri[len("s3://"):]

def _exists_s3(path: str, region: Optional[str]) -> bool:
    fs = _s3_fs(region)
    obj = _s3_object_path(path)
    try:
        info = fs.get_file_info([obj])[0]
        return info.type != pafs.FileType.NotFound
    except Exception:
        return False

def _read_parquet_s3(path: str,
                     aws_region: Optional[str],
                     columns: Optional[List[str]] = None,
                     filter: Optional[ds.Expression] = None) -> pd.DataFrame:
    fs = _s3_fs(aws_region)
    dataset = ds.dataset(_s3_object_path(path), filesystem=fs, format="parquet")
    table = dataset.to_table(columns=columns, filter=filter)
    return table.to_pandas(types_mapper=pd.ArrowDtype)

# ───────────────────────────── time utils ────────────────────────────────────

def _ensure_utc(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True, errors="coerce")
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        med = pd.Series(s).astype("float64").abs().median()
        unit = "ms" if med < 1e15 else "ns"
        return pd.to_datetime(s, utc=True, unit=unit, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")

def _fmt_dt(ts) -> str:
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return str(ts)

def _book_month_path(root_book: str, symbol: str, y: int, m: int) -> str:
    return f"{root_book.rstrip('/')}/{symbol}/{y}-{m:02d}.parquet"

def _trade_month_path(root_trade: str, symbol: str, y: int, m: int) -> str:
    return f"{root_trade.rstrip('/')}/{symbol}/{y}-{m:02d}.parquet"

# ─────────────────────── schema-level depth detection ────────────────────────

def _detect_levels_from_schema(book_path: str, region: Optional[str]) -> tuple[tuple[int,int], tuple[int,int]]:
    fs = _s3_fs(region)
    dataset = ds.dataset(_s3_object_path(book_path), filesystem=fs, format="parquet")
    names = dataset.schema.names
    bids = sorted({int(m.group(1)) for n in names for m in [re.match(r"bid_(\d+)_price$", n)] if m})
    asks = sorted({int(m.group(1)) for n in names for m in [re.match(r"ask_(\d+)_price$", n)] if m})
    b_min, b_max = (min(bids, default=-1), max(bids, default=-1))
    a_min, a_max = (min(asks, default=-1), max(asks, default=-1))
    return (b_min, b_max), (a_min, a_max)

# ───────────────────────────── BOOK checks ───────────────────────────────────

def _basic_book_checks(df: pd.DataFrame) -> None:
    print("\n── BOOK: basic schema & dtypes")
    # On valide uniquement le top-of-book (cols minimales) pour éviter les faux positifs
    REQUIRED = {"timestamp","received_time","seq","bid_0_price","ask_0_price","bid_0_size","ask_0_size"}
    miss = sorted([c for c in REQUIRED if c not in df.columns])
    if miss:
        print(f"⛔ missing required top-of-book cols: {miss}")
    else:
        print("✅ required top-of-book columns present")

    # Timestamps → UTC
    df["timestamp"] = _ensure_utc(df["timestamp"])
    if "received_time" in df.columns:
        df["received_time"] = _ensure_utc(df["received_time"])

    if df["timestamp"].isna().all():
        print("⛔ timestamp coercion failed (all NA)")
    else:
        print("✅ timestamp coerced to UTC")
    if "received_time" in df.columns:
        print("✅ received_time coerced to UTC" if not df["received_time"].isna().all()
              else "⚠️ received_time coercion failed (all NA)")

    df.sort_values("timestamp", inplace=True, kind="stable")
    df = df.dropna(subset=["timestamp"])

    print("dtypes (sample):")
    show = [c for c in ["timestamp","received_time","seq","bid_0_price","ask_0_price","bid_0_size","ask_0_size"] if c in df.columns]
    print({k: str(df.dtypes[k]) for k in show})

    if not df.empty:
        print(f"time span (timestamp): {_fmt_dt(df['timestamp'].min())} → {_fmt_dt(df['timestamp'].max())}")
        print(f"duplicate timestamps: {int(df['timestamp'].duplicated().sum())}")
        if "received_time" in df.columns:
            print(f"time span (received_time): {_fmt_dt(df['received_time'].min())} → {_fmt_dt(df['received_time'].max())}")

    print("\n── BOOK: structural sanity checks")
    chk = df[["bid_0_price","ask_0_price"]].dropna()
    crossed = int((chk["bid_0_price"] >= chk["ask_0_price"]).sum())
    print(f"crossed top of book (bid0>=ask0): {crossed}")

    # Gaps (top-of-book)
    rs = df.set_index("timestamp")[["bid_0_price"]].resample("1s").last()
    gap_ratio = float(rs["bid_0_price"].isna().mean()) if len(rs)>0 else np.nan
    print(f"resample 1s missing ratio: {gap_ratio:.3f}")

    # Seq monotonic
    if "seq" in df.columns:
        seq = pd.to_numeric(df["seq"], errors="coerce")
        decr = int((seq.diff() < 0).fillna(False).sum())
        print(f"seq decreases: {decr}")

    # Spread stats
    sp = ((df["ask_0_price"] - df["bid_0_price"]) /
          ((df["ask_0_price"] + df["bid_0_price"])/2.0))*1e4
    sp = sp.replace([np.inf,-np.inf], np.nan)
    bad = int((sp<=0).sum()); zro = int((sp==0).sum())
    print("\n── spread (bps) stats on sample:")
    if sp.notna().any():
        p10, p50, p90 = np.nanpercentile(sp.dropna().values, [10,50,90])
        print(f"<=0 count: {bad} | ==0 count: {zro}")
        print(f"p10/med/p90: {p10:.2f} / {p50:.2f} / {p90:.2f}")
    else:
        print("insufficient data")

# ─────────────────────────── Probe windows (RAM-safe) ────────────────────────

def _ts_scalar_for(ts: pd.Timestamp, ts_type: pa.DataType) -> pa.Scalar:
    """Construit un scalaire Arrow avec même unité/tz que la colonne timestamp."""
    assert pa.types.is_timestamp(ts_type)
    unit = ts_type.unit  # 'ns'|'us'|'ms'
    tz   = ts_type.tz

    # normalise ts vers UTC; garde-naïf si colonne sans tz
    if tz:
        ts = (ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC"))
    else:
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)

    epoch_ns = int(ts.value)  # ns since epoch
    if unit == "ns":
        val = epoch_ns
    elif unit == "us":
        val = epoch_ns // 1_000
    elif unit == "ms":
        val = epoch_ns // 1_000_000
    else:
        val = epoch_ns
    return pa.scalar(val, type=ts_type)

def _probe_windows(book_path: str, region: Optional[str],
                   probes: int, win_sec: int,
                   ts_min: pd.Timestamp, ts_max: pd.Timestamp) -> None:
    print(f"\n── BOOK: {probes} random window probes (±{win_sec}s)")

    fs = _s3_fs(region)
    dataset = ds.dataset(_s3_object_path(book_path), filesystem=fs, format="parquet")

    ts_field = dataset.schema.field("timestamp")
    ts_type  = ts_field.type

    probe_cols = ([
        "timestamp","received_time","seq",
        "bid_0_price","ask_0_price","bid_0_size","ask_0_size"
    ] + [f"bid_{i}_price" for i in range(BOOK_LEVELS)]
      + [f"ask_{i}_price" for i in range(BOOK_LEVELS)])

    for k in range(probes):
        t  = ts_min + (ts_max - ts_min) * random.random()
        t  = pd.Timestamp(t).tz_convert("UTC")
        t0 = t - pd.Timedelta(seconds=win_sec)
        t1 = t + pd.Timedelta(seconds=win_sec)

        s0 = _ts_scalar_for(t0, ts_type)
        s1 = _ts_scalar_for(t1, ts_type)
        filt = (ds.field("timestamp") >= s0) & (ds.field("timestamp") <= s1)

        try:
            tbl = dataset.to_table(columns=probe_cols, filter=filt)
            dfw = tbl.to_pandas(types_mapper=pd.ArrowDtype)
        except pa.ArrowNotImplementedError:
            # Fallback : on lit les colonnes puis on filtre côté pandas (fenêtre minuscule)
            tbl = dataset.to_table(columns=probe_cols)
            dfw = tbl.to_pandas(types_mapper=pd.ArrowDtype)
            ts_series = pd.to_datetime(dfw["timestamp"], utc=True, errors="coerce")
            dfw = dfw[(ts_series >= t0) & (ts_series <= t1)].copy()

        dfw["timestamp"] = pd.to_datetime(dfw["timestamp"], utc=True, errors="coerce")
        if "received_time" in dfw.columns:
            dfw["received_time"] = pd.to_datetime(dfw["received_time"], utc=True, errors="coerce")
        dfw.sort_values("timestamp", inplace=True, kind="stable")

        print(
            f"probe {k+1}: window rows={len(dfw)}  t≈{_fmt_dt(t)}  "
            f"first/last ts={_fmt_dt(dfw['timestamp'].min() if not dfw.empty else None)} / "
            f"{_fmt_dt(dfw['timestamp'].max() if not dfw.empty else None)}  "
            f"first/last rt={_fmt_dt(dfw['received_time'].min() if ('received_time' in dfw and not dfw.empty) else None)} / "
            f"{_fmt_dt(dfw['received_time'].max() if ('received_time' in dfw and not dfw.empty) else None)}"
        )

# ─────────────────────────── TRADES checks ───────────────────────────────────

def _normalize_trades_view(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "timestamp" not in df.columns:
        if "origin_time" in df.columns:
            df["timestamp"] = _ensure_utc(df["origin_time"])
        elif "received_time" in df.columns:
            df["timestamp"] = _ensure_utc(df["received_time"])
    else:
        df["timestamp"] = _ensure_utc(df["timestamp"])
    if "received_time" in df.columns:
        df["received_time"] = _ensure_utc(df["received_time"])
    if "price" not in df.columns and "last_price" in df.columns:
        df = df.rename(columns={"last_price":"price"})
    if "quantity" in df.columns and "qty" not in df.columns:
        df = df.rename(columns={"quantity":"qty"})
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    keep = [c for c in TRADE_COLS_MIN if c in df.columns]
    return df[keep].dropna(subset=["timestamp","price","qty"]).sort_values("timestamp", kind="stable") if keep else pd.DataFrame(columns=list(TRADE_COLS_MIN))

def _validate_trades_month(tr_path: str, region: Optional[str], n: int) -> None:
    print(f"\n── TRADES: {tr_path}")
    cols = ["timestamp","received_time","price","qty"]
    try:
        view = _read_parquet_s3(tr_path, region, columns=cols)
    except Exception as e:
        print(f"⛔ cannot read trades: {type(e).__name__}: {e}")
        return
    if view.empty:
        print("⚠️ normalized view empty (check schema)")
        return
    have = set(view.columns); need = {"timestamp","price","qty"}
    if not need.issubset(have):
        print(f"⛔ missing required trade cols: {sorted(list(need - have))}")
        return
    view = _normalize_trades_view(view)
    print("✅ timestamp/price/qty present")
    print(f"time span: {_fmt_dt(view['timestamp'].min())} → {_fmt_dt(view['timestamp'].max())}")
    print(f"price<=0: {int((view['price']<=0).sum())}, qty<=0: {int((view['qty']<=0).sum())}")

    if n>0:
        head = view.head(n).copy()
        head["timestamp"] = head["timestamp"].map(_fmt_dt)
        if "received_time" in head.columns:
            head["received_time"] = head["received_time"].map(_fmt_dt)
        print("sample:")
        print(head.to_string(index=False))

# ─────────────────────────────── CLI ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Validate BOOK(15) / TRADES monthly Parquet on S3 (RAM-safe, received_time-aware).")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--ym", default="2024-06", help="YYYY-MM (e.g., 2023-01)")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--root-book", default="s3://tradebot-config-tokyo/data/book")
    ap.add_argument("--root-trade", default="s3://tradebot-config-tokyo/data/trade")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--probe-window-sec", type=int, default=5)
    args = ap.parse_args()

    y_s, m_s = args.ym.split("-")
    y, m = int(y_s), int(m_s)
    book_path  = _book_month_path(args.root_book, args.symbol, y, m)
    trade_path = _trade_month_path(args.root_trade, args.symbol, y, m) if args.root_trade else None

    if not _exists_s3(book_path, args.aws_region):
        print(f"⛔ BOOK file not found: {book_path}")
        return
    if trade_path and not _exists_s3(trade_path, args.aws_region):
        print(f"⛔ TRADES file not found: {trade_path}")

    print("════════ validate_micro_io (schema & coherence only) ════════")
    print(f"symbol={args.symbol} month={args.ym}")
    print(f"BOOK  = {book_path}")
    if trade_path: print(f"TRADES = {trade_path}")
    print("─────────────────────────────────────────────────────────────")

    # (A) Détection profondeur via schéma (zéro data)
    (b_min,b_max),(a_min,a_max) = _detect_levels_from_schema(book_path, args.aws_region)
    b_rng = f"{b_min if b_min>=0 else 'n/a'}..{b_max if b_max>=0 else 'n/a'}"
    a_rng = f"{a_min if a_min>=0 else 'n/a'}..{a_max if a_max>=0 else 'n/a'}"
    print(f"Detected bid levels (schema): {b_rng}")
    print(f"Detected ask levels (schema): {a_rng}")
    if b_max < BOOK_LEVELS-1 or a_max < BOOK_LEVELS-1:
        print(f"⚠️ depth: bid max={b_max}, ask max={a_max} (expected {BOOK_LEVELS-1})")

    # (B) BOOK — lecture minimale top-of-book
    base_cols = ["timestamp","received_time","seq","bid_0_price","ask_0_price","bid_0_size","ask_0_size"]
    try:
        df_book = _read_parquet_s3(book_path, args.aws_region, columns=base_cols)
    except Exception as e:
        print(f"⛔ cannot read book parquet: {type(e).__name__}: {e}")
        return
    if df_book.empty:
        print("⚠️ BOOK file empty"); return

    _basic_book_checks(df_book)

    # (C) Probes — petites fenêtres avec tous les niveaux de prix (RAM-safe)
    ts_min, ts_max = df_book["timestamp"].min(), df_book["timestamp"].max()
    if pd.notna(ts_min) and pd.notna(ts_max) and ts_min < ts_max and args.probes>0:
        _probe_windows(book_path, args.aws_region, args.probes, args.probe_window_sec, ts_min, ts_max)

    # (D) TRADES (optionnel)
    if trade_path:
        _validate_trades_month(trade_path, args.aws_region, args.n)

    # (E) Samples (top-of-book uniquement)
    print("\n── BOOK sample")
    head = df_book[["timestamp","received_time","bid_0_price","ask_0_price","bid_0_size","ask_0_size"]].head(args.n).copy()
    head["timestamp"] = head["timestamp"].map(_fmt_dt)
    head["received_time"] = head["received_time"].map(_fmt_dt)
    print(head.to_string(index=False))
    print("\nDone.")

if __name__ == "__main__":
    main()