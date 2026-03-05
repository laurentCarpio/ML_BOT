from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec

from .s3_paths import book_month_path
from .utils import next_month_str
from .windows import overlaps_any


def _schema_names(pf: pq.ParquetFile):
    # robust: arrow schema
    try:
        return list(pf.schema_arrow.names)
    except Exception:
        return list(pf.schema.names)


def _col_index(pf: pq.ParquetFile, col: str) -> int:
    # use arrow schema when available (fixes ParquetSchema.get_field_index issue)
    try:
        return pf.schema_arrow.get_field_index(col)
    except Exception:
        names = _schema_names(pf)
        return names.index(col)


def read_book_month_windows(
    book_root: str,
    symbol: str,
    month: str,
    win_s: np.ndarray,
    win_e: np.ndarray,
    fs=None,
) -> pd.DataFrame:
    """
    Reads ONLY row-groups overlapping windows; keeps L1 bid/ask and mid.
    Expects bid_0_price / ask_0_price in schema (your case).
    """
    if fs is None:
        fs = fsspec.filesystem("s3")

    path = book_month_path(book_root, symbol, month)
    p_no = path.replace("s3://", "")
    pf = pq.ParquetFile(p_no, filesystem=fs)

    cols = set(_schema_names(pf))
    ts_col = "timestamp" if "timestamp" in cols else ("ts" if "ts" in cols else None)
    if ts_col is None:
        raise ValueError(f"{path}: no timestamp column")

    bid_col = "bid_0_price" if "bid_0_price" in cols else None
    ask_col = "ask_0_price" if "ask_0_price" in cols else None
    if bid_col is None or ask_col is None:
        raise ValueError(f"{path}: missing bid_0_price/ask_0_price")

    ts_idx = _col_index(pf, ts_col)

    # -----------------------------------------
    # 1) select row-groups to read (no reads yet)
    # -----------------------------------------
    rg_keep: list[int] = []
    for rg in range(pf.num_row_groups):
        col_md = pf.metadata.row_group(rg).column(ts_idx)
        st = col_md.statistics

        # if no stats, we must keep (safe)
        if st is None:
            rg_keep.append(rg)
            continue

        rg_min = pd.to_datetime(st.min, utc=True, errors="coerce")
        rg_max = pd.to_datetime(st.max, utc=True, errors="coerce")
        if pd.isna(rg_min) or pd.isna(rg_max):
            rg_keep.append(rg)
            continue

        a_min = rg_min.to_datetime64()
        a_max = rg_max.to_datetime64()
        if overlaps_any(a_min, a_max, win_s, win_e):
            rg_keep.append(rg)

    if not rg_keep:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "mid"])

    # -----------------------------------------
    # 2) read all selected row-groups at once
    # -----------------------------------------
    tbl = pf.read_row_groups(rg_keep, columns=[ts_col, bid_col, ask_col])

    # If for any reason arrow gives empty table:
    if tbl is None or tbl.num_rows == 0:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "mid"])

    # 1 conversion to pandas
    df = tbl.to_pandas(ignore_metadata=True)
    df = df.rename(columns={ts_col: "timestamp", bid_col: "bid", ask_col: "ask"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")

    df = df.dropna(subset=["timestamp", "bid", "ask"])
    df = df.sort_values("timestamp")
    df = df.loc[(df["bid"] > 0) & (df["ask"] > 0)]
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # final exact window filter (important)
    ts = df["timestamp"].values.astype("datetime64[ns]")
    mask = np.zeros(len(df), dtype=bool)
    for s, e in zip(win_s, win_e):
        mask |= (ts >= s) & (ts <= e)

    out = df.loc[mask, ["timestamp", "bid", "ask", "mid"]]
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "mid"])
    return out


def read_book_with_next_windows(
    book_root: str,
    symbol: str,
    month: str,
    win_s: np.ndarray,
    win_e: np.ndarray,
    fs=None,
) -> pd.DataFrame:
    b0 = read_book_month_windows(book_root, symbol, month, win_s, win_e, fs=fs)

    m1 = next_month_str(month)
    try:
        b1 = read_book_month_windows(book_root, symbol, m1, win_s, win_e, fs=fs)
    except Exception:
        b1 = None

    # ✅ avoid FutureWarning: drop empty OR all-NA frames safely
    frames = []
    for df in (b0, b1):
        if df is None:
            continue
        if len(df) == 0:
            continue
        # "all-NA entries" protection
        if df.dropna(how="all").empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "mid"])

    return pd.concat(frames, ignore_index=True).sort_values("timestamp")