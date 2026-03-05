from __future__ import annotations
import json
import pandas as pd

from ml_bot.backtest.lib.io_s3 import read_parquet_s3, write_parquet_s3
from ml_bot.backtest.lib.s3_paths import events_tagged_month_path, events_evalready_month_path


EVALREADY_COLS = ["timestamp", "dir0", "pass_all_hard", "fees_rt_bps", "slip_bps", "MS"]


def build_evalready_month(cfg: dict, month: str) -> str:
    """
    Reads tagged month on S3, writes evalready month on S3.
    Returns output path.
    """
    src = events_tagged_month_path(cfg["events_tagged_root"], month)
    dst = events_evalready_month_path(cfg["events_evalready_root"], month)

    df = read_parquet_s3(src, columns=None)
    # minimal required columns (some months might contain extra cols)
    missing = [c for c in EVALREADY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{src}: missing columns {missing}")

    out = df.loc[df["pass_all_hard"] == True, EVALREADY_COLS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["fees_rt_bps"] = pd.to_numeric(out["fees_rt_bps"], errors="coerce")
    out["slip_bps"] = pd.to_numeric(out["slip_bps"], errors="coerce")
    out["dir0"] = pd.to_numeric(out["dir0"], errors="coerce")

    out = out.dropna(subset=["timestamp", "dir0", "fees_rt_bps", "slip_bps"])
    write_parquet_s3(out, dst)
    return dst