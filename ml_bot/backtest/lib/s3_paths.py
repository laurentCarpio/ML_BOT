from __future__ import annotations

def events_tagged_month_path(events_tagged_root: str, month: str) -> str:
    # ex: s3://.../data/v1/wf_stage0_tagged_2025-05.parquet
    return f"{events_tagged_root}/wf_stage0_tagged_{month}.parquet"

def events_evalready_month_path(events_evalready_root: str, month: str) -> str:
    # ex: s3://.../data/v1_evalready/wf_stage0_evalready_2025-05.parquet
    return f"{events_evalready_root}/wf_stage0_evalready_{month}.parquet"

def book_month_path(book_root: str, symbol: str, month: str) -> str:
    # ex: s3://.../data/book/BTCUSDT/2025-05.parquet
    return f"{book_root}/{symbol}/{month}.parquet"

def candles_year_path(candles_root: str, symbol: str, year: int) -> str:
    # ex: s3://.../data/bougie/BTCUSDT/BTCUSDT-1m-2025.parquet
    return f"{candles_root}/{symbol}/{symbol}-1m-{year}.parquet"