from __future__ import annotations
import json
import uuid
import pandas as pd


def read_parquet_s3(path: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, storage_options={"anon": False})


def write_parquet_s3(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False, engine="pyarrow", storage_options={"anon": False})


def write_json_s3(obj: dict, path: str) -> None:
    import fsspec
    fs = fsspec.filesystem("s3")
    with fs.open(path.replace("s3://", ""), "wb") as f:
        f.write(json.dumps(obj, indent=2).encode("utf-8"))


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"
