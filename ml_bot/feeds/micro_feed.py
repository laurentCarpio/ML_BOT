# next_bot/feeds/micro_feed.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict, Tuple as Tup
import pandas as pd
import numpy as np
import fsspec
import logging

import pyarrow as pa
import pyarrow.dataset as ds

log = logging.getLogger(__name__)

# Try to import the filesystem module explicitly
try:
    import pyarrow.fs as pa_fs
    ARROW_HAS_FS = True
except Exception:
    pa_fs = None
    ARROW_HAS_FS = False

# (optional but helpful) quick env log
try:
    import fsspec as _fsspec  # will be present via s3fs deps even if not pinned
    print(f"[micro_feed] pyarrow={getattr(pa, '__version__', 'unknown')} | "
          f"pyarrow.fs={'ok' if ARROW_HAS_FS else 'missing'} | "
          f"s3fs present | fsspec={getattr(_fsspec, '__version__', 'unknown')}")
except Exception:
    print(f"[micro_feed] pyarrow={getattr(pa, '__version__', 'unknown')} | "
          f"pyarrow.fs={'ok' if ARROW_HAS_FS else 'missing'} | s3fs/fsspec check failed")
    

# Colonnes attendues (format feeder)
TRADE_COLS_REQ = ("timestamp","price","qty","is_aggr_buy")
L1_COLS_REQ    = ("timestamp","best_bid","best_ask","bid_qty","ask_qty","spread")

def _ensure_utc(s: pd.Series) -> pd.Series:
    """
    Convertit en datetime64[ns, UTC].
    - tz-aware  -> tz_convert('UTC')
    - tz-naïf   -> tz_localize('UTC')
    - int/float -> auto 'ms' vs 'ns', puis UTC
    - object    -> to_datetime(..., utc=True)
    """
    import pandas.api.types as ptypes

    if ptypes.is_datetime64tz_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.tz_convert("UTC")

    if ptypes.is_datetime64_dtype(s):  # naïf
        s2 = pd.to_datetime(s, errors="coerce")
        return s2.dt.tz_localize("UTC")

    if ptypes.is_integer_dtype(s) or ptypes.is_float_dtype(s):
        # heuristic: <1e15 => ms, else ns
        med = pd.Series(s).astype("float64").abs().median() if len(s) else 0.0
        unit = "ms" if med < 1e15 else "ns"
        return pd.to_datetime(s, utc=True, unit=unit, errors="coerce")

    # object/str fallback
    return pd.to_datetime(s, utc=True, errors="coerce")

# --- utilitaires Arrow → Pandas pour lire une tranche filtrée ---

def _to_utc_scalar(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")

def _arrow_object_path(path: str) -> str:
    # ds.dataset(..., filesystem=S3FileSystem) attend "bucket/key", pas "s3://bucket/key"
    return path[5:] if path.startswith("s3://") else path

def _read_parquet_slice_arrow(
    path: str,
    columns: list[str],
    filter_expr: ds.Expression | None,
    aws_region: Optional[str]
) -> pd.DataFrame:
    if path.startswith("s3://") and ARROW_HAS_FS:
        fs = pa_fs.S3FileSystem(region=aws_region) if aws_region else pa_fs.S3FileSystem()
        root = _arrow_object_path(path)  # 'bucket/key...'
        dataset = ds.dataset(root, filesystem=fs, format="parquet")
    else:
        dataset = ds.dataset(path, format="parquet")

    table = dataset.to_table(columns=columns, filter=filter_expr) if filter_expr is not None \
            else dataset.to_table(columns=columns)
    if table.num_rows == 0:
        return pd.DataFrame(columns=columns)
    return table.to_pandas(types_mapper=None)

def _read_parquet(path: str, aws_region: Optional[str], columns: Optional[list[str]]=None) -> pd.DataFrame:
    so = {"client_kwargs": {"region_name": aws_region}} if (aws_region and path.startswith("s3://")) else None
    return pd.read_parquet(path, engine="pyarrow", columns=columns, storage_options=so) \
           if path.startswith("s3://") else \
           pd.read_parquet(path, engine="pyarrow", columns=columns)

def _downcast_inplace(df: pd.DataFrame, cols_float: Iterable[str], col_bool: Optional[str]=None):
    for c in cols_float:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if col_bool and (col_bool in df.columns):
        df[col_bool] = df[col_bool].astype(bool)

# Normalisation TRADES (identique à ce que tu avais, juste sans copy massive)
def _normalize_trades(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw  # pas de copy() pour limiter RAM

    if "timestamp" not in df.columns:
        if "origin_time" in df.columns:
            df = df.assign(timestamp=pd.to_datetime(df["origin_time"], utc=True, errors="coerce"))
        elif "received_time" in df.columns:
            df = df.assign(timestamp=pd.to_datetime(df["received_time"], utc=True, errors="coerce"))

    if "price" not in df.columns and "last_price" in df.columns:
        df = df.rename(columns={"last_price": "price"})
    if "qty" not in df.columns and "quantity" in df.columns:
        df = df.rename(columns={"quantity": "qty"})

    if "is_aggr_buy" not in df.columns:
        if "is_buyer_maker" in df.columns:
            df = df.assign(is_aggr_buy=~df["is_buyer_maker"].astype(bool))
        elif "side" in df.columns:
            s = df["side"].astype(str).str.lower()
            df = df.assign(is_aggr_buy=s.eq("buy"))
        else:
            df = df.assign(is_aggr_buy=np.nan)

    need = ["timestamp","price","qty","is_aggr_buy"]
    if not set(need).issubset(df.columns):
        return pd.DataFrame(columns=need)

    # downcast agressif (RAM)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")
    df["qty"]   = pd.to_numeric(df["qty"],   errors="coerce").astype("float32")
    # is_aggr_buy en bool natif (pas d’objets)
    df["is_aggr_buy"] = df["is_aggr_buy"].astype(bool)

    df = df.dropna(subset=["timestamp","price","qty"])

    # assure tz UTC
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        if getattr(df["timestamp"].dt, "tz", None) is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    out = df[need].sort_values("timestamp", kind="stable").reset_index(drop=True)
    return out

@dataclass
class MicroReplayConfig:
    root_l1_dir: str   = "s3://tradebot-config-tokyo/data/level_1"
    root_tr_dir: str   = "s3://tradebot-config-tokyo/data/trade"
    aws_region: Optional[str] = "ap-northeast-1"
    pre_seconds: int = 60
    post_seconds: int = 15
    # ↓ on n'utilise plus de cache de mois complet
    max_cached_months: int = 0

class MicroReplayFeed:
    """
    Feed micro qui lit L1 et trades **par mois** et fournit
    des slices temporels autour d’un timestamp t0 (UTC).
    - L1: fichiers mensuels déjà normalisés → simple lecture + ensure UTC
    - TR: fichiers mensuels bruts     → normalisation à la volée
    """
    def __init__(self, cfg: MicroReplayConfig, symbol: str):
        self.cfg = cfg
        self.symbol = symbol
        self._l1_month_cache: Dict[Tup[int,int], pd.DataFrame] = {}
        self._tr_month_cache: Dict[Tup[int,int], pd.DataFrame] = {}
        self._fs = fsspec.filesystem("s3", client_kwargs={"region_name": cfg.aws_region}) if str(cfg.root_l1_dir).startswith("s3://") else None

    def _read_l1_slice(self, year, month, t_start, t_end):
        p = self._path_l1(year, month)
        if not self._exists(p):
            return pd.DataFrame(columns=L1_COLS_REQ)

        cols = ["timestamp","best_bid","best_ask","bid_qty","ask_qty","spread"]

        if ARROW_HAS_FS:
            # Vérif stricte du schéma
            fs = pa_fs.S3FileSystem(region=self.cfg.aws_region) if (p.startswith("s3://")) else None
            if p.startswith("s3://"):
                root = _arrow_object_path(p)
                dataset = ds.dataset(root, filesystem=fs, format="parquet")
            else:
                dataset = ds.dataset(p, format="parquet")
            present = {f.name for f in dataset.schema}
            missing = [c for c in cols if c not in present]
            if missing:
                raise ValueError(f"[L1 {self.symbol} {year}-{month:02d}] colonnes manquantes: {missing} | présentes={sorted(present)}")

            # Filtre temporel (timestamp est tz=UTC dans tes L1)
            ts_start = int(t_start.value)  # ns
            ts_end   = int(t_end.value)
            filt = (ds.field("timestamp") >= pa.scalar(ts_start, type=pa.timestamp("ns", tz="UTC"))) & \
                (ds.field("timestamp") <= pa.scalar(ts_end,   type=pa.timestamp("ns", tz="UTC")))
            df = _read_parquet_slice_arrow(p, cols, filt, self.cfg.aws_region)
        else:
            df = _read_parquet(p, self.cfg.aws_region, columns=cols)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"[L1 {self.symbol} {year}-{month:02d}] colonnes manquantes: {missing}")
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = _ensure_utc(df["timestamp"])
            df = df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)]

        if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df.reset_index(drop=True)

    def _read_trades_slice(self, year: int, month: int, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
        p = self._path_tr(year, month)
        if not self._exists(p):
            return pd.DataFrame(columns=TRADE_COLS_REQ)

        # Schéma brut Crypto-Lake (validé par validate_micro_io)
        required = {"origin_time", "price", "quantity", "side"}
        optional = {"received_time", "trade_id", "symbol", "exchange", "dt", "is_buyer_maker"}

        if ARROW_HAS_FS:
            # Dataset Arrow + vérif stricte
            if p.startswith("s3://"):
                fs = pa_fs.S3FileSystem(region=self.cfg.aws_region) if self.cfg.aws_region else pa_fs.S3FileSystem()
                root = _arrow_object_path(p)
                dataset = ds.dataset(root, filesystem=fs, format="parquet")
            else:
                dataset = ds.dataset(p, format="parquet")

            present = {f.name for f in dataset.schema}
            missing_req = sorted(required - present)
            if missing_req:
                raise ValueError(
                    f"[TRADES {self.symbol} {year}-{month:02d}] colonnes REQUISES manquantes: {missing_req} | "
                    f"présentes={sorted(present)}"
                )

            cols_to_read = sorted(list(required | (optional & present)))

            # Filtre temporel Arrow sur origin_time (dtype: timestamp[ns] sans tz)
            ts_start = int(t_start.value)
            ts_end   = int(t_end.value)
            filt = (ds.field("origin_time") >= pa.scalar(ts_start, type=pa.timestamp("ns"))) & \
                (ds.field("origin_time") <= pa.scalar(ts_end,   type=pa.timestamp("ns")))

            df_raw = _read_parquet_slice_arrow(p, cols_to_read, filt, self.cfg.aws_region)
        else:
            # Fallback pandas: lecture complète + filtre
            df_raw = _read_parquet(p, self.cfg.aws_region, columns=None)
            present = set(df_raw.columns)
            missing_req = sorted(required - present)
            if missing_req:
                raise ValueError(
                    f"[TRADES {self.symbol} {year}-{month:02d}] colonnes REQUISES manquantes: {missing_req} | "
                    f"présentes={sorted(present)}"
                )
            if not pd.api.types.is_datetime64_any_dtype(df_raw["origin_time"]):
                df_raw["origin_time"] = _ensure_utc(df_raw["origin_time"])
            df_raw = df_raw[(df_raw["origin_time"] >= t_start) & (df_raw["origin_time"] <= t_end)]
            cols_to_read = sorted(list(required | (optional & present)))
            df_raw = df_raw[cols_to_read]

        if df_raw.empty:
            return pd.DataFrame(columns=TRADE_COLS_REQ)

        df_norm = _normalize_trades(df_raw)
        _downcast_inplace(df_norm, ["price","qty"], col_bool="is_aggr_buy")
        return df_norm.reset_index(drop=True)

    def _evict_except(self, keep_keys: Iterable[Tup[int,int]]):
        keep = set(keep_keys)
        for k in list(self._l1_month_cache.keys()):
            if k not in keep and len(self._l1_month_cache) > self.cfg.max_cached_months:
                del self._l1_month_cache[k]
        for k in list(self._tr_month_cache.keys()):
            if k not in keep and len(self._tr_month_cache) > self.cfg.max_cached_months:
                del self._tr_month_cache[k]

    def _path_l1(self, year: int, month: int) -> str:
        return f"{self.cfg.root_l1_dir.rstrip('/')}/{self.symbol}/{year:04d}-{month:02d}.parquet"
    
    def _path_tr(self, year: int, month: int) -> str:
        return f"{self.cfg.root_tr_dir.rstrip('/')}/{self.symbol}/{year:04d}-{month:02d}.parquet"

    def _exists(self, path: str) -> bool:
        if path.startswith("s3://"):
            return self._fs.exists(path) if self._fs else False
        return Path(path).exists()

    def _load_month(self, year: int, month: int):
        key = (year, month)

        # ---------- L1 (déjà normalisé) ----------
        if key not in self._l1_month_cache:
            p = self._path_l1(year, month)
            if not self._exists(p):
                self._l1_month_cache[key] = pd.DataFrame(columns=L1_COLS_REQ)
            else:
                raw = _read_parquet(p, self.cfg.aws_region, columns=list(L1_COLS_REQ))
                # assure tz UTC pour timestamp
                if not pd.api.types.is_datetime64_any_dtype(raw["timestamp"]):
                    raw["timestamp"] = _ensure_utc(raw["timestamp"])
                elif getattr(raw["timestamp"].dt, "tz", None) is None:
                    raw["timestamp"] = raw["timestamp"].dt.tz_localize("UTC")
                else:
                    raw["timestamp"] = raw["timestamp"].dt.tz_convert("UTC")
                # downcast léger
                _downcast_inplace(raw, ["best_bid","best_ask","bid_qty","ask_qty","spread"])
                self._l1_month_cache[key] = raw

        # ---------- TRADES (normalisation à la volée) ----------
        if key not in self._tr_month_cache:
            p = self._path_tr(year, month)
            if not self._exists(p):
                self._tr_month_cache[key] = pd.DataFrame(columns=TRADE_COLS_REQ)
            else:
                raw = _read_parquet(p, self.cfg.aws_region)  # lire brut
                trn = _normalize_trades(raw)                # -> format feeder
                self._tr_month_cache[key] = trn

    def _months_covering(self, t_start: pd.Timestamp, t_end: pd.Timestamp) -> list[tuple[int,int]]:
        months = []
        cur = pd.Timestamp(year=t_start.year, month=t_start.month, day=1, tz="UTC")
        endm = pd.Timestamp(year=t_end.year, month=t_end.month, day=1, tz="UTC")
        while cur <= endm:
            months.append((cur.year, cur.month))
            cur = (cur + pd.offsets.MonthBegin(1))
        return months

    def get_window(self, t0, *, pre_seconds: Optional[int]=None, post_seconds: Optional[int]=None
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pre = int(self.cfg.pre_seconds if pre_seconds is None else pre_seconds)
        post = int(self.cfg.post_seconds if post_seconds is None else post_seconds)

        # t0 → UTC
        t0 = _to_utc_scalar(t0)
        t_start = t0 - pd.Timedelta(seconds=pre)
        t_end   = t0 + pd.Timedelta(seconds=post)

        months = self._months_covering(t_start, t_end)
        if not months:
            return (pd.DataFrame(columns=TRADE_COLS_REQ),
                    pd.DataFrame(columns=L1_COLS_REQ))

        # Lire slicé par mois, puis concat
        tr_parts, l1_parts = [], []
        for y, m in months:
            l1_part = self._read_l1_slice(y, m, t_start, t_end)
            if not l1_part.empty:
                l1_parts.append(l1_part)
            tr_part = self._read_trades_slice(y, m, t_start, t_end)
            if not tr_part.empty:
                tr_parts.append(tr_part)

        tr = pd.concat(tr_parts, ignore_index=True, copy=False) if tr_parts else pd.DataFrame(columns=TRADE_COLS_REQ)
        l1 = pd.concat(l1_parts, ignore_index=True, copy=False) if l1_parts else pd.DataFrame(columns=L1_COLS_REQ)

        # Double-sécurité (au cas où la fenêtre chevauche 2 mois de façon non alignée)
        if not tr.empty:
            tr = tr[(tr["timestamp"] >= t_start) & (tr["timestamp"] <= t_end)].reset_index(drop=True)
        if not l1.empty:
            l1 = l1[(l1["timestamp"] >= t_start) & (l1["timestamp"] <= t_end)].reset_index(drop=True)

        return tr, l1