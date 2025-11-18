# next_bot/feeds/replay_feed.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import os
import pandas as pd
import numpy as np
from ml_bot.core.utils.trade_logger import logger_backtest

# S3 support
try:
    import fsspec
except Exception:
    fsspec = None  # sera None si non installé (cas local strict)

# --- Helpers de base ---

_PANDAS_FREQ_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h"
}

def _to_pd_freq(freq: str) -> str:
    return _PANDAS_FREQ_MAP.get(freq, freq)

def _is_s3_uri(p: str) -> bool:
    return str(p).startswith("s3://")

def _storage_options_for_s3(region: str | None) -> dict:
    # IAM Task Role recommandé (anon=False)
    so = {"anon": False}
    if region:
        so["client_kwargs"] = {"region_name": region}
    return so

def _resample_ohlcv(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    logger_backtest.debug(f"[resample] start tf={tf} rows={len(df_1m)}")
    df = df_1m.set_index("timestamp")
    agg = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume":"sum",
    }
    out = df.resample(tf).agg(agg).dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    logger_backtest.debug(f"[resample] done tf={tf} -> rows={len(out)}")
    return out

def _year_range(start: pd.Timestamp, end: pd.Timestamp) -> List[int]:
    years = list(range(start.year, end.year + 1))
    logger_backtest.debug(f"[years] start={start} end={end} -> {years}")
    return years

def _year_paths(root_parquet_dir: str, symbol: str, start: str, end: str) -> List[str]:
    """
    Construit des chemins annuels pour les bougies 1m :

      S3  : s3://.../data/bougie/<SYMBOL>/<SYMBOL>-1m-<YEAR>.parquet
      Local : <root>/<SYMBOL>/<SYMBOL>-1m-<YEAR>.parquet

    NOTE: root_parquet_dir doit pointer sur la racine "bougie",
          ex: s3://tradebot-config-tokyo/data/bougie
    """
    years = _year_range(pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True))

    if _is_s3_uri(root_parquet_dir):
        base = root_parquet_dir.rstrip("/") + f"/{symbol}"
        paths = [f"{base}/{symbol}-1m-{y}.parquet" for y in years]
        logger_backtest.debug(f"[paths:S3] {symbol} -> {paths}")
        return paths
    else:
        base = Path(root_parquet_dir) / symbol
        paths = [str((base / f"{symbol}-1m-{y}.parquet").resolve()) for y in years]
        logger_backtest.debug(f"[paths:local] {symbol} -> {paths}")
        return paths
    
def _load_parquet_concat(paths: List[str], aws_region: str | None) -> pd.DataFrame:
    if not paths:
        logger_backtest.warning("[load] paths is empty → returning empty DataFrame")
        return pd.DataFrame()

    cols = ["timestamp","open","high","low","close","volume"]

    # S3 mode
    if _is_s3_uri(paths[0]):
        if fsspec is None:
            logger_backtest.error("s3fs/fsspec non installés. Installe: pip install s3fs pyarrow")
            raise RuntimeError("s3fs/fsspec non installés. Installe: pip install s3fs pyarrow")
        so = _storage_options_for_s3(aws_region)
        try:
            fs = fsspec.filesystem(
                "s3",
                **({k: v for k, v in so.items() if k != "client_kwargs"} |
                   ({"client_kwargs": so["client_kwargs"]} if "client_kwargs" in so else {}))
            )
        except Exception as e:
            logger_backtest.error(f"[load:S3] init filesystem failed: {e}")
            raise

        dfs: List[pd.DataFrame] = []
        missing: List[str] = []
        for p in paths:
            try:
                if fs.exists(p):
                    logger_backtest.debug(f"[load:S3] reading {p}")
                    dfs.append(pd.read_parquet(p, columns=cols, storage_options=so, engine="pyarrow"))
                else:
                    missing.append(p)
            except Exception as e:
                logger_backtest.error(f"[load:S3] read failed {p}: {e}")
                raise
        if missing:
            logger_backtest.warning(f"[load:S3] missing files: {missing}")
        if not dfs:
            logger_backtest.warning("[load:S3] no DataFrame loaded (all missing?)")
            return pd.DataFrame()
        out = pd.concat(dfs, ignore_index=True)
        logger_backtest.info(f"[load:S3] loaded rows={len(out)} from {len(dfs)} files")
        return out

    # Local mode
    dfs = []
    missing: List[str] = []
    for p in paths:
        if os.path.exists(p):
            logger_backtest.debug(f"[load:local] reading {p}")
            try:
                dfs.append(pd.read_parquet(p, columns=cols, engine="pyarrow"))
            except Exception as e:
                logger_backtest.error(f"[load:local] read failed {p}: {e}")
                raise
        else:
            missing.append(p)
    if missing:
        logger_backtest.warning(f"[load:local] missing files: {missing}")
    if not dfs:
        logger_backtest.warning("[load:local] no DataFrame loaded (all missing?)")
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    logger_backtest.info(f"[load:local] loaded rows={len(out)} from {len(dfs)} files")
    return out

def _normalize_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Valide/convertit le schéma attendu et normalise timestamp ms→UTC."""
    required = ["timestamp","open","high","low","close","volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}. Attendu={required}")

    # timestamp → datetime UTC
    ts = df["timestamp"]
    try_ms = False

    if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
        try_ms = True
    elif ts.dtype == object:
        # si ce sont des chaînes numériques longues (>= 12 digits) → ms
        sample = ts.dropna().astype(str).head(10)
        if len(sample) and all(s.isdigit() and len(s) >= 12 for s in sample):
            try_ms = True

    if try_ms:
        df["timestamp"] = pd.to_datetime(ts.astype("int64"), unit="ms", utc=True, errors="coerce")
        logger_backtest.debug("[normalize] timestamp interpreted as epoch-ms")
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
        logger_backtest.debug("[normalize] timestamp parsed as ISO-like string")

    before = len(df)
    df = df.dropna(subset=["timestamp"]).copy()
    dropped_ts = before - len(df)
    if dropped_ts:
        logger_backtest.warning(f"[normalize] dropped {dropped_ts} rows with invalid timestamp")

    # types numériques pour OHLCV
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["open","high","low","close"]).copy()
    dropped_ohlc = before - len(df)
    if dropped_ohlc:
        logger_backtest.warning(f"[normalize] dropped {dropped_ohlc} rows with invalid OHLC")

    # tri + dédup
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    return df

@dataclass
class ReplayConfig:
    root_parquet_dir: str                 # e.g. s3://tradebot-config-tokyo/data/bougie ou chemin local
    symbol: str                           # e.g. BTCUSDT
    start: str                            # ISO date string UTC
    end: str                              # ISO date string UTC (exclusive)
    freqs: List[str]                      # e.g. ["1m","3m","5m","15m","30m","1h"]
    warmup_bars_by_freq: Dict[str, int]   # e.g. {"1m":600,"3m":400,...}
    window_bars_by_freq: Dict[str, int]   # e.g. {"1m":120,"3m":60,...}
    aws_region: str | None = None         # e.g. "ap-northeast-1"
    step_seconds: int = 60                # avancer de 60s entre fenêtres

class ReplayFeed:
    """
    Charge les Parquet 1m (par année) depuis LOCAL ou S3 (bougies),
    calcule les séries resamplées, et itère des fenêtres synchronisées.
    """
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.logger = logger_backtest
        self.logger.info(f"[ReplayFeed] init symbol={cfg.symbol} root={cfg.root_parquet_dir} region={cfg.aws_region} freqs={cfg.freqs}")
        self._base_1m = self._load_1m()
        self._by_freq: Dict[str, pd.DataFrame] = self._build_freq_views()

    def _load_1m(self) -> pd.DataFrame:
        symbol = self.cfg.symbol
        start = pd.Timestamp(self.cfg.start, tz="UTC")
        end   = pd.Timestamp(self.cfg.end,   tz="UTC")

        paths = _year_paths(self.cfg.root_parquet_dir, symbol, self.cfg.start, self.cfg.end)
        self.logger.debug(f"[load_1m] candidate files: {paths}")
        df = _load_parquet_concat(paths, self.cfg.aws_region)

        if df.empty:
            where = "S3" if _is_s3_uri(self.cfg.root_parquet_dir) else "local"
            msg = f"Aucun Parquet {where} trouvé pour {symbol} entre {paths[0]} … {paths[-1]}"
            self.logger.error(f"[load_1m] {msg}")
            raise FileNotFoundError(msg)

        # Normalisation / tri / filtre plage
        df = _normalize_ohlcv_schema(df)

        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
        self.logger.info(f"[load_1m] {symbol} rows={len(df)} window=[{start} → {end})")
        if df.empty:
            self.logger.error(f"[load_1m] {symbol} empty after date filter")
            raise FileNotFoundError(f"{symbol}: empty after date filter {start}→{end}")
        return df

    def _build_freq_views(self) -> Dict[str, pd.DataFrame]:
        by_freq: Dict[str, pd.DataFrame] = {}
        # 1m de base
        by_freq["1m"] = self._base_1m.copy()
        first = by_freq["1m"]["timestamp"].iloc[0]
        last  = by_freq["1m"]["timestamp"].iloc[-1]
        self.logger.info(f"[build_views] 1m rows={len(by_freq['1m'])} range=[{first} → {last}]")

        # autres TF
        for f in self.cfg.freqs:
            if f == "1m":
                continue
            pd_freq = _to_pd_freq(f)
            try:
                by_freq[f] = _resample_ohlcv(self._base_1m, pd_freq)
                self.logger.info(f"[build_views] {f} rows={len(by_freq[f])}")
            except Exception as e:
                self.logger.error(f"[build_views] resample failed for {f}: {e}")
                raise
        return by_freq

    def iter_windows(self) -> Iterator[Tuple[pd.Timestamp, Dict[str, pd.DataFrame]]]:
        """
        Itère sur des ancres temporelles t, et retourne pour chaque t :
        {freq -> df_window (les N dernières bougies selon window_bars_by_freq[freq])}
        Ne yield que quand toutes les fréquences ont suffisamment de warmup + window.
        """
        cfg = self.cfg
        base = self._by_freq["1m"]
        if base.empty:
            self.logger.warning("[iter] base 1m empty → stop")
            return

        t_start = base["timestamp"].iloc[0]
        t_end   = base["timestamp"].iloc[-1]
        self.logger.info(f"[iter] start={t_start} end={t_end} step={cfg.step_seconds}s")

        # ancre initiale
        t = t_start

        # index rapides par freq
        indexed = {f: df.set_index("timestamp") for f, df in self._by_freq.items()}

        step = pd.Timedelta(seconds=cfg.step_seconds)
        yielded = 0
        checked = 0

        while t <= t_end:
            windows: Dict[str, pd.DataFrame] = {}
            ok = True
            for f in cfg.freqs:
                pd_freq = _to_pd_freq(f)
                bucket = t.floor(pd_freq)

                df_f = indexed[f]
                df_upto = df_f.loc[:bucket]
                if df_upto.empty:
                    self.logger.debug(f"[iter] {f}@{bucket} empty upto → not ready")
                    ok = False
                    break

                warmup = cfg.warmup_bars_by_freq.get(f, 0)
                need   = cfg.window_bars_by_freq.get(f, 1)

                if len(df_upto) < (warmup + need):
                    # trop tôt pour cette TF
                    self.logger.debug(f"[iter] {f}@{bucket} not enough bars: have={len(df_upto)} need={warmup+need}")
                    ok = False
                    break

                # fenêtre = les N dernières barres (pas le warmup)
                df_win = df_upto.iloc[-need:].reset_index()
                windows[f] = df_win

            checked += 1
            if ok:
                yielded += 1
                if yielded % 1000 == 0:
                    self.logger.info(f"[iter] yielded={yielded} checked={checked} t={t}")
                yield (t, windows)

            t = t + step

        self.logger.info(f"[iter] done: yielded={yielded} checked={checked}")