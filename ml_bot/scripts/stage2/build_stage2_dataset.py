#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage2_dataset.py — Stage2 (microstructure + label Y) — memory-safe

Entrées (S3 uniquement):
  - Signals: s3://.../report/<SYMBOL>/<YEAR>_signals.csv
  - Bougies: s3://.../bougie/<SYMBOL>/<SYMBOL>-1m-<YEAR>.parquet   (annuel)
  - Book15 : s3://.../level_1/<SYMBOL>/<YEAR>-*.parquet             (mensuel)
  - Trades : s3://.../trade/<SYMBOL>/<YEAR>-*.parquet               (mensuel)

Sorties:
  - s3://.../stage2/<SPLIT>/<SYMBOL>/<YEAR>/parts/part-*.parquet
"""

from __future__ import annotations
import argparse, re, gc, os, itertools
from typing import Optional, Dict, Any, List, Set
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs as pa_fs
import sys, fsspec
from datetime import datetime
from calendar import monthrange
from pyarrow import types as patypes
from pandas.api.types import is_numeric_dtype
import json
from collections import defaultdict

_MONTH_RE = re.compile(r"(\d{4})-(\d{2})")
BOOK_LEVELS = 15

# Par défaut: 16 symbols & 4 TF (tu peux toujours passer --symbols / --tfs au CLI)
DEFAULT_SYMBOLS = [
    "APTUSDT","BCHUSDT","BNBUSDT","BTCUSDT","CRVUSDT","DOGEUSDT","DOTUSDT","ETHUSDT",
    "LINKUSDT","LTCUSDT","OPUSDT","SOLUSDT","SUIUSDT","WLDUSDT","XRPUSDT"
]

DEFAULT_TFS = ["5m","15m","30m","1h","2h","4h"]

# Colonnes meta/label à exclure de la normalisation
EXCLUDE_FROM_NORM = {
    "t","symbol","year","tf","side","entry","horizon_sec","Y",
    # Labels/outcomes & coûts
    "pnl_net_max_bps","pnl_net_min_bps","THRESH_BPS",
    # (Optionnel) si tu préfères ne PAS normaliser ces features de coût/latence :
    # "fees_bps"
}

# === Ajout recommandé (étend l’exclusion sans écraser) ===
EXCLUDE_FROM_NORM |= {
    "has_trades_win10s", "has_trades_win15s",
    "trades_win_sec_used", "horizon_sec",
    # Flags event-driven (on ne les normalise pas)
    "best_bid_refill_rate_isnan",
    "bid_absorb_ratio_3s_isnan",
    "bid_absorb_ratio_10s_isnan",
}

TRADE_WIN_DEFAULT = [15, 30, 60, 120]

TRADE_WIN_OVERRIDES = {
    "CRVUSDT": [30, 60, 120, 240, 300],   # +long pour silence tape
    "MKRUSDT": [30, 60, 120, 240, 300],
}

# multiplicateur léger selon TF (ex: 4h peut accepter +x1.5)
TRADE_WIN_TF_FACTOR = {
    "5m": 1.0,
    "15m": 1.0,
    "30m": 1.25,
    "1h": 1.25,
    "2h": 1.5,
    "4h": 1.5,
}


# ------------------------------
# Trades helpers (factorisés)
# ------------------------------
def _clip_ratio_nonneg(x: float, upper: float = 100.0) -> float:
    """
    Petit helper pour ratios: renvoie NaN si non-fini, sinon clamp dans [0, upper].
    """
    if not np.isfinite(x):
        return np.nan
    if x < 0:
        x = 0.0
    return float(min(x, upper))

def _aggr_ratio(dfw: pd.DataFrame) -> float:
    """
    Retourne le ratio d'agression acheteuse = buy_qty / (buy_qty + sell_qty)
    dfw est censé être déjà agrégé à 1s (colonnes buy_qty/sell_qty).
    """
    if dfw is None or dfw.empty:
        return np.nan
    b = float(dfw.get("buy_qty",  pd.Series(dtype=float)).sum())
    s = float(dfw.get("sell_qty", pd.Series(dtype=float)).sum())
    den = b + s
    return (b/den) if den > 0 else np.nan

def _trade_windows_for(symbol: str, tf: str) -> list[int]:
    """
    Combine l'override par symbole + un facteur léger par TF,
    déduplique et trie.
    """
    base = TRADE_WIN_OVERRIDES.get(symbol, TRADE_WIN_DEFAULT)
    fac = TRADE_WIN_TF_FACTOR.get(tf, 1.0)
    wins = sorted({ max(1, int(round(w * fac))) for w in base })
    return wins

# Welford: stats incrémentales (mean, var, count)
class _Welford:
    __slots__ = ("n","mean","M2")
    def __init__(self): self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update_arr(self, arr: np.ndarray):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return
        for x in arr:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (x - self.mean)
    def result(self):
        if self.n < 2: return float(self.mean), float("nan")
        var = self.M2 / (self.n - 1)
        return float(self.mean), float(np.sqrt(max(var, 1e-30)))

# Stats normalisation par TF
class NormStats:
    # store: dict[tf][col] = {"mean":..., "std":...}
    def __init__(self): self.stats = defaultdict(dict)
    def save(self, path: str, so: dict):
        data = {"by_tf": self.stats}
        with fsspec.open(_normalize_url(path), "w", **(so or {})) as f:
            json.dump(data, f)
    @classmethod
    def load(cls, path: str, so: dict):
        ns = cls()
        with fsspec.open(_normalize_url(path), "r", **(so or {})) as f:
            data = json.load(f)
        ns.stats = defaultdict(dict, data.get("by_tf", {}))
        return ns
    def get(self, tf: str, col: str):
        return self.stats.get(tf, {}).get(col, None)
    def set(self, tf: str, col: str, mean: float, std: float):
        self.stats[tf][col] = {"mean": float(mean), "std": float(std)}

# ------------------------------
# TF horizons
# ------------------------------
TF_HORIZONS = {"5m":  360, "15m":1080, "30m":2160, "1h":4320, "2h":7200, "4h":14400}

# ------------------------------
# Helpers S3 / IO
# ------------------------------
def _month_tag_from_path(p: str) -> Optional[str]:
    m = _MONTH_RE.search(p)
    return f"{m.group(1)}-{m.group(2)}" if m else None

def _available_span_from_glob(path_glob: str, so: dict, debug: bool=False, tag: str="") -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    paths = _expand_paths(path_glob, so or {})
    if not paths:
        if debug: _log(True, f"[span] {tag} no match for {path_glob}")
        return None, None
    months = []
    for p in paths:
        mt = _month_tag_from_path(p)
        if mt:
            months.append(mt)
    if not months:
        if debug: _log(True, f"[span] {tag} no YYYY-MM in {len(paths)} paths")
        return None, None
    months_sorted = sorted(set(months))
    y0, m0 = map(int, months_sorted[0].split("-"))
    y1, m1 = map(int, months_sorted[-1].split("-"))
    start = pd.Timestamp(year=y0, month=m0, day=1, tz="UTC")
    last_day = monthrange(y1, m1)[1]
    end = pd.Timestamp(year=y1, month=m1, day=last_day, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if debug:
        _log(True, f"[span] {tag} months={months_sorted[:3]}{'...' if len(months_sorted)>3 else ''} -> {start} .. {end}")
    return start, end

def _arrow_time_filter(dset: ds.Dataset, ts_field: str, t0: pd.Timestamp, t1: pd.Timestamp):
    f = dset.schema.field(ts_field)
    ftype = f.type
    if not patypes.is_timestamp(ftype):
        raise ValueError(f"Field {ts_field} is not a timestamp: {ftype}")
    t0 = pd.Timestamp(t0).tz_convert("UTC") if t0.tzinfo else pd.Timestamp(t0, tz="UTC")
    t1 = pd.Timestamp(t1).tz_convert("UTC") if t1.tzinfo else pd.Timestamp(t1, tz="UTC")
    if ftype.tz is None:
        t0v = t0.tz_convert("UTC").to_pydatetime().replace(tzinfo=None)
        t1v = t1.tz_convert("UTC").to_pydatetime().replace(tzinfo=None)
    else:
        t0v = t0.tz_convert(ftype.tz).to_pydatetime()
        t1v = t1.tz_convert(ftype.tz).to_pydatetime()
    s0 = pa.scalar(t0v, type=ftype)
    s1 = pa.scalar(t1v, type=ftype)
    return (ds.field(ts_field) >= s0) & (ds.field(ts_field) <= s1)

def _log(debug: bool, msg: str) -> None:
    if debug:
        print(str(msg), flush=True)

def _expand_paths(path_or_glob: str, so: dict) -> list[str]:
    uri = _normalize_url(path_or_glob)
    _ensure_s3(uri)
    fs, _, matches = fsspec.get_fs_token_paths(uri, storage_options=so or {})
    proto = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
    out = []
    for m in matches:
        if not re.match(r"^[a-z0-9]+://", str(m)):
            m = f"{proto}://{m}"
        out.append(_normalize_url(m))
    return sorted(out)

def _ensure_s3(url: str):
    if not re.match(r"^[a-z0-9]+://", url.strip()):
        raise ValueError(f"Chemin non-fsspec: {url}. On n’accepte que s3:// ...")

def _normalize_url(url: str) -> str:
    return re.sub(r"^([A-Za-z0-9]+)://", lambda m: m.group(1).lower() + "://", url.strip())

def _so(region: str | None, anon: bool) -> dict:
    so = {}
    if region: so["client_kwargs"] = {"region_name": region}
    if anon:   so["anon"] = True
    return so

def _pa_filesystem_from_so(so: dict) -> pa_fs.FileSystem:
    try:
        import s3fs
        s3 = s3fs.S3FileSystem(**(so or {}))
        return pa_fs.PyFileSystem(pa_fs.FSSpecHandler(s3))
    except Exception:
        return pa_fs.S3FileSystem()

def _dataset_from_glob_s3(path_glob: str, so: dict, debug: bool=False, tag: str="") -> ds.Dataset:
    uri = _normalize_url(path_glob); _ensure_s3(uri)
    pafs = _pa_filesystem_from_so(so or {})
    paths = _expand_paths(uri, so)
    if not paths:
        _log(debug, f"[dataset] {tag} glob empty uri={uri}")
        raise FileNotFoundError(f"No files match: {uri}")
    _log(debug, f"[dataset] {tag} match_count={len(paths)} first={paths[0]}")
    d = ds.dataset(paths, format="parquet", filesystem=pafs)
    _log(debug, f"[dataset] {tag} schema={list(d.schema.names)}")
    return d

def _read_book_window_ds(book_glob: str, t0: pd.Timestamp, t1: pd.Timestamp,
                         so: dict, columns: list[str]) -> pd.DataFrame:
    d = _dataset_from_glob_s3(book_glob, so)
    f = _arrow_time_filter(d, "timestamp", t0, t1)
    tbl = d.to_table(columns=columns, filter=f)
    if tbl.num_rows == 0:
        idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=[c for c in columns if c != "timestamp"], index=idx)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.set_index("timestamp").sort_index()

def _read_trades_window_ds(trades_glob: str, t0: pd.Timestamp, t1: pd.Timestamp,
                           so: dict, columns: list[str] | None = None) -> pd.DataFrame:
    default_cols = [
        "timestamp", "price", "qty", "is_aggr_buy",
        "origin_time", "received_time",
        "is_buyer_maker", "side", "last_price", "quantity",
    ]
    want_cols = columns or default_cols

    d = _dataset_from_glob_s3(trades_glob, so)
    names = set(d.schema.names)

    ts_field = None
    for cand in ("origin_time", "timestamp"):
        if cand in names:
            ts_field = cand
            break
    if ts_field is None:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))

    cols_exist = [c for c in want_cols if c in names]
    if ts_field not in cols_exist:
        cols_exist = [ts_field] + cols_exist

    f = _arrow_time_filter(d, ts_field, t0, t1)
    tbl = d.to_table(columns=cols_exist, filter=f)
    if tbl.num_rows == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))

    df = tbl.to_pandas(types_mapper=pd.ArrowDtype).copy()

    if "price" not in df.columns and "last_price" in df.columns:
        df = df.rename(columns={"last_price": "price"})
    if "qty" not in df.columns and "quantity" in df.columns:
        df = df.rename(columns={"quantity": "qty"})

    if "is_aggr_buy" not in df.columns:
        if "is_buyer_maker" in df.columns:
            df["is_aggr_buy"] = ~df["is_buyer_maker"].astype(bool)
        elif "side" in df.columns:
            df["is_aggr_buy"] = df["side"].astype(str).str.lower().eq("buy")
        else:
            df["is_aggr_buy"] = np.nan

    for c in ("origin_time", "timestamp"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["qty"]   = pd.to_numeric(df.get("qty"),   errors="coerce")
    df = df.dropna(subset=[ts_field, "price", "qty"]).sort_values(ts_field)

    drop_flag = (ts_field == "timestamp")
    df = df.set_index(ts_field, drop=drop_flag)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "timestamp"

    return df.sort_index()

def _read_parquet_exact(path: str, ts_col: Optional[str], so: Optional[dict]) -> pd.DataFrame:
    p = _normalize_url(path); _ensure_s3(p)
    df = pd.read_parquet(p, engine="pyarrow", storage_options=so)
    if ts_col is None: return df
    s = df[ts_col]
    s = s if getattr(s.dtype, "tz", None) is not None else pd.to_datetime(s, utc=True)
    df = df.copy(); df[ts_col] = s
    return df.set_index(ts_col).sort_index()

def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC") if getattr(ts, "tzinfo", None) else pd.Timestamp(ts, tz="UTC")

# ------------------------------
# Candles indicators
# ------------------------------
def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _bb_width(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = df["close"].rolling(n, min_periods=n).mean()
    std = df["close"].rolling(n, min_periods=n).std()
    upper = mid + k*std; lower = mid - k*std
    return (upper - lower) / mid

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    up   = h.diff(); down = -l.diff()
    plus_dm  = np.where((up>down) & (up>0), up, 0.0)
    minus_dm = np.where((down>up) & (down>0), down, 0.0)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0,np.nan)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()

# ------------------------------
# Book-derived micro features
# ------------------------------
def _obi_row(row: pd.Series, K: int) -> float:
    sb = sum(float(row.get(f"bid_{i}_size", 0.0) or 0.0) for i in range(K))
    sa = sum(float(row.get(f"ask_{i}_size", 0.0) or 0.0) for i in range(K))
    tot = sb + sa
    return float((sb - sa) / tot) if tot > 0 else np.nan

def _slope(prices: List[float], sizes: List[float]) -> float:
    p, q = np.asarray(prices, float), np.asarray(sizes, float)
    x = np.cumsum(q)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(p).all(): return np.nan
    xm, pm = x.mean(), p.mean()
    den = ((x-xm)**2).sum()
    if den <= 0: return np.nan
    return float(((x-xm)*(p-pm)).sum() / den)

def _wall_opp_share(row: pd.Series, K: int, side: str) -> float:
    opp = "ask" if side=="buy" else "bid"
    sizes = [float(row.get(f"{opp}_{i}_size", 0.0) or 0.0) for i in range(K)]
    tot = float(np.sum(sizes))
    return float(np.max(sizes)/tot) if tot > 0 else np.nan

def _ask_wall_depth(row: pd.Series, K: int) -> float:
    """
    Profondeur totale côté ask sur les K premiers niveaux.
    Sert à mesurer la dynamique du 'mur' côté ask (ask_wall_decay_3s).
    """
    sizes = [float(row.get(f"ask_{i}_size", 0.0) or 0.0) for i in range(K)]
    return float(np.sum(sizes))

def _cum_depth_within_bps_mid(row: pd.Series, side: str, bps: float, mid_t: float) -> float:
    if not np.isfinite(mid_t) or mid_t <= 0: 
        return np.nan
    tol = bps / 1e4 * mid_t
    depth = 0.0
    if side == "buy":
        for i in range(BOOK_LEVELS):
            ap = float(row.get(f"ask_{i}_price", np.nan)); aq = float(row.get(f"ask_{i}_size", 0.0) or 0.0)
            if np.isfinite(ap) and abs(ap - mid_t) <= tol: depth += aq
    else:
        for i in range(BOOK_LEVELS):
            bp = float(row.get(f"bid_{i}_price", np.nan)); bq = float(row.get(f"bid_{i}_size", 0.0) or 0.0)
            if np.isfinite(bp) and abs(bp - mid_t) <= tol: depth += bq
    return float(depth)

# ------------------------------
# S3 sink (parts/)
# ------------------------------
class PartsSink:
    def __init__(self, out_year_dir: str, so: dict, cols_order: list[str] | None = None):
        self.base = _normalize_url(out_year_dir).rstrip("/")
        _ensure_s3(self.base)
        self.parts = f"{self.base}/parts"
        self.so = so or {}
        self.k = 0
        self.cols_order = cols_order

    def write(self, rows: list[dict]):
        if not rows: return
        df = pd.DataFrame(rows)

        if "Y" in df.columns:
            s = pd.to_numeric(df["Y"], errors="coerce")
            df["Y"] = pd.array(s, dtype="Int8")

        part = f"{self.parts}/part-{self.k:05d}.parquet"
        df.to_parquet(part, engine="pyarrow", index=False, compression="zstd", storage_options=self.so)
        self.k += 1

# ------------------------------
# Good-months helpers
# ------------------------------
def _months_touched(t0: pd.Timestamp, t1: pd.Timestamp) -> Set[str]:
    t0 = _ensure_utc(t0); t1 = _ensure_utc(t1)
    if t1 < t0: t0, t1 = t1, t0
    cur = pd.Timestamp(year=t0.year, month=t0.month, day=1, tz="UTC")
    end = pd.Timestamp(year=t1.year, month=t1.month, day=1, tz="UTC")
    out = set()
    while cur <= end:
        out.add(f"{cur.year:04d}-{cur.month:02d}")
        # next month
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year+1, month=1, day=1, tz="UTC")
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month+1, day=1, tz="UTC")
    return out

# ------------------------------
# Core: process one symbol/year (fenêtré, memory-safe)
# -----------------------------
def _read_trades_adaptive_before_t(trades_glob: str,
                                   t: pd.Timestamp,
                                   so: dict,
                                   candidates: list[int] = [15, 30, 60, 120],
                                   debug: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Lit les trades sur une fenêtre [t - W, t], W choisi de façon adaptative
    (15 -> 30 -> 60 -> 120 s) jusqu'à trouver des trades ou épuiser la liste.
    Retourne (df_1s, df_raw, used_win_sec).

    - df_1s : agrégé à 1s avec colonnes buy_qty, sell_qty, tot_qty
    - df_raw: trades bruts indexés par timestamp (possiblement vide)
    """
    for W in candidates:
        t0 = t - pd.Timedelta(seconds=W)
        try:
            tr = _read_trades_window_ds(trades_glob, t0, t, so)
        except Exception as e:
            if debug: _log(True, f"[trades_adapt] exception W={W}: {e}")
            tr = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

        if not tr.empty:
            tr = tr.copy()
            tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
            tr["qty"]   = pd.to_numeric(tr["qty"],   errors="coerce")
            tr["notional"] = tr["price"] * tr["qty"]

            g = tr.groupby(pd.Grouper(freq="1s"))
            buy  = g.apply(lambda x: float(x.loc[x["is_aggr_buy"] == True,  "qty"].sum()))
            sell = g.apply(lambda x: float(x.loc[x["is_aggr_buy"] == False, "qty"].sum()))
            out = pd.DataFrame({"buy_qty": buy, "sell_qty": sell}).fillna(0.0)
            out["tot_qty"] = out["buy_qty"] + out["sell_qty"]

            return out, tr, W

    idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
    empty_1s = pd.DataFrame(index=idx, columns=["buy_qty","sell_qty","tot_qty"]).astype(float).fillna(0.0)
    empty_raw = pd.DataFrame(index=idx)
    return empty_1s, empty_raw, (candidates[-1] if candidates else 15)

def _process_symbol_year(
    symbol: str,
    year: int,
    signals_path: str,
    candles_path: str,
    book_glob: str,
    trades_glob: str,
    so: Optional[dict],
    out_parquet_path: str, 
    debug: bool=False,
    default_tf: str="1h",
    default_fees_bps: float=6.0,
    split_bounds: tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None),
    allowed_months: Optional[Set[str]] = None,
    allowed_tfs: Optional[List[str]] = None
) -> int:

    so = so or {}
    sig_path = _normalize_url(signals_path); _ensure_s3(sig_path)
    _log(debug, f"[start] symbol={symbol} year={year}")
    sig = pd.read_csv(sig_path, storage_options=so)
    _log(debug, f"[signals] raw_rows={len(sig)} path={sig_path}")
    if sig.empty: 
        _log(debug, "[signals] vide -> return 0")
        return 0
    
    req_min = {"t","symbol","year","side","entry"}
    missing_min = req_min - set(sig.columns)
    if missing_min:
        _log(True, f"[signals] colonnes manquantes (obligatoires): {sorted(missing_min)}")
        raise ValueError(f"Signals missing columns: {sorted(missing_min)}")

    if "fees_bps" not in sig.columns:
        alt = None
        for c in ("fees", "fee_bps", "fees_bps_trade"):
            if c in sig.columns:
                alt = c; break
        if alt is not None:
            sig["fees_bps"] = pd.to_numeric(sig[alt], errors="coerce").fillna(default_fees_bps)
            _log(True, f"[signals] 'fees_bps' manquait, mappé depuis '{alt}', nan -> {default_fees_bps}")
        else:
            sig["fees_bps"] = float(default_fees_bps)
            _log(True, f"[signals] 'fees_bps' manquait, rempli par défaut = {default_fees_bps}")

    if "tf" not in sig.columns:
        sig["tf"] = str(default_tf)
        _log(True, f"[signals] 'tf' manquait, rempli par défaut = {default_tf}")
    else:
        sig["tf"] = sig["tf"].astype(str).str.strip().str.lower()
        nan_tf = sig["tf"].eq("").sum()
        if nan_tf:
            sig.loc[sig["tf"].eq(""), "tf"] = str(default_tf)
            _log(True, f"[signals] tf vides: {nan_tf} -> {default_tf}")

    # Filtrer sur TF autorisés (passés au main via closure)
    # On injectera 'allowed_tfs' via un param additionnel dans _process_symbol_year
    if allowed_tfs is not None:
        sig = sig[sig["tf"].isin(allowed_tfs)]
        if sig.empty:
            _log(True, "[signals] aucun signal après filtre --tfs -> skip")
            return 0
    
    sig["entry"] = pd.to_numeric(sig["entry"], errors="coerce")
    sig["fees_bps"] = pd.to_numeric(sig["fees_bps"], errors="coerce").fillna(default_fees_bps)
    sig = sig.dropna(subset=["entry"])
    
    before = len(sig)
    sig = sig.loc[(sig["symbol"]==symbol) & (sig["year"]==int(year))].copy()
    _log(debug, f"[signals] after_filter rows={len(sig)} (was {before}) unique_tf={sig['tf'].unique().tolist() if len(sig)>0 else []}")
    if sig.empty: 
        _log(debug, "[signals] aucun signal pour ce couple symbol/year")
        return 0
    sig["t"] = pd.to_datetime(sig["t"], utc=True)
    _log(debug, f"[signals] t-range: {sig['t'].min()} .. {sig['t'].max()}")

    # Split window constraint
    s0, s1 = split_bounds
    if s0 is not None: sig = sig[sig["t"] >= s0]
    if s1 is not None: sig = sig[sig["t"] <= s1]
    if sig.empty:
        _log(True, "[split] aucun signal dans la fenêtre de split -> skip")
        return 0

    # Good-months (require t month ∈ allowed; and label window stays within allowed months)
    allowed_months = set(allowed_months or [])
    if allowed_months:
        sig = sig[sig["t"].dt.strftime("%Y-%m").isin(allowed_months)]
        if sig.empty:
            _log(True, "[good_months] aucun signal dans les mois autorisés -> skip")
            return 0

    sig["side_num"] = sig["side"].astype(str).str.lower().map({"buy":1,"sell":-1}).astype(int)

    _log(debug, f"[ohlc] load {candles_path}")
    ohlc = _read_parquet_exact(candles_path, "timestamp", so)[["open","high","low","close"]].copy()
    if ohlc.empty:
        _log(True, f"[ohlc] vide pour {symbol} {year} -> skip")
        return 0

    # NEW: estimer la durée d'une bougie en secondes (pour 30m)
    if len(ohlc.index) >= 2:
        dt_secs = (ohlc.index[1] - ohlc.index[0]).total_seconds()
        bar_secs = max(float(dt_secs), 1.0)
    else:
        bar_secs = 60.0
    bars_30m = max(1, int(round(1800.0 / bar_secs)))

    ohlc["atr"] = _atr(ohlc, 14)
    ohlc["bb_width"] = _bb_width(ohlc, 20, 2.0)
    ohlc["adx"] = _adx(ohlc, 14)

    out_dir = _normalize_url(os.path.dirname(out_parquet_path)); _ensure_s3(out_dir)
    sink = PartsSink(out_dir, so=so)

    DEFAULT_H = TF_HORIZONS.get(default_tf, 4320)

    book_start, book_end     = _available_span_from_glob(book_glob,   so, debug=debug, tag="book")
    trades_start, trades_end = _available_span_from_glob(trades_glob, so, debug=debug, tag="trades")

    eff_start = book_start
    eff_end   = book_end
    if trades_start is not None:
        eff_start = max(eff_start, trades_start) if eff_start is not None else trades_start
    if trades_end is not None:
        eff_end   = min(eff_end,   trades_end)   if eff_end   is not None else trades_end

    if eff_start is None or eff_end is None or eff_start >= eff_end:
        _log(True, f"[span] Aucun intervalle exploitable pour {symbol} {year} — skip")
        return 0

    H_MAX = max(TF_HORIZONS.values())

    before_prune = len(sig)
    # par (ne garde que la marge de 15s en entrée) :
    sig = sig[sig["t"] >= (eff_start + pd.Timedelta(seconds=15))].copy()
    # la borne de fin est vérifiée plus tard par signal: if t_out1 > eff_end: continue

    _log(debug, f"[signals] pruned_to_available_span rows={len(sig)} (was {before_prune}) eff={eff_start}..{eff_end}")
    if sig.empty:
        _log(True, f"[WARN] aucun signal restant après pruning sur span dispo")
        return 0
    
    BOOK_TOB = ["timestamp","bid_0_price","ask_0_price","bid_0_size","ask_0_size"]
    DEPTH_COLS = []
    for i in range(BOOK_LEVELS):
        DEPTH_COLS += [f"bid_{i}_price", f"ask_{i}_price", f"bid_{i}_size", f"ask_{i}_size"]
    BOOK_COLS_WINDOW = list(dict.fromkeys(BOOK_TOB + DEPTH_COLS))

    batch, BATCH_MAX, total = [], 2000, 0
    skip_stats = {"book_empty":0, "tob_missing":0, "trades_err":0, "label_book_empty":0}

    len_sig = len(sig)
    for i, (_, r) in enumerate(sig.sort_values("t").iterrows(), start=1):
        # init systématique pour éviter UnboundLocalError et faciliter le finally
        book_win = tob = tr1 = tr10 = tr15 = tr_lat = None
        try:
            if debug and (i % 100 == 0):
                _log(True, f"[progress] {symbol} {year} processed={i}/{len_sig}")

            t = r["t"]; sgn = int(r["side_num"]); side = "buy" if sgn==1 else "sell"
            entry = float(r["entry"]); fees_bps = float(r["fees_bps"]); tf = str(r["tf"])
            H = int(TF_HORIZONS.get(tf, DEFAULT_H))

            # 🔹 helper side-aware local basé sur le signe du trade
            def SA(x: float) -> float:
                return float(sgn * x) if np.isfinite(x) else np.nan

            t_feat0 = _ensure_utc(t - pd.Timedelta(seconds=20))  # élargi à 20s
            t_feat1 = _ensure_utc(t + pd.Timedelta(seconds=15))
            t_out0  = _ensure_utc(t + pd.Timedelta(seconds=15))
            t_out1  = _ensure_utc(t + pd.Timedelta(seconds=H))

            if allowed_months:
                touched = _months_touched(t_out0, t_out1)
                if not touched.issubset(allowed_months):
                    continue

            # ===== Book window (features) =====
            try:
                book_win = _read_book_window_ds(book_glob, t_feat0, t_feat1, so, BOOK_COLS_WINDOW)
            except Exception as e:
                _log(debug, f"[book] exception: {e}")
                skip_stats["book_empty"] += 1
                continue
            if book_win.empty:
                _log(debug, "[book] vide -> skip signal")
                skip_stats["book_empty"] += 1
                continue

            tob = book_win[["bid_0_price","ask_0_price","bid_0_size","ask_0_size"]].resample("1s").last().ffill()
            bid0 = tob["bid_0_price"].astype(float); ask0 = tob["ask_0_price"].astype(float)
            mid  = (bid0 + ask0)/2.0
            spread_bps = ((ask0 - bid0)/mid)*1e4
            b0sz = tob["bid_0_size"].astype(float); a0sz = tob["ask_0_size"].astype(float)

            idx_ev = book_win.index
            j = idx_ev.searchsorted(t, side="right") - 1
            if j < 0:
                skip_stats["tob_missing"] += 1
                continue
            row_t = book_win.iloc[j]
            # --- cap d’ancienneté (mitigation "stale quotes") ---
            ts_last = idx_ev[j]
            MAX_STALE = pd.Timedelta(seconds=10)
            if (t - ts_last) > MAX_STALE:
                # Dernière quote trop ancienne par rapport à t → on considère manquant
                skip_stats["tob_missing"] += 1
                continue
            # (optionnel) vérifier que les séries resamplées ont bien un point récent
            if bid0.loc[:t].last_valid_index() is None or (t - bid0.loc[:t].last_valid_index()) > MAX_STALE:
                skip_stats["tob_missing"] += 1
                continue

            # ===== Trades (fenêtre adaptative, causale) =====
            cands = _trade_windows_for(symbol, tf)
            tr1, tr_raw, used_win_sec = _read_trades_adaptive_before_t(
                trades_glob, t, so, candidates=cands, debug=debug
            )

            w10 = slice(t - pd.Timedelta(seconds=10), t)
            w15 = slice(t - pd.Timedelta(seconds=15), t)

            tr10 = tr1.loc[w10] if not tr1.empty else pd.DataFrame()
            tr15 = tr1.loc[w15] if not tr1.empty else pd.DataFrame()

            # NEW: fenêtre 3s pour features short/long
            w3 = slice(t - pd.Timedelta(seconds=3), t)
            tr3 = tr1.loc[w3] if not tr1.empty else pd.DataFrame()

            def _sum_buy_sell(dfw: pd.DataFrame) -> tuple[float, float]:
                if dfw is None or dfw.empty:
                    return 0.0, 0.0
                b = float(dfw.get("buy_qty",  pd.Series(dtype=float)).sum())
                s = float(dfw.get("sell_qty", pd.Series(dtype=float)).sum())
                return b, s

            b3,  s3  = _sum_buy_sell(tr3)
            b10, s10 = _sum_buy_sell(tr10)

            # Imbalance d'agression (sell / (sell+buy))
            def _imbl_raw(s: float, b: float) -> float:
                den = s + b
                return float(s/den) if den > 0 else np.nan

            imbl_aggr_3s_raw  = _imbl_raw(s3,  b3)
            imbl_aggr_10s_raw = _imbl_raw(s10, b10)

            imbl_aggr_3s  = imbl_aggr_3s_raw  if np.isfinite(imbl_aggr_3s_raw)  else 0.5
            imbl_aggr_10s = imbl_aggr_10s_raw if np.isfinite(imbl_aggr_10s_raw) else 0.5

            def _imbl_side(imbl: float) -> float:
                if not np.isfinite(imbl):
                    return 0.0
                # centré, côté-aware : (imbl - 0.5) * side_num
                return float((1 if side == "buy" else -1) * (imbl - 0.5))

            imbl_aggr_3s_side  = _imbl_side(imbl_aggr_3s)
            imbl_aggr_10s_side = _imbl_side(imbl_aggr_10s)


            aggr_ratio_10s = _aggr_ratio(tr10)
            aggr_ratio_15s = _aggr_ratio(tr15)
            b15 = float(tr15.get("buy_qty",  pd.Series(dtype=float)).sum())
            s15 = float(tr15.get("sell_qty", pd.Series(dtype=float)).sum())
            den15 = b15 + s15
            net_delta_15s = ((b15 - s15)/den15) if den15 > 0 else np.nan

            # ---- Fallbacks neutres si pas de trades (évite NaN dans le dataset) ----
            # aggr_ratio_*       ∈ [0..1]  → 0.5
            # aggr_ratio_*_side  ∈ [-1..1] → 0.0 (centré)
            # net_delta_15s/_side∈ [-1..1] → 0.0
            has10 = int((not tr10.empty) and (tr10.get("tot_qty", pd.Series(dtype=float)).sum() > 0))
            has15 = int((not tr15.empty) and (tr15.get("tot_qty", pd.Series(dtype=float)).sum() > 0))
            # versions "remplies"
            agr10 = float(aggr_ratio_10s) if np.isfinite(aggr_ratio_10s) else 0.5
            agr15 = float(aggr_ratio_15s) if np.isfinite(aggr_ratio_15s) else 0.5
            nd15  = float(net_delta_15s)  if np.isfinite(net_delta_15s)  else 0.0
            # sides centrés
            def _side_map(x: float) -> float:
                return float((1 if side=="buy" else -1) * x) if np.isfinite(x) else 0.0
            agr10_side = _side_map(2*agr10 - 1.0)
            agr15_side = _side_map(2*agr15 - 1.0)
            nd15_side  = _side_map(nd15)

            # NEW: ajout / exécution au bid pour 3s/10s
            b0_3  = b0sz.loc[w3]  if b0sz.loc[w3].size  > 0 else pd.Series(dtype=float)
            b0_10 = b0sz.loc[w10] if b0sz.loc[w10].size > 0 else pd.Series(dtype=float)

            added_bid_3s  = float(np.maximum(b0_3.diff(),  0.0).sum()) if b0_3.size  > 0 else np.nan
            added_bid_10s = float(np.maximum(b0_10.diff(), 0.0).sum())  if b0_10.size > 0 else np.nan

            executed_sell_3s  = float(s3)
            executed_sell_10s = float(s10)

            # LONG: absorption (sells / added bid) — versions "raw" avant clip
            bid_absorb_ratio_3s_raw  = np.nan
            best_bid_refill_rate_raw = np.nan
            bid_absorb_ratio_10s_raw = np.nan

            if np.isfinite(added_bid_3s) and added_bid_3s > 0 and executed_sell_3s > 0:
                bid_absorb_ratio_3s_raw  = float(executed_sell_3s / added_bid_3s)
                best_bid_refill_rate_raw = float(added_bid_3s / executed_sell_3s)  # SHORT

            if np.isfinite(added_bid_10s) and added_bid_10s > 0 and executed_sell_10s > 0:
                bid_absorb_ratio_10s_raw = float(executed_sell_10s / added_bid_10s)

            # Flags "événement" : 1 si ratio indéfini (NaN) avant clip
            bid_absorb_ratio_3s_isnan  = int(not np.isfinite(bid_absorb_ratio_3s_raw))
            bid_absorb_ratio_10s_isnan = int(not np.isfinite(bid_absorb_ratio_10s_raw))
            best_bid_refill_rate_isnan = int(not np.isfinite(best_bid_refill_rate_raw))

            # Clip dans [0, upper] (non négatif)
            bid_absorb_ratio_3s  = _clip_ratio_nonneg(bid_absorb_ratio_3s_raw,  upper=100.0)
            bid_absorb_ratio_10s = _clip_ratio_nonneg(bid_absorb_ratio_10s_raw, upper=100.0)
            best_bid_refill_rate = _clip_ratio_nonneg(best_bid_refill_rate_raw, upper=100.0)

            # LONG: persistance refill (ticks) sur 10s
            bid_refill_persistence_ticks = np.nan
            b0_for_refill = b0sz.loc[w10]
            if b0_for_refill.size > 1:
                diffs = b0_for_refill.diff().fillna(0.0).to_numpy(dtype=float)
                refill_cycles = 0
                expecting_refill = False
                for d in diffs:
                    if not np.isfinite(d):
                        continue
                    if not expecting_refill and d < 0:
                        # "hit"
                        expecting_refill = True
                    elif expecting_refill:
                        if d > 0:
                            # refill après hit
                            refill_cycles += 1
                            expecting_refill = False
                        elif d < 0:
                            # plusieurs hits de suite, on reste en attente de refill
                            continue
                bid_refill_persistence_ticks = float(refill_cycles)

            # ===== Features =====
            _sp = spread_bps.loc[:t]
            spread_bps_entry = float(_sp.iloc[-1]) if len(_sp) else np.nan

            # NEW: delta_spread_3s = spread_now - min(spread sur 3s)
            w3 = slice(t - pd.Timedelta(seconds=3), t)
            sp3 = spread_bps.loc[w3]
            if sp3.size > 0 and np.isfinite(spread_bps_entry):
                spread_min_3s = float(np.nanmin(sp3.values))
                delta_spread_3s = float(spread_bps_entry - spread_min_3s) if np.isfinite(spread_min_3s) else np.nan
            else:
                delta_spread_3s = np.nan
            
             # NEW: spread moyen sur la dernière seconde [t-1s, t]
            spread_bps_mean_1s = np.nan
            w1 = slice(t - pd.Timedelta(seconds=1), t)
            sp1 = spread_bps.loc[w1]
            if sp1.size > 0:
                spread_bps_mean_1s = float(np.nanmean(sp1.values))

            p90_10s = float(np.nanpercentile(spread_bps.loc[w10].values, 90)) if spread_bps.loc[w10].size>0 else np.nan
            spread_widen_flag_10s = int(np.isfinite(spread_bps_entry) and np.isfinite(p90_10s) and (spread_bps_entry >= p90_10s))

            abs_d_bsz = np.abs(b0sz.loc[w10].diff()).sum() if b0sz.loc[w10].size>1 else np.nan
            abs_d_asz = np.abs(a0sz.loc[w10].diff()).sum() if a0sz.loc[w10].size>1 else np.nan
            quote_churn_10s = float(abs_d_bsz + abs_d_asz) if np.isfinite(abs_d_bsz) and np.isfinite(abs_d_asz) else np.nan

            def _last_at_or_before(s: pd.Series, tt: pd.Timestamp) -> float:
                # renvoie la dernière valeur ≤ tt, NaN si introuvable
                if s.loc[:tt].empty:
                    return np.nan
                return float(s.loc[:tt].ffill().iloc[-1])

            bsz_t = _last_at_or_before(b0sz, t)
            asz_t = _last_at_or_before(a0sz, t)

            w_mp = (asz_t/(bsz_t+asz_t)) if (np.isfinite(bsz_t) and np.isfinite(asz_t) and (bsz_t+asz_t)>0) else 0.5
            bid_t = _last_at_or_before(bid0, t)
            ask_t = _last_at_or_before(ask0, t)

            mid_t = (bid_t + ask_t)/2.0 if np.isfinite(bid_t) and np.isfinite(ask_t) else np.nan
            microprice = w_mp*ask_t + (1.0-w_mp)*bid_t if np.isfinite(w_mp) and np.isfinite(bid_t) and np.isfinite(ask_t) else np.nan
            microprice_bias = ((microprice - mid_t)/mid_t) if (np.isfinite(microprice) and np.isfinite(mid_t) and mid_t>0) else np.nan

            # NEW: microprice_bias_ma_1s / 3s + microprice_shift_1s
            mp_ma_1s = np.nan
            mp_ma_3s = np.nan
            microprice_shift_1s = np.nan

            mp_win = pd.DataFrame({
                "bid": bid0.loc[w3],
                "ask": ask0.loc[w3],
                "bsz": b0sz.loc[w3],
                "asz": a0sz.loc[w3],
            }).dropna()
            if not mp_win.empty:
                mid_s = (mp_win["bid"] + mp_win["ask"]) / 2.0
                den = mp_win["bsz"] + mp_win["asz"]
                w_mp_s = np.where(
                    (den > 0) & np.isfinite(den),
                    mp_win["asz"] / den,
                    0.5
                )
                # série microprice sur la fenêtre 3s
                micro_s = pd.Series(
                    w_mp_s * mp_win["ask"].to_numpy()
                    + (1.0 - w_mp_s) * mp_win["bid"].to_numpy(),
                    index=mp_win.index,
                )
                mp_bias_s = (micro_s - mid_s) / mid_s

                if np.isfinite(mp_bias_s).any():
                    mp_bias_s_clean = mp_bias_s[np.isfinite(mp_bias_s)]
                    mp_ma_1s = float(mp_bias_s_clean.iloc[-1])
                    mp_ma_3s = float(mp_bias_s_clean.mean())

                # NEW: microprice_shift_1s = variation (en bps) sur 1 seconde
                if micro_s.size > 0 and np.isfinite(mid_t) and mid_t > 0:
                    last_micro = float(micro_s.iloc[-1])
                    t_minus_1s = t - pd.Timedelta(seconds=1)
                    micro_prev = micro_s.loc[:t_minus_1s]
                    if not micro_prev.empty:
                        micro_prev_last = float(micro_prev.iloc[-1])
                        if np.isfinite(micro_prev_last):
                            microprice_shift_1s = float(
                                1e4 * (last_micro - micro_prev_last) / mid_t
                            )

            microprice_bias_ma_1s = mp_ma_1s
            microprice_bias_ma_3s = mp_ma_3s

            mp_win = pd.DataFrame({
                "bid": bid0.loc[w3],
                "ask": ask0.loc[w3],
                "bsz": b0sz.loc[w3],
                "asz": a0sz.loc[w3],
            }).dropna()
            if not mp_win.empty:
                mid_s = (mp_win["bid"] + mp_win["ask"]) / 2.0
                den = mp_win["bsz"] + mp_win["asz"]
                w_mp_s = np.where((den > 0) & np.isfinite(den), mp_win["asz"] / den, 0.5)
                micro_s = w_mp_s * mp_win["ask"] + (1.0 - w_mp_s) * mp_win["bid"]
                mp_bias_s = np.where(
                    (mid_s > 0) & np.isfinite(mid_s) & np.isfinite(micro_s),
                    (micro_s - mid_s) / mid_s,
                    np.nan,
                )
                # dernier point = ~MA 1s, moyenne = MA 3s
                if np.isfinite(mp_bias_s).any():
                    mp_bias_s_clean = mp_bias_s[np.isfinite(mp_bias_s)]
                    mp_ma_1s = float(mp_bias_s_clean[-1])
                    mp_ma_3s = float(np.nanmean(mp_bias_s_clean))

            microprice_bias_ma_1s = mp_ma_1s
            microprice_bias_ma_3s = mp_ma_3s

            def _slope_pair(K: int):
                bp = [row_t.get(f"bid_{i}_price", np.nan) for i in range(K)]
                bq = [row_t.get(f"bid_{i}_size",  np.nan) for i in range(K)]
                ap = [row_t.get(f"ask_{i}_price", np.nan) for i in range(K)]
                aq = [row_t.get(f"ask_{i}_size",  np.nan) for i in range(K)]
                return _slope(bp,bq), _slope(ap,aq)

            obi_5  = _obi_row(row_t,5)
            obi_15 = _obi_row(row_t,15)
            slope_bid_5,  slope_ask_5  = _slope_pair(5)
            slope_bid_15, slope_ask_15 = _slope_pair(15)
            wall_opp_share_5  = _wall_opp_share(row_t,5,side)
            wall_opp_share_15 = _wall_opp_share(row_t,15,side)
            cum_depth_within_5bps_opp  = _cum_depth_within_bps_mid(row_t, side, 5.0,  mid_t)
            cum_depth_within_10bps_opp = _cum_depth_within_bps_mid(row_t, side, 10.0, mid_t)

            ret_1s = mid.pct_change().loc[w10]
            ret_stdev_1s_10s_bps = (np.nanstd(ret_1s.values)*1e4) if ret_1s.size>1 else np.nan

            t3 = t - pd.Timedelta(seconds=3)
            _m3 = mid.loc[:t3]; _m0 = mid.loc[:t]
            mid_t3 = float(_m3.iloc[-1]) if len(_m3) else np.nan
            mid_t0 = float(_m0.iloc[-1]) if len(_m0) else np.nan
            mid_jump_bps_3s = (10000.0 * (mid_t0 - mid_t3)/mid_t3) if (np.isfinite(mid_t0) and np.isfinite(mid_t3) and mid_t3>0) else np.nan

            # NEW: mid_vs_VWAP_3s (bps) basé sur trades bruts tr_raw
            mid_minus_vwap_3s = np.nan
            if tr_raw is not None and not tr_raw.empty:
                tr3_raw = tr_raw.loc[w3] if isinstance(tr_raw.index, pd.DatetimeIndex) else pd.DataFrame()
                if not tr3_raw.empty and "price" in tr3_raw.columns and "qty" in tr3_raw.columns:
                    v_q = pd.to_numeric(tr3_raw["qty"], errors="coerce")
                    v_p = pd.to_numeric(tr3_raw["price"], errors="coerce")
                    mask = np.isfinite(v_q) & np.isfinite(v_p) & (v_q > 0)
                    v_q = v_q[mask]
                    v_p = v_p[mask]
                    den_v = float(v_q.sum())
                    if den_v > 0 and np.isfinite(mid_t) and mid_t > 0:
                        vwap_3s = float((v_p * v_q).sum() / den_v)
                        if np.isfinite(vwap_3s) and vwap_3s > 0:
                            mid_minus_vwap_3s = float(1e4 * (mid_t - vwap_3s) / vwap_3s)
            
            # NEW: ask_wall_decay_3s (dynamique du mur côté ask sur 3s)
            ask_wall_decay_3s = np.nan
            sub_bw_3s = book_win.loc[w3] if not book_win.empty else pd.DataFrame()
            if not sub_bw_3s.empty:
                ask_wall_series = sub_bw_3s.apply(lambda rr: _ask_wall_depth(rr, BOOK_LEVELS), axis=1)
                if ask_wall_series.size > 0:
                    ask_wall_max = float(np.nanmax(ask_wall_series.values))
                    ask_wall_now = float(_ask_wall_depth(row_t, BOOK_LEVELS))
                    if np.isfinite(ask_wall_max) and ask_wall_max > 0 and np.isfinite(ask_wall_now):
                        ask_wall_decay_3s = float((ask_wall_now - ask_wall_max) / (ask_wall_max + 1e-9))
            
            # NEW: wall_persistence_score_5s (mur persistant côté opposé sur 5s)
            wall_persistence_score_5s = np.nan
            w5 = slice(t - pd.Timedelta(seconds=5), t)
            sub_bw_5s = book_win.loc[w5] if not book_win.empty else pd.DataFrame()
            if not sub_bw_5s.empty:
                # on réutilise _wall_opp_share(row, 5, side) pour chaque event
                shares_5s = sub_bw_5s.apply(
                    lambda rr: _wall_opp_share(rr, 5, side),
                    axis=1
                )
                if shares_5s.size > 0:
                    vals = shares_5s.to_numpy(dtype=float)
                    mask = np.isfinite(vals)
                    if mask.any():
                        wall_persistence_score_5s = float(
                            np.mean(vals[mask] >= 0.5)
                        )

            # Contexte bougies
            octx_full = ohlc.loc[:t].copy()
            octx = octx_full.tail(200)

            bb_width = float(octx["bb_width"].iloc[-1]) if len(octx) > 0 and np.isfinite(octx["bb_width"].iloc[-1]) else np.nan
            adx = float(octx["adx"].iloc[-1]) if len(octx) > 0 and np.isfinite(octx["adx"].iloc[-1]) else np.nan

            atr_bps_sig = float(r["atr_bps"]) if "atr_bps" in r and np.isfinite(r["atr_bps"]) else np.nan
            atr_bps = atr_bps_sig if np.isfinite(atr_bps_sig) else (
                float((octx["atr"].iloc[-1] / mid_t) * 1e4)
                if len(octx) > 0 and np.isfinite(mid_t) and mid_t > 0 else np.nan
            )

            atr_window = octx["atr"].dropna()
            atr_percentile = float((atr_window.rank(pct=True).iloc[-1])) if len(atr_window) > 10 else np.nan

            # NEW: ATR percentile sur ~30m d'historique
            atr_pct_rank_30m = np.nan
            if not octx_full.empty:
                atr_win_30m = octx_full["atr"].dropna().tail(bars_30m)
                if len(atr_win_30m) >= 2:
                    atr_pct_rank_30m = float(atr_win_30m.rank(pct=True).iloc[-1])

            # NEW: percentile de bb_width sur l'historique local (regime compression/expansion)
            bb_width_pctl = np.nan
            bb_hist = octx_full["bb_width"].dropna()
            if len(bb_hist) >= 20:
                bb_width_pctl = float(bb_hist.rank(pct=True).iloc[-1])

            # OBI persistence (10s)
            times_10s = pd.date_range(t - pd.Timedelta(seconds=10), t, freq="1s")
            def _obi_at(second_ts: pd.Timestamp) -> float:
                jj = idx_ev.searchsorted(second_ts, side="right") - 1
                if jj < 0: return np.nan
                return _obi_row(book_win.iloc[jj], 5)
            obi_hist = np.array([_obi_at(tt) for tt in times_10s])
            side_num = 1 if side=="buy" else -1
            imbalance_persistence = float(np.nanmean((np.sign(obi_hist) == side_num).astype(float))) if np.isfinite(obi_hist).any() else np.nan

            # NEW: OBI delta & ratio sur 1 seconde (K=5)
            obi_5_delta_1s = np.nan
            obi_5_ratio_1s = np.nan
            obi_5_delta_1s_side = np.nan
            obi_5_ratio_1s_side = np.nan

            obi_now = obi_5  # OBI(5) au temps t, déjà calculé plus haut
            obi_prev = _obi_at(t - pd.Timedelta(seconds=1))

            if np.isfinite(obi_now) and np.isfinite(obi_prev):
                # variation brute
                obi_5_delta_1s = float(obi_now - obi_prev)
                # ratio normalisé dans [-1,1]
                denom = abs(obi_now) + abs(obi_prev) + 1e-9
                obi_5_ratio_1s = float(obi_now / denom)
                # versions side-aware
                obi_5_delta_1s_side = SA(obi_5_delta_1s)
                obi_5_ratio_1s_side = SA(obi_5_ratio_1s)

            added_bid = float(np.maximum(b0sz.loc[w10].diff(), 0.0).sum()) if b0sz.loc[w10].size>0 else np.nan
            added_ask = float(np.maximum(a0sz.loc[w10].diff(), 0.0).sum()) if a0sz.loc[w10].size>0 else np.nan
            added_top = added_bid + added_ask if np.isfinite(added_bid) and np.isfinite(added_ask) else np.nan
            executed_top = float(tr10["tot_qty"].sum()) if ("tot_qty" in tr10) else np.nan
            den_exec = added_top if (np.isfinite(added_top) and added_top > 1e-6) else np.nan
            executed_vs_added_ratio = float(executed_top/den_exec) if (np.isfinite(executed_top) and np.isfinite(den_exec)) else np.nan
            if np.isfinite(executed_vs_added_ratio):
                executed_vs_added_ratio = float(np.clip(executed_vs_added_ratio, 0.0, 50.0))

            # ===== Label — [t+15s, t+H]
            if t_out1 > eff_end:
                continue
            try:
                book_out = _read_book_window_ds(book_glob, t_out0, t_out1, so, ["timestamp","bid_0_price","ask_0_price"])
            except Exception as e:
                _log(debug, f"[label/book] exception: {e}")
                book_out = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

            if book_out.empty:
                _log(debug, "[label/book] outcome vide")
                skip_stats["label_book_empty"] += 1
                pnl_net_max_bps = np.nan; pnl_net_min_bps = np.nan; THRESH_BPS = np.nan; Y = np.nan
            else:
                tob_out = book_out[["bid_0_price","ask_0_price"]].resample("1s").last().ffill()
                mid_out = (tob_out["bid_0_price"].astype(float) + tob_out["ask_0_price"].astype(float))/2.0
                pnl_path_bps = 1e4 * sgn * ((mid_out - entry)/entry)
                cost_bps = float(fees_bps + 0.5*spread_bps_entry) if np.isfinite(spread_bps_entry) else np.nan
                if np.isfinite(cost_bps):
                    pnl_net_path = pnl_path_bps - cost_bps
                    pnl_net_max_bps = float(np.nanmax(pnl_net_path.values)) if pnl_net_path.size>0 else np.nan
                    pnl_net_min_bps = float(np.nanmin(pnl_net_path.values)) if pnl_net_path.size>0 else np.nan
                else:
                    pnl_net_max_bps = np.nan; pnl_net_min_bps = np.nan
                atr_bps_eff = float(r.get("atr_bps", np.nan)) if "atr_bps" in sig.columns and np.isfinite(r.get("atr_bps", np.nan)) else float(atr_bps)
                THRESH_BPS = float(max(1.5*atr_bps_eff, 0.5*spread_bps_entry + fees_bps)) if (np.isfinite(atr_bps_eff) and np.isfinite(spread_bps_entry) and np.isfinite(fees_bps)) else np.nan

                if np.isfinite(pnl_net_max_bps) and np.isfinite(THRESH_BPS) and pnl_net_max_bps > +THRESH_BPS:
                    Y = 1
                elif np.isfinite(pnl_net_min_bps) and np.isfinite(THRESH_BPS) and pnl_net_min_bps < -THRESH_BPS:
                    Y = -1
                else:
                    Y = 0 if np.isfinite(THRESH_BPS) else np.nan

            row = {
                "t": t, "symbol": symbol, "year": year, "tf": tf, "side": side, "entry": entry,
                "spread_bps_entry": spread_bps_entry,
                "spread_bps_mean_1s": spread_bps_mean_1s,  # NEW
                "spread_widen_flag_10s": int(spread_widen_flag_10s),
                "cum_depth_within_5bps_opp": cum_depth_within_5bps_opp,
                "cum_depth_within_10bps_opp": cum_depth_within_10bps_opp,
                "quote_churn_10s": quote_churn_10s,
                "wall_opp_share_5": wall_opp_share_5, "wall_opp_share_15": wall_opp_share_15,
                "obi_5": obi_5, "obi_15": obi_15,
                "obi_5_delta_1s": obi_5_delta_1s,      # NEW
                "obi_5_ratio_1s": obi_5_ratio_1s,      # NEW
                "microprice_bias": microprice_bias,
                "slope_bid_5": slope_bid_5, "slope_ask_5": slope_ask_5,
                "slope_bid_15": slope_bid_15, "slope_ask_15": slope_ask_15,
                "aggr_ratio_10s": agr10, "aggr_ratio_15s": agr15,
                "net_delta_15s": nd15,   
                "ret_stdev_1s_10s_bps": ret_stdev_1s_10s_bps,
                "mid_jump_bps_3s": mid_jump_bps_3s,
                "bb_width": bb_width, "adx": adx, "atr_percentile": atr_percentile,
                "fees_bps": fees_bps, "atr_bps": atr_bps,
                "imbalance_persistence": imbalance_persistence,
                "executed_vs_added_ratio": executed_vs_added_ratio,
                "horizon_sec": H,
                "pnl_net_max_bps": pnl_net_max_bps,
                "pnl_net_min_bps": pnl_net_min_bps,
                "THRESH_BPS": THRESH_BPS,
                "has_trades_win10s": has10,
                "has_trades_win15s": has15,
                "trades_win_sec_used": int(used_win_sec),
                "Y": Y,
                # NEW: features spécialisés SHORT/LONG
                "imbl_aggr_3s": imbl_aggr_3s,
                "imbl_aggr_10s": imbl_aggr_10s,
                "delta_spread_3s": delta_spread_3s,
                "best_bid_refill_rate": best_bid_refill_rate,
                "bid_absorb_ratio_3s": bid_absorb_ratio_3s,
                "bid_absorb_ratio_10s": bid_absorb_ratio_10s,
                "bid_refill_persistence_ticks": bid_refill_persistence_ticks,
                "microprice_bias_ma_1s": microprice_bias_ma_1s,
                "microprice_bias_ma_3s": microprice_bias_ma_3s,
                "microprice_shift_1s": microprice_shift_1s,  # NEW
                "mid_minus_vwap_3s": mid_minus_vwap_3s,
                "atr_pct_rank_30m": atr_pct_rank_30m,
                "bb_width_pctl": bb_width_pctl,
                "ask_wall_decay_3s": ask_wall_decay_3s,
                "wall_persistence_score_5s": wall_persistence_score_5s,  # NEW
                # NEW: flags NaN sur les ratios événementiels
                "best_bid_refill_rate_isnan": best_bid_refill_rate_isnan,
                "bid_absorb_ratio_3s_isnan": bid_absorb_ratio_3s_isnan,
                "bid_absorb_ratio_10s_isnan": bid_absorb_ratio_10s_isnan,
            }

            row.update({
                "obi_5_side": SA(obi_5), "obi_15_side": SA(obi_15),
                "obi_5_delta_1s_side": obi_5_delta_1s_side,   # NEW
                "obi_5_ratio_1s_side": obi_5_ratio_1s_side,   # NEW
                "microprice_bias_side": SA(microprice_bias),
                "slope_bid_5_side": SA(slope_bid_5), "slope_ask_5_side": SA(slope_ask_5),
                "slope_bid_15_side": SA(slope_bid_15), "slope_ask_15_side": SA(slope_ask_15),
                "aggr_ratio_10s_side": agr10_side,
                "aggr_ratio_15s_side": agr15_side,
                "net_delta_15s_side": nd15_side,
                "mid_jump_bps_3s_side": SA(mid_jump_bps_3s) if np.isfinite(mid_jump_bps_3s) else np.nan,
                # NEW side-aware
                "imbl_aggr_3s_side": imbl_aggr_3s_side,
                "imbl_aggr_10s_side": imbl_aggr_10s_side,
                "microprice_bias_ma_1s_side": SA(microprice_bias_ma_1s) if np.isfinite(microprice_bias_ma_1s) else np.nan,
                "microprice_bias_ma_3s_side": SA(microprice_bias_ma_3s) if np.isfinite(microprice_bias_ma_3s) else np.nan,
                "microprice_shift_1s_side": SA(microprice_shift_1s) if np.isfinite(microprice_shift_1s) else np.nan,  # NEW
                "mid_minus_vwap_3s_side": SA(mid_minus_vwap_3s) if np.isfinite(mid_minus_vwap_3s) else np.nan,
                "wall_persistence_score_5s_side": SA(wall_persistence_score_5s - 0.5) if np.isfinite(wall_persistence_score_5s) else np.nan,
            })

            batch.append(row); total += 1
            if len(batch) >= BATCH_MAX:
                sink.write(batch); batch.clear(); gc.collect()

        finally:
            # Libère les bindings explicitement (locals() n'est pas fiable ici)
            try: del book_win
            except Exception: pass
            try: del tob, tr1, tr10, tr15, tr_lat, tr_raw
            except Exception: pass
            gc.collect()
        
    # --- fin de la boucle ---
    if batch:
        sink.write(batch); batch.clear()

    _log(debug, f"[done] {symbol} {year} total_rows={total} skip_stats={skip_stats}")
    _log(True, f"[skip_stats] {symbol} {year} {skip_stats}")
    if total == 0:
        _log(True, f"[WARN] no rows for {symbol} {year} — vérifiez globs BOOK/TRADES, colonnes 'timestamp', et plage temporelle signals vs data")
    return total

def _dataset_from_parts_glob(glob_path: str, so: dict) -> ds.Dataset:
    pafs = _pa_filesystem_from_so(so or {})
    paths = _expand_paths(glob_path, so or {})
    if not paths:
        raise FileNotFoundError(f"No parts found: {glob_path}")
    return ds.dataset(paths, format="parquet", filesystem=pafs)

def _fit_norm_from_train_parts(train_root: str, so: dict) -> NormStats:
    """
    Parcourt tous les parts du split 'train' (tous symboles/années), regroupe par TF,
    et calcule mean/std pour chaque colonne numérique (hors EXCLUDE_FROM_NORM).
    """
    ns = NormStats()

    # On ouvre tout le split train d'un coup via un dataset Arrow
    dset = _dataset_from_parts_glob(f"{_normalize_url(train_root).rstrip('/')}/**/parts/*.parquet", so)
    cols = list(dset.schema.names)

    # Determiner les colonnes normalisables: colonnes numériques présentes + pas dans EXCLUDE
    # On s'aide d'un petit scan de schéma
    schema = dset.schema
    numeric_cols = []
    for f in schema:
        if f.name in EXCLUDE_FROM_NORM:
            continue
        if patypes.is_floating(f.type):     # <- floats only
            numeric_cols.append(f.name)

    # On a besoin de 'tf' pour regrouper par TF
    if "tf" not in cols:
        raise ValueError("Parts train sans colonne 'tf' — requis pour la normalisation par-TF")

    # On lit par batches pour rester mémoire-safe
    # Note: Arrow Scanner permet un batch_size; on itère sur batches
    scanner = dset.scanner(columns=["tf"] + numeric_cols, batch_size=200_000)
    # Accumulateurs Welford: dict[tf][col] -> Welford()
    acc = defaultdict(lambda: defaultdict(_Welford))

    for batch in scanner.to_batches():
        tbl = pa.Table.from_batches([batch])
        pdf = tbl.to_pandas(types_mapper=pd.ArrowDtype)
        if pdf.empty: continue

        pdf["tf"] = pdf["tf"].astype(str)
        for tf_val, g in pdf.groupby("tf"):
            for col in numeric_cols:
                if col not in g.columns: 
                    continue
                arr = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                acc[tf_val][col].update_arr(arr)

        del pdf

    # Finaliser stats
    for tf_val, cols_map in acc.items():
        for col, wf in cols_map.items():
            mean, std = wf.result()
            ns.set(tf_val, col, mean, std if np.isfinite(std) and std>0 else 1.0)

    return ns

def _rewrite_split_with_norm(src_split_root: str, dst_split_root: str, ns: NormStats, so: dict):
    dset = _dataset_from_parts_glob(f"{_normalize_url(src_split_root).rstrip('/')}/**/parts/*.parquet", so)
    schema = dset.schema

    # floats only (paired with change #1)
    numeric_cols = []
    for f in schema:
        if f.name in EXCLUDE_FROM_NORM: 
            continue
        if patypes.is_floating(f.type):
            numeric_cols.append(f.name)

    scanner = dset.scanner(columns=list(schema.names), batch_size=200_000)

    for batch in scanner.to_batches():
        tbl = pa.Table.from_batches([batch])

        # IMPORTANT: no ArrowDtype here
        pdf = tbl.to_pandas()                     # <- do NOT pass types_mapper=pd.ArrowDtype
        if pdf.empty:
            continue

        pdf["tf"] = pdf["tf"].astype(str)

        for tf_val, g_tf in pdf.groupby("tf"):
            stats_tf = ns.stats.get(tf_val, {})
            if not stats_tf:
                continue

            cols_norm = [c for c in numeric_cols if (c in g_tf.columns) and (c in stats_tf)]
            if not cols_norm:
                continue

            # ensure destination columns are float64 BEFORE any assignment
            for c in cols_norm:
                if not pd.api.types.is_float_dtype(pdf[c].dtype):
                    pdf[c] = pd.to_numeric(pdf[c], errors="coerce").astype("float64")

            # now standardize
            for c in cols_norm:
                st = stats_tf[c]
                m = float(st["mean"])
                s = float(st["std"]) if float(st["std"]) > 0 else 1.0
                vals = pd.to_numeric(g_tf[c], errors="coerce").astype("float64")
                pdf.loc[g_tf.index, c] = (vals - m) / s

        for (sym, year), g_sy in pdf.groupby(["symbol", "year"]):
            out_dir = f"{_normalize_url(dst_split_root).rstrip('/')}/{sym}/{int(year)}/parts"
            part_path = f"{out_dir}/part-{int(np.random.randint(0, 1e9)):09d}.parquet"
            g_sy.to_parquet(part_path, engine="pyarrow", index=False, compression="zstd", storage_options=so)

# ------------------------------
# Split bounds
# ------------------------------
# --- replace _split_bounds() with this version (adds pad hours) ---
def _split_bounds(name: str, pad_hours: int = 4) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    name = (name or "all").lower()
    if name == "train":
        s0, s1 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-04-30 23:59:59.999999", tz="UTC")
    elif name == "val":
        s0, s1 = pd.Timestamp("2025-05-01", tz="UTC"), pd.Timestamp("2025-07-31 23:59:59.999999", tz="UTC")
    elif name == "test":
        s0, s1 = pd.Timestamp("2025-08-01", tz="UTC"), pd.Timestamp("2025-10-31 23:59:59.999999", tz="UTC")
    else:
        return (None, None)

    # pad end by +4h so labels [t+15s, t+H] fit when H=4h
    return (s0, s1 + pd.Timedelta(hours=pad_hours))

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser("Stage2 dataset builder (micro + label Y)")

    # === Entrées obligatoires
    p.add_argument("--signals-root", required=True,
                   help="s3://.../report (contient <SYMBOL>/<YEAR>_signals.csv)")
    p.add_argument("--bougie-root", required=True,
               help="s3://.../bougie/<SYMBOL>/<SYMBOL>-1m-<YEAR>.parquet (annuel, template avec <SYMBOL> et <YEAR>)")
    p.add_argument("--book-root", required=True,
                   help="s3://.../level_1/<SYMBOL>/<YEAR>-*.parquet (mensuel)")
    p.add_argument("--trades-root", required=True,
                   help="s3://.../trade/<SYMBOL>/<YEAR>-*.parquet (mensuel)")
    p.add_argument("--out-root", required=True,
                   help="s3://.../stage2 (dossier de sortie)")

    # === Liste des symboles / années à traiter
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
               help="Liste des symboles (défaut: 16 majeurs)")
    
    p.add_argument("--years", nargs="+", type=int, default=None,
                   help="Liste des années ex: 2023 2024 2025 (défaut: déduit de good_months.csv pour les symboles fournis)")

    # === Fichier de contrôle qualité
    p.add_argument("--good-months", required=True,
                   help="Chemin S3 vers good_months.csv (filtrage des mois valides)")

    # === Split temporel
    p.add_argument("--split", choices=["train", "val", "test", "all", "all_splits"], default="all_splits",
    help=("Fenêtre temporelle : "
          "train=2024-01..2025-03, "
          "val=2025-04..2025-06, "
          "test=2025-07..2025-09, "
          "all=sans contrainte, "
          "all_splits=tous les trois d'un coup")
)

    # === Options AWS / S3
    p.add_argument("--s3-region", default="ap-northeast-1",
                   help="Région AWS ex: ap-northeast-1")
    p.add_argument("--s3-anon", action="store_true",
                   help="Accès anonyme (lecture publique seulement)")

    # === Divers
    p.add_argument("--debug", action="store_true", help="Verbose debug logging")
    p.add_argument("--default-fees-bps", type=float, default=6.0,
                   help="Frais en bps si 'fees_bps' manque dans le CSV (ex: 6.0)")
    
    # === Timeframes & Normalisation
    p.add_argument("--tfs", nargs="+", default=DEFAULT_TFS, help="TF autorisés (défaut: 5m 15m 30m 1h 2h 4h)")
    
    p.add_argument("--norm-mode", choices=["none","fit","apply","fit_apply"], default="fit_apply",
                   help=("Normalisation par-TF globale: "
                         "fit=calcule stats sur parts train, "
                         "apply=applique stats existantes à tous les splits, "
                         "fit_apply=fit sur train puis applique à tous les splits, "
                         "none=aucune normalisation"))
    p.add_argument("--norm-path", default=None,
                   help="Chemin S3 pour sauvegarder/charger les stats (ex: s3://.../stage2/_norm/stats.json)")
    p.add_argument("--norm-out-root", default=None,
                   help="Racine de sortie pour datasets normalisés (défaut: <out-root> avec suffixe _norm)")

    return p.parse_args()

# ------------------------------
# main
# ------------------------------
def main():
    args = parse_args()
    print(f"[boot] stage2 starting | debug={args.debug} | symbols={args.symbols} | years={args.years}", flush=True)
    so = _so(args.s3_region, args.s3_anon)

    # Charger good_months.csv (S3)
    gm = pd.read_csv(_normalize_url(args.good_months), storage_options=so)
    gm = gm.rename(columns=str.lower)
    if "symbol" not in gm.columns or "ym" not in gm.columns:
        raise ValueError("good_months.csv doit contenir les colonnes 'symbol' et 'ym'")
    gm["symbol"] = gm["symbol"].astype(str)
    gm["ym"] = gm["ym"].astype(str)

    # Déduire les années disponibles à partir de good_months.csv si --years est omis
    if args.years is None:
        gm_sel = gm[gm["symbol"].isin(args.symbols)]
        years_in_gm = (
            pd.to_datetime(gm_sel["ym"] + "-01", utc=True, errors="coerce")
              .dt.year.dropna().astype(int).unique().tolist()
        )
        args.years = sorted(years_in_gm)
        print(f"[auto-years] années déduites de good_months.csv: {args.years}", flush=True)

    # TFS autorisés: forcer minuscules/trim (évite 4H/4h)
    args.tfs = [str(tf).strip().lower() for tf in (args.tfs or [])]

    # Map global des mois "bons" PAR SYMBOLE (toutes années)
    allowed_by_symbol: Dict[str, Set[str]] = {
        sym: set(gm.loc[gm["symbol"] == sym, "ym"].astype(str))
        for sym in args.symbols
    }

    # Gestion des splits
    splits = [args.split] if args.split != "all_splits" else ["train","val","test"]

    # utilitaire mois+1
    def _next_ym(ym: str) -> str:
        y, m = map(int, ym.split("-"))
        y2 = y + (1 if m == 12 else 0)
        m2 = 1 if m == 12 else m + 1
        return f"{y2:04d}-{m2:02d}"

    for split_name in splits:
        s0, s1 = _split_bounds(split_name)
        print(f"[split] {split_name} | {s0} .. {s1}", flush=True)

        # Mois du split (avec un tampon: le mois suivant la borne sup)
        if (s0 is not None) and (s1 is not None):
            split_months = _months_touched(s0, s1)
            if split_months:
                split_months |= {_next_ym(max(split_months))}
        else:
            # 'all' : pas de contrainte => tous les mois des symbols
            split_months = set().union(*allowed_by_symbol.values()) if allowed_by_symbol else set()

        for sym in args.symbols:
            for year in args.years:
                sig_path    = f"{args.signals_root.rstrip('/')}/{sym}/{year}_signals.csv"
                bougie_path = f"{args.bougie_root.rstrip('/').replace('<SYMBOL>', sym).replace('<YEAR>', str(year))}"
                book_glob   = f"{args.book_root.rstrip('/').replace('<SYMBOL>', sym).replace('<YEAR>', str(year))}"
                trades_glob = f"{args.trades_root.rstrip('/').replace('<SYMBOL>', sym).replace('<YEAR>', str(year))}"
                out_path    = f"{args.out_root.rstrip('/')}/{split_name}/{sym}/{year}/data.parquet"

                # Intersecter mois autorisés (par symbole) avec la fenêtre du split
                allowed_for_pair = allowed_by_symbol.get(sym, set()) & split_months

                if not allowed_for_pair:
                    print(f"⏭️ [{split_name}] skip {sym} {year}: aucun mois autorisé dans la fenêtre du split", flush=True)
                    continue

                try:
                    n = _process_symbol_year(
                        sym, year, sig_path, bougie_path, book_glob, trades_glob, so, out_path,
                        debug=args.debug, default_fees_bps=args.default_fees_bps,
                        split_bounds=(s0, s1),
                        allowed_months=allowed_for_pair,
                        allowed_tfs=args.tfs
                    )
                    if n > 0:
                        print(f"✅ [{split_name}] wrote parts under {os.path.dirname(out_path)}/parts (rows={n})")
                    else:
                        print(f"⚠️ [{split_name}] no rows for {sym} {year}")
                except FileNotFoundError as e:
                    print(f"⚠️ [{split_name}] skip {sym} {year}: {e}")

    # =========================
    # Normalisation post-build
    # =========================
    out_root_norm = args.norm_out_root or (_normalize_url(args.out_root).rstrip("/") + "_norm")
    norm_stats_path = args.norm_path or (out_root_norm.rstrip("/") + "/_norm/stats.json")

    if args.norm_mode in ("fit","fit_apply"):
        # Fit sur TRAIN SEULEMENT
        train_root = f"{_normalize_url(args.out_root).rstrip('/')}/train"
        print(f"[norm] FIT from train: {train_root}", flush=True)
        try:
            ns = _fit_norm_from_train_parts(train_root, so)
            ns.save(norm_stats_path, so)
            print(f"[norm] stats saved to {norm_stats_path}", flush=True)
        except FileNotFoundError:
            print("[norm] ⚠️ aucun parts train trouvés pour fit; skip", flush=True)

    if args.norm_mode in ("apply","fit_apply"):
        # Charger stats avec protection
        try:
            ns = NormStats.load(norm_stats_path, so)
        except FileNotFoundError:
            print(f"[norm] ⚠️ stats introuvables: {norm_stats_path} — skip APPLY", flush=True)
        else:
            print(f"[norm] APPLY using {norm_stats_path}", flush=True)
            for split_name in (["train","val","test"] if args.split=="all_splits" else [args.split]):
                src_split_root = f"{_normalize_url(args.out_root).rstrip('/')}/{split_name}"
                dst_split_root = f"{_normalize_url(out_root_norm).rstrip('/')}/{split_name}"
                try:
                    _rewrite_split_with_norm(src_split_root, dst_split_root, ns, so)
                    print(f"✅ [norm] wrote normalized parts under {dst_split_root}", flush=True)
                except FileNotFoundError:
                    print(f"⚠️ [norm] no parts found to normalize in {src_split_root}", flush=True)

if __name__ == "__main__":
    main()