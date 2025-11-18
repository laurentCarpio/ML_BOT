# build_stage1_sign.py
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import fsspec
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

# Allow local run `python next_bot/app/run_replay.py`
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml_bot.core.utils.trade_logger import logger_backtest
from ml_bot.feeds.replay_feed import ReplayConfig, ReplayFeed
from ml_bot.strategy.prepare_data import calculate_indicators
from ml_bot.strategy.v1_breakout import V1BreakoutStrategy

# =============================================================
# Data classes & trace helpers
# =============================================================
@dataclass
class DecisionTrace:
    ts: str
    tf: str
    stage: str          # "retail" | "gate" | "exec"
    decision: str       # "CANDIDATE" | "REJECT" | "GO"
    reason: str         # "ABCD_NO_ENTRY" | "COOLDOWN" | ...
    ctx: Dict[str, Any]

class TraceCollector:
    def __init__(self, sample_cap: int = 400):
        self.sample_cap = sample_cap
        self.samples = deque(maxlen=sample_cap)
        self.counters = Counter()
        self.candidates = 0
        self.hits = 0

    def add(self, item: DecisionTrace):
        key = f"{item.stage}:{item.decision}:{item.reason}"
        self.counters[key] += 1
        if item.decision == "CANDIDATE":
            self.candidates += 1
        if item.decision == "GO":
            self.hits += 1
        if item.decision in ("REJECT", "GO") and len(self.samples) < self.sample_cap:
            self.samples.append(item)

    def summary_lines(self) -> list[str]:
        lines = [f"candidates={self.candidates} | hits={self.hits}"]
        for k, v in self.counters.most_common(15):
            lines.append(f"{k} -> {v}")
        return lines

# =============================================================
# Global stop flag & signals
# =============================================================
_SHUTDOWN = False
def _on_term(signum, frame):
    global _SHUTDOWN
    try:
        name = signal.Signals(signum).name
    except Exception:
        name = str(signum)
    logger_backtest.warning(f"[RUN] Received {name} → graceful shutdown requested")
    _SHUTDOWN = True

signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT,  _on_term)

# =============================================================
# S3 & IO helpers
# =============================================================
def _s3_fs(region: str | None):
    so = {"client_kwargs": {"region_name": region}} if region else {}
    return fsspec.filesystem("s3", **so)

def _require_year_month_coverage(
    root_dir: str,
    symbol: str,
    year: int,
    aws_region: str | None,
    warn_ratio: float = 0.5,
    start_iso: str | None = None,
    end_iso: str | None = None,
    require_full_year: bool = True,
):
    """
    Vérifie que le fichier annuel 1m existe et couvre bien les mois requis.

    Attend des fichiers de la forme :
      root_dir/<SYMBOL>/<SYMBOL>-1m-<YEAR>.parquet

    Exemple:
      root_dir = s3://tradebot-config-tokyo/data/bougie
      → s3://tradebot-config-tokyo/data/bougie/BTCUSDT/BTCUSDT-1m-2023.parquet
    """
    fs = _s3_fs(aws_region)

    # 🔴 Ancien schéma (à supprimer) :
    # annual_path = root_dir.rstrip("/") + f"/{symbol}/{year}.parquet"

    # ✅ Nouveau schéma aligné avec audit_runner :
    annual_path = f"{root_dir.rstrip('/')}/{symbol}/{symbol}-1m-{year}.parquet"

    if not fs.exists(annual_path):
        raise FileNotFoundError(f"[{symbol}] Parquet annuel manquant : {annual_path}")

    so = {"client_kwargs": {"region_name": aws_region}} if aws_region else {}
    df = pd.read_parquet(annual_path, storage_options=so, columns=["timestamp"])
    if df.empty:
        raise RuntimeError(f"[{symbol}] {annual_path} vide")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df_year = df[df["timestamp"].dt.year == year]
    if df_year.empty:
        raise RuntimeError(f"[{symbol}] {annual_path} ne contient aucune donnée pour {year}")

    if require_full_year or not (start_iso and end_iso):
        months_needed = list(range(1, 13))
        df_check = df_year
    else:
        start_dt = pd.Timestamp(start_iso, tz="UTC")
        end_dt   = pd.Timestamp(end_iso,   tz="UTC")
        start_y = max(start_dt, pd.Timestamp(year=year, month=1, day=1, tz="UTC"))
        end_y   = min(end_dt,   pd.Timestamp(year=year+1, month=1, day=1, tz="UTC"))
        if start_y >= end_y:
            return
        df_check = df[(df["timestamp"] >= start_y) & (df["timestamp"] < end_y)]
        months_needed = sorted(df_check["timestamp"].dt.month.unique().tolist())

    counts = df_check.groupby(df_check["timestamp"].dt.month).size().to_dict()
    missing_months = [m for m in months_needed if m not in counts]
    if missing_months:
        raise RuntimeError(
            f"[{symbol}] {annual_path} incomplet : mois manquants = {missing_months} (counts/mois={counts})"
        )

    vals = [counts[m] for m in months_needed if m in counts]
    med = np.median(vals) if vals else 0
    if med > 0:
        weak = {m: counts[m] for m in months_needed if counts.get(m, 0) < warn_ratio * med}
        if weak:
            logger_backtest.warning(
                f"[{symbol}] ⚠️ Couverture faible pour {len(weak)} mois (warn_ratio={warn_ratio:.0%}) → {weak} | médiane={int(med)}"
            )
    logger_backtest.info(
        f"[{symbol}] ✅ {annual_path} couvre les mois requis {months_needed} (points/mois={counts})"
    )

def _emit_abcd_signal_compact(sym: str, t, tf: str, side: str, reason: str,
                              A: dict, B: dict, C: dict, D: dict):
    def b(k): return 1 if bool(B.get(k, False)) else 0
    msg = (
        f"[ABCD_SIGNAL] ts={t} tf={tf} side={side} reason={reason} "
        f"regime={C.get('regime','UNKNOWN')} bb_width={float(C.get('bb_width', float('nan'))):.6f} "
        f"rsi={float(D.get('rsi', float('nan'))):.2f} "
        f"A_up={int(A.get('up',0))} A_down={int(A.get('down',0))} "
        f"A_5m={A.get('5m','NA')} A_15m={A.get('15m','NA')} A_30m={A.get('30m','NA')} A_1h={A.get('1h','NA')} "
        f"B1_pullback_up={b('B1_pullback_up')} B1_pullback_down={b('B1_pullback_down')} "
        f"B2_weak_break_up={b('B2_weak_break_up')} B2_weak_break_down={b('B2_weak_break_down')}"
    )
    logger_backtest.info(msg)

def _period_seconds(freq: str) -> int:
    return {"1m":60, "3m":180, "5m":300, "15m":900, "30m":1800, "1h":3600, "2h":7200}.get(freq, 60)

def _parse_map(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":")
        out[k.strip()] = int(v.strip())
    return out

def _normalize_list(lst):
    if isinstance(lst, list) and len(lst) == 1:
        parts = [p for p in lst[0].replace(",", " ").split() if p]
        return parts if parts else lst
    return lst

def _year_bounds(y: int) -> tuple[str, str]:
    start = pd.Timestamp(year=y, month=1, day=1, tz="UTC")
    end   = pd.Timestamp(year=y+1, month=1, day=1, tz="UTC")
    return start.isoformat().replace("+00:00", ""), end.isoformat().replace("+00:00", "")

def _s3_out_base_from_root(root_parquet_dir: str) -> str:
    parts = root_parquet_dir.rstrip("/").split("/")
    if parts[-1].lower() in ("parquet", "bougie"):
        parts[-1] = "stage1"
    else:
        parts.append("stage1")
    return "/".join(parts)

# === Remplace complètement _write_signals_year_to_s3 par ceci ===
# Colonnes attendues (ordre figé)
SIGNALS_COLS = [
    "t","symbol","year","tf","side","side_num","entry",
    "symbol_id","freq_id","regime","bb_width","rsi_1h","fees_bps","atr_bps"
]

def _write_signals_year_to_s3(
    symbol: str,
    year: int,
    rows: list[dict],
    root_parquet_dir: str,
    aws_region: str | None,
    overwrite: bool = False,
):
    if not rows:
        return

    # normalise en DataFrame + colonnes figées
    df = pd.DataFrame(rows)
    for c in SIGNALS_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SIGNALS_COLS]

    s3_out_base = _s3_out_base_from_root(root_parquet_dir)
    out_path = f"{s3_out_base}/{symbol}/{year}_signals.csv"

    is_s3 = out_path.startswith("s3://")
    so = {"client_kwargs": {"region_name": aws_region}} if (aws_region and is_s3) else {}
    fs = fsspec.filesystem("s3", **so) if is_s3 else None
    exists = (fs.exists(out_path) if is_s3 else os.path.exists(out_path))

    # overwrite => toujours "wb" + header
    mode = "wb" if overwrite or (not exists) else "ab"
    header = True if overwrite or (not exists) else False

    if not is_s3:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with fsspec.open(out_path, mode, **so) as f:
        df.to_csv(f, index=False, header=header)

    logger_backtest.info(
        f"[{symbol}] 📤 wrote {len(df)} rows to {out_path} "
        f"({'OVERWRITE' if overwrite else ('CREATE' if not exists else 'APPEND')})"
    )

# =============================================================
# File logging → local & S3
# =============================================================
def _mk_log_name(symbol: str, start: str, end: str) -> str:
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"replay_{symbol}_{start}_{end}_{stamp}.log"

def _attach_file_handler(logger, path: str):
    import logging
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(logger.level)
    logger.addHandler(fh)
    return fh

def _upload_file_to_s3(local_path: str, s3_uri: str, aws_region: Optional[str]):
    so = {"client_kwargs": {"region_name": aws_region}} if aws_region else {}
    with fsspec.open(s3_uri, "wb", **so) as f_s3, open(local_path, "rb") as f_local:
        f_s3.write(f_local.read())

# =============================================================
# Argument parsing
# =============================================================
def _period_seconds(freq: str) -> int:
    return {
        "1m":60, "3m":180, "5m":300,
        "15m":900, "30m":1800,
        "1h":3600, "2h":7200,
        "4h":14400,        
    }.get(freq, 60)

def parse_args() -> argparse.Namespace:
    # 👇 Ajoute conflict_handler="resolve" par sûreté (si un flag est redéfini plus bas, il sera remplacé)
    ap = argparse.ArgumentParser(conflict_handler="resolve")

    # Entrées S3 + périmètre
    ap.add_argument("--root-parquet-dir", default="s3://tradebot-config-tokyo/data/bougie")
    ap.add_argument("--aws-region", default="ap-northeast-1")

    # Périmètre backtest
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","LTCUSDT","DOGEUSDT","SUIUSDT"])
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end",   default="2024-01-01")


    # --- parse_args() ---
    ap.add_argument("--freqs", nargs="+", default=["15m","30m","1h","2h","4h","5m"])  # ← 4h ajouté (avant 5m)
    ap.add_argument("--window", default="5m:80,15m:60,30m:40,1h:30,2h:20,4h:12")  # ← 4h:12
    ap.add_argument("--cooldown", default="5m:12,15m:8,30m:6,1h:4,2h:3,4h:2")       # ← 4h:2

    # Fréquences & fenêtres
    ap.add_argument("--fees-bps", type=float, default=6.0)

    # Exécution
    ap.add_argument("--heartbeat", type=int, default=5000)
    ap.add_argument("--experience", default="v4")

    # Tracing / debug
    ap.add_argument("--trace-verbosity", choices=["quiet","normal","loud","abcd_debug"], default="quiet")
    ap.add_argument("--trace-abcd-every-n", type=int, default=10)
    ap.add_argument("--step-seconds", type=int, default=None)

    # Param A/B/C/D
    ap.add_argument("--bb-n", type=int, default=20)
    ap.add_argument("--bb-k", type=float, default=2.0)
    ap.add_argument("--bb-quiet", type=float, default=0.012)
    ap.add_argument("--bb-volatile", type=float, default=0.05)
    ap.add_argument("--rsi-n", type=int, default=14)
    ap.add_argument("--rsi-overbought", type=float, default=70.0)
    ap.add_argument("--rsi-oversold", type=float, default=30.0)

    # Logging fichier + S3
    ap.add_argument("--log-to-s3", action="store_true", help="Écrit un log local et l'upload sur S3 à la fin.")
    ap.add_argument("--log-s3-prefix", default="s3://tradebot-config-tokyo/data/logs")
    ap.add_argument("--log-local-path", default=None)
    ap.add_argument("--log-upload-every-heartbeat", action="store_true",
                    help="Upload le fichier de log à chaque heartbeat en plus du flush final.")
    return ap.parse_args()

def normalize_args(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    args.symbols = _normalize_list(args.symbols)
    args.freqs   = _normalize_list(args.freqs)
    window_map   = _parse_map(args.window)
    cooldown_map = _parse_map(args.cooldown)
    warmup_map   = {f: 220 for f in args.freqs}  # SMA200 safe-warmup
    if args.step_seconds is None:
        args.step_seconds = min(_period_seconds(f) for f in args.freqs)
    if args.year is not None:
        start_iso, end_iso = _year_bounds(int(args.year))
        logger_backtest.info(f"[RUN] --year={args.year} → override range: {start_iso} → {end_iso} (exclusive)")
        args.start, args.end = start_iso[:10], end_iso[:10]
    for f in args.freqs:
        warmup_map.setdefault(f, 0)
        window_map.setdefault(f, 1)
        cooldown_map.setdefault(f, 0)
    return window_map, cooldown_map, warmup_map, {}

def trace_profile(args: argparse.Namespace) -> tuple[bool, int]:
    if args.trace_verbosity == "quiet":
        return False, args.trace_abcd_every_n
    if args.trace_verbosity == "normal":
        return False, args.trace_abcd_every_n
    if args.trace_verbosity == "loud":
        return True, 1
    return True, args.trace_abcd_every_n  # abcd_debug

def recap(args, warmup_map, window_map, cooldown_map):
    logger_backtest.info("════════════════ REPLAY CONFIG ════════════════")
    logger_backtest.info(f"source     : S3 @ {args.root_parquet_dir} (region={args.aws_region})")
    logger_backtest.info(f"symbols    : {args.symbols}")
    logger_backtest.info(f"year       : {args.year if args.year is not None else '(range explicite)'}")
    logger_backtest.info(f"range      : {args.start} → {args.end} (exclusive)")
    logger_backtest.info(f"freqs      : {args.freqs} | step={args.step_seconds}s")
    logger_backtest.info(f"warmup     : {warmup_map}")
    logger_backtest.info(f"window     : {window_map}")
    logger_backtest.info(f"cooldown   : {cooldown_map}")
    logger_backtest.info(f"experience : {args.experience}")
    logger_backtest.info(f"BB width   : n={args.bb_n} k={args.bb_k} quiet={args.bb_quiet} volatile={args.bb_volatile}")
    logger_backtest.info(f"RSI(1h)    : n={args.rsi_n} overbought={args.rsi_overbought} oversold={args.rsi_oversold}")
    logger_backtest.info("══════════════════════════════════════════════")

# =============================================================
# Strategy prep
# =============================================================
class _PDShim:
    def __init__(self, symbol: str):
        self._MyStrategy__symbol = symbol

def build_feed(args, sym: str, warmup_map, window_map) -> ReplayFeed:
    cfg = ReplayConfig(
        root_parquet_dir=args.root_parquet_dir,
        symbol=sym,
        start=args.start,
        end=args.end,
        freqs=args.freqs,
        warmup_bars_by_freq=warmup_map,
        window_bars_by_freq=window_map,
        aws_region=args.aws_region,
        step_seconds=args.step_seconds,
    )
    return ReplayFeed(cfg)

def enrich_all_freqs(feed: ReplayFeed, args, sym: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    pd_shim = _PDShim(sym)
    enriched_by_freq: Dict[str, pd.DataFrame] = {}
    enriched_idx_by_freq: Dict[str, pd.DataFrame] = {}

    for f in args.freqs:
        full_df = feed._by_freq[f]
        if not full_df.empty and (not full_df["timestamp"].is_monotonic_increasing):
            full_df = full_df.sort_values("timestamp", kind="stable", ignore_index=True)
        try:
            df_base = calculate_indicators(pd_shim, full_df.copy(), f)
        except Exception as e:
            raise RuntimeError(f"[{sym}] calculate_indicators failed for {f}: {e}")
        df_en = df_base
        if not df_en.empty and (not df_en["timestamp"].is_monotonic_increasing):
            df_en = df_en.sort_values("timestamp", kind="stable", ignore_index=True)
        enriched_by_freq[f] = df_en
        enriched_idx_by_freq[f] = df_en.set_index("timestamp", drop=True)

    return enriched_by_freq, enriched_idx_by_freq

def inject_ctx(strategies: dict[str, V1BreakoutStrategy],
               enriched_idx_by_freq: dict[str, pd.DataFrame]):
    ctx = {"1h": enriched_idx_by_freq.get("1h")}
    for f, strat in strategies.items():
        strat.set_context(ctx)

# =============================================================
# Windows projection
# =============================================================
def project_windows(windows: dict[str, pd.DataFrame],
                    enriched_idx_by_freq: dict[str, pd.DataFrame],
                    need_by_freq: dict[str, int]) -> dict[str, pd.DataFrame]:
    out = {}
    for f, dfw in list(windows.items()):
        if dfw is None or dfw.empty:
            out[f] = pd.DataFrame()
            continue
        if not dfw["timestamp"].is_monotonic_increasing:
            dfw = dfw.sort_values("timestamp", kind="stable", ignore_index=True)
        last_ts_w = dfw["timestamp"].iat[-1]
        try:
            df_upto = enriched_idx_by_freq[f].loc[:last_ts_w]
        except KeyError:
            out[f] = pd.DataFrame()
            continue
        need = need_by_freq.get(f, 1)
        out[f] = df_upto.iloc[-need:].reset_index() if len(df_upto) >= need else pd.DataFrame()
    return out

# =============================================================
# ABCD execution (C/D non bloquants)
# =============================================================
def compute_abcd(strat: V1BreakoutStrategy,
                 windows_proj: dict[str, pd.DataFrame],
                 args) -> tuple[dict, dict, dict, dict]:
    A = strat.block_A_mtf_trend(windows_proj, freqs=tuple(args.freqs))
    B = strat.block_B_pattern(windows_proj.get("5m"))
    C = strat.block_C_regime(windows_proj.get("30m"),
                             n=args.bb_n, k=args.bb_k,
                             quiet_thr=args.bb_quiet, volatile_thr=args.bb_volatile)
    D = strat.block_D_rsi_circuit(windows_proj.get("1h"),
                                  n=args.rsi_n,
                                  overbought=args.rsi_overbought,
                                  oversold=args.rsi_oversold)
    return A, B, C, D

def snapshot_log(sym: str, t, A, B, C, D, iters: int, TRACE_ABCD: bool, TRACE_N: int):
    if TRACE_ABCD and (iters % max(1, int(TRACE_N))) == 0:
        def _summ_A(a):
            return f"A(up={a.get('up',0)},down={a.get('down',0)}," \
                   f"5m={a.get('5m','-')},15m={a.get('15m','-')}," \
                   f"30m={a.get('30m','-')},1h={a.get('1h','-')})"
        def _summ_B(b):
            flags = [k for k, v in b.items() if bool(v)]
            return "(" + (",".join(flags) if flags else "-") + ")"
        def _summ_C(c):
            return f"(regime={c.get('regime','UNKNOWN')},bb_width={c.get('bb_width',np.nan)})"
        def _summ_D(d):
            return f"(rsi={d.get('rsi',np.nan)},blockL={bool(d.get('block_long',False))},blockS={bool(d.get('block_short',False))})"
        logger_backtest.info(f"[{sym}] ABCD @ {t} | {_summ_A(A)} | B{_summ_B(B)} | C{_summ_C(C)} | D{_summ_D(D)}")

SYMBOLS_CANON = ["APTUSDT","BCHUSDT","BNBUSDT","BTCUSDT","CRVUSDT",
                 "DOGEUSDT","DOTUSDT","ETHUSDT","LINKUSDT","LTCUSDT",
                 "MKRUSDT","OPUSDT","SOLUSDT","SUIUSDT","WLDUSDT","XRPUSDT"]

FREQ_ID = {"15m":0, "30m":1, "1h":2, "2h":3, "5m":4, "4h":5}  # ← 4h ajouté

def _symbol_id(sym: str) -> int:
    return SYMBOLS_CANON.index(sym) if sym in SYMBOLS_CANON else -1

def _freq_id(freq: str) -> int:
    return FREQ_ID[freq] if freq in FREQ_ID else -1

def _side_num(side: str) -> int:
    return 1 if side.lower() == "buy" else -1

def record_signal(rows_store: dict, sym: str, year: int, args, t, tf, side, entry,
                  regime: str, bb_width: float, rsi_1h: float, atr14_tf: float):
    entry = float(entry)
    rows_store[sym][year].append({
        "t": pd.Timestamp(t).tz_convert("UTC").isoformat().replace("+00:00","Z"),
        "symbol": sym,
        "year": int(year),
        "tf": tf,
        "side": side,
        "side_num": _side_num(side),
        "entry": entry,
        "symbol_id": _symbol_id(sym),
        "freq_id": _freq_id(tf),
        "regime": regime if regime is not None else "UNKNOWN",
        "bb_width": float(bb_width) if np.isfinite(bb_width) else np.nan,
        "rsi_1h": float(rsi_1h) if np.isfinite(rsi_1h) else np.nan,
        "fees_bps": float(getattr(args, "fees_bps", 2.0)),
        "atr_bps": float(1e4 * atr14_tf / entry) if (atr14_tf is not None and np.isfinite(atr14_tf) and entry > 0) else np.nan,
    })

# =============================================================
# Main per-symbol loop
# =============================================================

def _a_dir(x: str | None) -> str:
    x = (x or "").upper()
    if x in ("UP","LONG","BULL"):  return "UP"
    if x in ("DOWN","SHORT","BEAR"): return "DOWN"
    return "-"

def _has_window(tf: str, windows_proj: dict[str, pd.DataFrame], need_by_freq: dict[str, int]) -> bool:
    df = windows_proj.get(tf)
    return (df is not None) and (not df.empty) and (len(df) >= int(need_by_freq.get(tf, 1)))

def choose_exec_tf_stage2(
    args, A, windows_proj, need_by_freq, side, allow_5m_fallback: bool = True,
) -> str | None:
    want = "UP" if (side or "buy").lower() == "buy" else "DOWN"
    a_map = {tf: _a_dir(A.get(tf)) for tf in ("5m","15m","30m","1h","2h","4h")}  # ← 4h inclus

    # Priorité TF hautes (4h en premier si demandé)
    for tf in ("4h","2h","1h","30m","15m"):
        if tf in args.freqs and _has_window(tf, windows_proj, need_by_freq):
            if a_map.get(tf) == want:
                return tf

    if allow_5m_fallback and "5m" in args.freqs:
        if _has_window("5m", windows_proj, need_by_freq) and a_map.get("5m") == want:
            return "5m"
    return None

def run_symbol(args, sym: str,
               window_map: dict, cooldown_map: dict, warmup_map: dict,
               TRACE_ABCD: bool, TRACE_N: int,
               global_log_handlers: dict,
               signals_by_sy: dict):

    # Optional file logging per symbol
    log_file_path = None
    log_s3_uri = None
    file_handler = None
    if args.log_to_s3:
        if args.log_local_path:
            log_file_path = args.log_local_path
            if os.path.isdir(log_file_path):
                log_file_path = os.path.join(log_file_path, _mk_log_name(sym, args.start, args.end))
        else:
            log_file_path = _mk_log_name(sym, args.start, args.end)
        try:
            file_handler = _attach_file_handler(logger_backtest, log_file_path)
            global_log_handlers[sym] = file_handler
            logger_backtest.info(f"[{sym}] file logging → {log_file_path}")
        except Exception as e:
            logger_backtest.error(f"[{sym}] échec attache file handler: {e}")
            file_handler = None

        prefix = args.log_s3_prefix.rstrip("/")
        log_s3_uri = f"{prefix}/{os.path.basename(log_file_path)}"
        logger_backtest.info(f"[{sym}] S3 target for logs → {log_s3_uri}")

    # ========= ENFORCEMENT: bougies annuelles complètes =========
    year0 = pd.to_datetime(args.start, utc=True).year
    require_full_year = (args.year is not None)
    _require_year_month_coverage(
        root_dir=args.root_parquet_dir,
        symbol=sym,
        year=year0,
        aws_region=args.aws_region,
        warn_ratio=0.5,
        start_iso=args.start,
        end_iso=args.end,
        require_full_year=require_full_year,
    )

    # Build feed
    feed = build_feed(args, sym, warmup_map, window_map)

    # Base diag
    base_freq = "15m" if "15m" in feed._by_freq else args.freqs[0]
    base = feed._by_freq[base_freq]
    if base.empty:
        raise RuntimeError(f"{sym}: série {base_freq} vide (check dates/parquets)")
    if not base["timestamp"].is_monotonic_increasing:
        base = base.sort_values("timestamp", kind="stable", ignore_index=True)

    first_ts, last_ts = base["timestamp"].iloc[0], base["timestamp"].iloc[-1]
    logger_backtest.info(f"[{sym}] {base_freq}: {len(base)} lignes | {first_ts} → {last_ts}")

    max_wait_sec = max((warmup_map.get(f, 0) + window_map.get(f, 1)) * _period_seconds(f) for f in args.freqs)
    earliest_yield = first_ts + pd.Timedelta(seconds=max_wait_sec)
    logger_backtest.info(f"[{sym}] premier yield ≥ {earliest_yield} (si end > cette date)")

    # Enrich
    enriched_by_freq, enriched_idx_by_freq = enrich_all_freqs(feed, args, sym)

    # Strategies
    strategies = {f: V1BreakoutStrategy(symbol=sym, frequency=f) for f in args.freqs}
    inject_ctx(strategies, enriched_idx_by_freq)
    strat0 = next(iter(strategies.values()))

    # Cooldown & loop state
    next_allowed_time = {f: pd.Timestamp.min.tz_localize("UTC") for f in args.freqs}
    need_by_freq = {f: window_map.get(f, 1) for f in args.freqs}
    last_t = None
    stuck_repeats = 0
    STUCK_MAX = 10000
    MAX_ITERS = int(1e8)

    iters = 0
    hits = 0
    trace = TraceCollector(sample_cap=400)

    try:
        for t, windows in feed.iter_windows():
            if _SHUTDOWN:
                logger_backtest.warning(f"[{sym}] Shutdown flag set → breaking main loop @ {t}")
                break

            iters += 1

            # Stuck guard
            if last_t is not None and t == last_t:
                stuck_repeats += 1
                if stuck_repeats % 1000 == 0:
                    logger_backtest.warning(f"[{sym}] ⚠️ t bloqué sur {t} depuis {stuck_repeats} itérations")
                if stuck_repeats >= STUCK_MAX:
                    logger_backtest.error(f"[{sym}] ❌ t ne progresse plus (t={t}, repeats={stuck_repeats}) — break sécurité")
                    break
            else:
                last_t = t
                stuck_repeats = 0

            if iters >= MAX_ITERS:
                logger_backtest.error(f"[{sym}] ❌ MAX_ITERS atteint ({MAX_ITERS}) — break sécurité")
                break

            if args.heartbeat and (iters % args.heartbeat == 0):
                total_signals = sum(len(rows) for by_year in signals_by_sy.values() for rows in by_year.values())
                logger_backtest.info(f"[{sym}] heartbeat: iters={iters}, hits={hits}, t={t}, signals={total_signals}")
                if args.log_to_s3 and args.log_upload_every_heartbeat and log_file_path and log_s3_uri:
                    try:
                        _upload_file_to_s3(log_file_path, log_s3_uri, args.aws_region)
                        logger_backtest.debug(f"[{sym}] heartbeat: log uploaded to {log_s3_uri}")
                    except Exception as e:
                        logger_backtest.warning(f"[{sym}] heartbeat: log upload failed: {e}")

            # Projeter fenêtres & blocs ABCD
            windows_proj = project_windows(windows, enriched_idx_by_freq, need_by_freq)
            A, B, C, D = compute_abcd(strat0, windows_proj, args)
            snapshot_log(sym, t, A, B, C, D, iters, TRACE_ABCD, TRACE_N)

            # Gate decision
            enter, side, reason = strat0.abcd_decide(A, B, C, D)
            if not enter or side is None:
                trace.add(DecisionTrace(ts=str(t), tf=base_freq, stage="gate",
                                        decision="REJECT", reason=reason,
                                        ctx={"regime": C.get("regime"), "bb_width": float(C.get("bb_width", np.nan)),
                                             "rsi": float(D.get("rsi", np.nan))}))
                continue

            # Choix TF exécution (2h→1h→30m→15m, fallback 5m)
            chosen_freq = choose_exec_tf_stage2(
                args=args,
                A=A,
                windows_proj=windows_proj,
                need_by_freq=need_by_freq,
                side=side,
                allow_5m_fallback=True,
            )
            if chosen_freq is None:
                trace.add(DecisionTrace(ts=str(t), tf="NA", stage="gate",
                                        decision="REJECT", reason="NO_ALIGNED_TF", ctx={"A": str(A)}))
                continue

            if chosen_freq == "5m":
                logger_backtest.debug(
                    f"[{sym} @ {t}] 5m fallback: A_1h={_a_dir(A.get('1h'))}, A_30m={_a_dir(A.get('30m'))}, A_15m={_a_dir(A.get('15m'))}"
                )

            # Cooldown
            cd_bars = int(cooldown_map.get(chosen_freq, 0))
            if cd_bars > 0:
                next_ok = next_allowed_time.get(chosen_freq, pd.Timestamp.min.tz_localize("UTC"))
                if t < next_ok:
                    trace.add(DecisionTrace(ts=str(t), tf=chosen_freq, stage="gate",
                                            decision="REJECT", reason="COOLDOWN",
                                            ctx={"next_ok": str(next_ok)}))
                    continue
                next_allowed_time[chosen_freq] = t + pd.Timedelta(seconds=_period_seconds(chosen_freq) * cd_bars)

            # Préparation du signal
            df_emit = windows_proj[chosen_freq]
            entry_price = float(df_emit["close"].iloc[-1])
            regime = C.get("regime", "UNKNOWN")
            bbw = float(C.get("bb_width", np.nan))
            rsi1h = float(D.get("rsi", np.nan))
            atr14_tf = float(df_emit["atr14"].iloc[-1]) if "atr14" in df_emit.columns else np.nan

            # Enregistrement (en mémoire uniquement)
            year = pd.Timestamp(t).year
            record_signal(
                signals_by_sy, sym, year, args, t, chosen_freq, side, entry_price,
                regime, bbw, rsi1h, atr14_tf
            )

            # Logs compacts
            _emit_abcd_signal_compact(sym, t, chosen_freq, side, reason, A, B, C, D)

            def _fmt_A(a: dict) -> str:
                return f"up={a.get('up',0)},down={a.get('down',0)}," \
                       f"5m={a.get('5m','-')},15m={a.get('15m','-')}," \
                       f"30m={a.get('30m','-')},1h={a.get('1h','-')}"
            def _fmt_B(b: dict) -> str:
                flags = [k for k, v in b.items() if bool(v)]
                return ",".join(flags) if flags else "-"

            logger_backtest.info(
                f"[TRACE] tf={chosen_freq} ts={t} stage=retail decision=SIGNAL reason={reason} | "
                f"A({_fmt_A(A)}) | B({_fmt_B(B)}) | "
                f"C(regime={C.get('regime','UNKNOWN')},bb_width={float(C.get('bb_width', float('nan')))}"
                f") | D(rsi={float(D.get('rsi', float('nan')))},"
                f"block_long={bool(D.get('block_long', False))},block_short={bool(D.get('block_short', False))})"
            )
            hits += 1

            # NOTE: plus de flush intermédiaires — on écrit à la fin seulement.

            if (iters % 20000) == 0:
                try:
                    s = trace.summary_lines()
                    logger_backtest.info(f"[{sym}] mid-trace " + " | ".join(s[:10]))
                except Exception:
                    pass

    finally:
        # ╭────────────────────────────────────────────────────────────╮
        # │   ÉCRITURE FINALE UNIQUE (par année) — FORCE HEADER CLEAN  │
        # ╰────────────────────────────────────────────────────────────╯
        s3_out_base = _s3_out_base_from_root(args.root_parquet_dir)
        is_s3 = isinstance(s3_out_base, str) and s3_out_base.startswith("s3://")
        so = {"client_kwargs": {"region_name": args.aws_region}} if is_s3 else {}
        fs = fsspec.filesystem("s3", **so) if is_s3 else None

        # colonnes finales et ordre imposé
        FINAL_COLS = [
            "t","symbol","year","tf","side","side_num","entry",
            "symbol_id","freq_id","regime","bb_width","rsi_1h",
            "fees_bps","atr_bps"
        ]

        total = 0
        for y, rows in list(signals_by_sy[sym].items()):
            out_path = f"{s3_out_base}/{sym}/{y}_signals.csv"

            # 1) supprime l’ancien fichier (si présent)
            try:
                if fs:
                    if fs.exists(out_path):
                        fs.rm(out_path)
                        logger_backtest.info(f"[{sym}] cleanup: removed old {out_path}")
                else:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    if os.path.exists(out_path):
                        os.remove(out_path)
                        logger_backtest.info(f"[{sym}] cleanup: removed old {out_path}")
            except Exception as e:
                logger_backtest.warning(f"[{sym}] cleanup: failed to remove {out_path}: {e}")

            # 2) DataFrame neuf + ordre de colonnes figé
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # ajoute les colonnes manquantes au besoin (et dans l’ordre)
            for c in FINAL_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[FINAL_COLS]

            # 3) écriture en 'wb' + header=True (pas d’APPEND possible)
            try:
                if fs:
                    # assure le dossier
                    parent = f"{s3_out_base}/{sym}"
                    if not fs.exists(parent):
                        fs.mkdirs(parent, exist_ok=True)
                    with fsspec.open(out_path, "wb", **so) as f:
                        df.to_csv(f, index=False, header=True)
                else:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as f:
                        df.to_csv(f, index=False, header=True)
                total += len(df)
                logger_backtest.info(f"[{sym}] 📤 wrote {len(df)} rows to {out_path} (CREATE)")
            except Exception as e:
                logger_backtest.error(f"[{sym}] write failed for {out_path}: {e}")

            signals_by_sy[sym][y].clear()

        logger_backtest.info(f"[{sym}] FINAL write done — total rows: {total}")

        # Trace summary fin
        for line in trace.summary_lines():
            logger_backtest.info(f"[{sym}] TRACE {line}")

        # Upload final du log si demandé
        if args.log_to_s3 and (log_file_path is not None):
            try:
                _upload_file_to_s3(log_file_path, log_s3_uri, args.aws_region)
                logger_backtest.info(f"[{sym}] log uploaded to {log_s3_uri}")
            except Exception as e:
                logger_backtest.error(f"[{sym}] log upload failed: {e}")

        # Detach file handler
        fh = global_log_handlers.get(sym)
        if fh:
            try:
                logger_backtest.removeHandler(fh)
                fh.close()
            except Exception:
                pass

        logger_backtest.info(f"[{sym}] Fin replay | iters={iters}, hits={hits}")
        logger_backtest.info("---------------------------------------------------------------")
                
# =============================================================
# main()
# =============================================================
def main():
    args = parse_args()
    window_map, cooldown_map, warmup_map, _ = normalize_args(args)
    TRACE_ABCD, TRACE_N = trace_profile(args)
    recap(args, warmup_map, window_map, cooldown_map)

    signals_by_sy: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    global_log_handlers: dict[str, Any] = {}

    for sym in args.symbols:
        try:
            run_symbol(args, sym, window_map, cooldown_map, warmup_map,
                       TRACE_ABCD, TRACE_N, global_log_handlers, signals_by_sy)
        except Exception as e:
            logger_backtest.error(f"[{sym}] ❌ run failed: {e}")
            raise

if __name__ == "__main__":
    main()