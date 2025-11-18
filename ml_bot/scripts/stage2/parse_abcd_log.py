#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parser strict (no-legacy, no-fallback) des logs ABCD (Étape 1 / retail).

Il attend OBLIGATOIREMENT une ligne TRACE pour chaque signal :
  [TRACE] tf=<TF> ts=<ISO> stage=retail decision=SIGNAL reason=<REASON> | A(...) | B(...) | C(...) | D(...)

Et des snapshots facultatifs :
  [SYMBOL] ABCD @ <ts> | A(...) | B(...) | C(...) | D(...)

Et, éventuellement, des rejets gate :
  [TRACE] tf=<TF> ts=<ISO> stage=gate decision=REJECT reason=<REASON> | <ctx_kv_optionnel>

Si un bloc A/B/C/D manque dans la ligne SIGNAL → le script lève une exception.
"""

from __future__ import annotations
import re
import json
import argparse
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, Any, Optional, Iterable

import fsspec
import pandas as pd

# ---------- Regex (STRICT) ----------
RE_SNAP = re.compile(
    r"""\[\w+\]\s+ABCD\s+@\s+(?P<ts>[^|]+)\|\s+
        A\((?P<A>[^)]*)\)\s+\|\s+
        B\((?P<B>[^)]*)\)\s+\|\s+
        C\((?P<C>[^)]*)\)\s+\|\s+
        D\((?P<D>[^)]*)\)
    """,
    re.VERBOSE,
)

RE_SIGNAL = re.compile(
    r"""\[TRACE\]\s+tf=(?P<tf>\S+)\s+ts=(?P<ts>.+?)\s+
        stage=retail\s+decision=SIGNAL\s+reason=(?P<reason>[^|]+)\s*\|\s*
        A\((?P<A>[^)]*)\)\s+\|\s+
        B\((?P<B>[^)]*)\)\s+\|\s+
        C\((?P<C>[^)]*)\)\s+\|\s+
        D\((?P<D>[^)]*)\)
    """,
    re.VERBOSE,
)

RE_GATE = re.compile(
    r"""\[TRACE\]\s+tf=(?P<tf>\S+)\s+ts=(?P<ts>.+?)\s+
        stage=gate\s+decision=REJECT\s+reason=(?P<reason>[^|]+)\s*\|\s*(?P<ctx>.*)
    """,
    re.VERBOSE,
)

# ---------- Helpers ----------
def _parse_kv_block(block: str) -> Dict[str, Any]:
    """
    "up=1,down=0,5m=UP" → dict ; "FLAG_X" → True.
    """
    block = block.strip()
    if not block or block == "-":
        return {}
    out: Dict[str, Any] = {}
    for part in block.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            out[part] = True
            continue
        k, v = [x.strip() for x in part.split("=", 1)]
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                out[k] = float(v) if "." in v else int(v)
            except Exception:
                out[k] = v
    return out

def _open_lines(path: str, aws_region: Optional[str]) -> Iterable[str]:
    if path.startswith("s3://"):
        so = {"client_kwargs": {"region_name": aws_region}} if aws_region else {}
        with fsspec.open(path, "rt", encoding="utf-8", **so) as f:
            for line in f:
                yield line.rstrip("\n")
    else:
        with open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")

def _series_stats(values):
    if not values:
        return None
    s = pd.Series(values)
    return {
        "count": int(len(values)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }

# ---------- Stats aggregator ----------
class Stats:
    def __init__(self):
        self.signal_count = 0
        self.signal_reasons = Counter()
        self.signal_regime_counts = Counter()
        self.signal_bbw_by_regime = defaultdict(list)
        self.signal_rsi_values = []

        self.gate_reasons = Counter()

        self.A_ups = []
        self.A_dns = []
        self.A_tf_state_counts = Counter()
        self.B_flags = Counter()

        self.regime_counts_all = Counter()
        self.bbw_all_by_regime = defaultdict(list)

        self.first_ts: Optional[pd.Timestamp] = None
        self.last_ts: Optional[pd.Timestamp] = None

    def _bump_ts(self, ts: str):
        try:
            t = pd.Timestamp(ts.strip())
            if self.first_ts is None or t < self.first_ts:
                self.first_ts = t
            if self.last_ts is None or t > self.last_ts:
                self.last_ts = t
        except Exception:
            pass

    def add_snapshot(self, ts: str, A: str, B: str, C: str, D: str):
        self._bump_ts(ts)
        AA = _parse_kv_block(A)
        BB = _parse_kv_block(B)
        CC = _parse_kv_block(C)
        # DD = _parse_kv_block(D)  # non utilisé dans les stats globales

        self.A_ups.append(int(AA.get("up", 0)))
        self.A_dns.append(int(AA.get("down", 0)))
        for tf in ("1m","3m","5m","15m","30m","1h","2h","4h"):
            v = AA.get(tf)
            if isinstance(v, str) and v:
                self.A_tf_state_counts[(tf, v)] += 1

        for k, v in _parse_kv_block(B).items():
            if v is True:
                self.B_flags[k] += 1

        regime = CC.get("regime")
        bbw = CC.get("bb_width")
        if regime is not None:
            self.regime_counts_all[str(regime)] += 1
        if isinstance(bbw, (int, float)):
            self.bbw_all_by_regime[str(regime)].append(float(bbw))

    def add_gate(self, ts: str, reason: str, ctx: str):
        self._bump_ts(ts)
        self.gate_reasons[reason.strip()] += 1
        # ctx gate optionnel — on glane regime/bbw si présents en k=v
        for kv in re.split(r"\s*,\s*|\s+", (ctx or "").strip()):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "regime":
                self.regime_counts_all[str(v)] += 1
            elif k == "bb_width":
                try:
                    self.bbw_all_by_regime["UNKNOWN"].append(float(v))
                except Exception:
                    pass

    def add_signal(self, ts: str, reason: str, A: str, B: str, C: str, D: str):
        self._bump_ts(ts)
        self.signal_count += 1
        self.signal_reasons[reason.strip()] += 1

        AA = _parse_kv_block(A)
        BB = _parse_kv_block(B)
        CC = _parse_kv_block(C)
        DD = _parse_kv_block(D)

        # Exigences strictes pour Étape 1
        if "regime" not in CC or "bb_width" not in CC:
            raise ValueError(f"C.regime/bb_width manquant dans SIGNAL @ {ts}: C={CC}")
        if "rsi" not in DD:
            raise ValueError(f"D.rsi manquant dans SIGNAL @ {ts}: D={DD}")

        regime = str(CC["regime"])
        bbw = float(CC["bb_width"])
        self.signal_regime_counts[regime] += 1
        self.signal_bbw_by_regime[regime].append(bbw)

        try:
            self.signal_rsi_values.append(float(DD["rsi"]))
        except Exception:
            raise ValueError(f"D.rsi non numérique dans SIGNAL @ {ts}: D={DD}")

        # Alimenter aussi les totaux globaux
        self.regime_counts_all[regime] += 1
        self.bbw_all_by_regime[regime].append(bbw)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        out["period"] = {
            "first_ts": None if self.first_ts is None else str(self.first_ts),
            "last_ts":  None if self.last_ts  is None else str(self.last_ts),
        }

        out["decisions"] = {
            "retail_signals": self.signal_count,
            "retail_signals_by_reason": dict(self.signal_reasons.most_common()),
            "gate_rejects_total": int(sum(self.gate_reasons.values())),
            "gate_by_reason": dict(self.gate_reasons.most_common()),
        }

        out["regime"] = {
            "signals_counts": dict(self.signal_regime_counts.most_common()),
            "all_counts": dict(self.regime_counts_all.most_common()),
        }

        out["bb_width_at_signals_by_regime"] = {k: _series_stats(v) for k, v in self.signal_bbw_by_regime.items()}
        out["bb_width_all_by_regime"]        = {k: _series_stats(v) for k, v in self.bbw_all_by_regime.items()}
        out["rsi_at_signals"]                = _series_stats(self.signal_rsi_values) or {"count": 0}

        out["A_block"] = {
            "ups_mean": mean(self.A_ups) if self.A_ups else 0.0,
            "dns_mean": mean(self.A_dns) if self.A_dns else 0.0,
            "ups_ge_2_ratio": float(sum(1 for x in self.A_ups if x >= 2)) / len(self.A_ups) if self.A_ups else 0.0,
            "dns_ge_2_ratio": float(sum(1 for x in self.A_dns if x >= 2)) / len(self.A_dns) if self.A_dns else 0.0,
            "tf_state_counts": {f"{tf}:{state}": cnt for (tf, state), cnt in self.A_tf_state_counts.most_common()},
        }

        out["B_flags"] = dict(self.B_flags.most_common())
        return out

# ---------- Parsing ----------
def parse_file(path: str, aws_region: Optional[str] = None) -> Stats:
    st = Stats()
    for line in _open_lines(path, aws_region):
        m = RE_SNAP.search(line)
        if m:
            st.add_snapshot(m.group("ts"), m.group("A"), m.group("B"), m.group("C"), m.group("D"))
            continue
        m = RE_SIGNAL.search(line)
        if m:
            st.add_signal(
                ts=m.group("ts"),
                reason=m.group("reason"),
                A=m.group("A"),
                B=m.group("B"),
                C=m.group("C"),
                D=m.group("D"),
            )
            continue
        m = RE_GATE.search(line)
        if m:
            st.add_gate(m.group("ts"), m.group("reason"), m.group("ctx") or "")
            continue
    return st

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="s3://tradebot-config-tokyo/data/logs/replay_ETHUSDT_2024-01-01_2025-01-01_20250917T193551Z.log")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--out-json", default=None, help="Écrit le JSON des stats (optionnel)")
    args = ap.parse_args()

    stats = parse_file(args.log, args.aws_region)
    summ = stats.summary()

    print("===== ABCD LOG STATS =====")
    per = summ["period"]
    print(f"Période: {per['first_ts']} → {per['last_ts']}")

    dec = summ["decisions"]
    print(f"Retail SIGNALS: {dec['retail_signals']}")
    if dec["retail_signals_by_reason"]:
        print("SIGNALS par reason:")
        for r, c in dec["retail_signals_by_reason"].items():
            print(f"  - {r}: {c}")

    if dec["gate_rejects_total"] > 0:
        print("\nGate REJECT par raison:")
        for r, c in dec["gate_by_reason"].items():
            print(f"  - {r}: {c}")

    print("\nRegime counts @ SIGNALS:")
    for k, v in summ["regime"]["signals_counts"].items():
        print(f"  - {k}: {v}")

    print("\nbb_width @ SIGNAL (count, min, p25, median, p75, max, mean):")
    for k, d in (summ["bb_width_at_signals_by_regime"] or {}).items():
        if d:
            print(f"  - {k}: n={d['count']} min={d['min']:.6f} p25={d['p25']:.6f} "
                  f"med={d['median']:.6f} p75={d['p75']:.6f} max={d['max']:.6f} mean={d['mean']:.6f}")

    rsi_sig = summ["rsi_at_signals"]
    print("\nRSI @ SIGNAL:")
    if rsi_sig.get("count", 0) > 0:
        print(f"  count={rsi_sig['count']} min={rsi_sig['min']:.2f} med={rsi_sig['median']:.2f} "
              f"max={rsi_sig['max']:.2f} mean={rsi_sig['mean']:.2f}")
    else:
        print("  count=0")

    Ab = summ["A_block"]
    print("\nA-block (global snapshots):")
    print(f"  ups_mean={Ab.get('ups_mean', 0):.3f} | dns_mean={Ab.get('dns_mean', 0):.3f}")
    print(f"  ratio ups>=2 = {Ab.get('ups_ge_2_ratio', 0):.3%} | ratio downs>=2 = {Ab.get('dns_ge_2_ratio', 0):.3%}")
    tfc = Ab.get("tf_state_counts", {})
    if tfc:
        print("  états TF (top 10):")
        for i, (k, v) in enumerate(tfc.items()):
            if i >= 10:
                break
            print(f"    - {k}: {v}")

    print("\nB flags (top 10):")
    for i, (k, v) in enumerate((summ.get("B_flags", {}) or {}).items()):
        if i >= 10:
            break
        print(f"  - {k}: {v}")

    if args.out_json:
        with fsspec.open(args.out_json, "wt", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)
        print(f"\n→ JSON écrit: {args.out_json}")

if __name__ == "__main__":
    main()