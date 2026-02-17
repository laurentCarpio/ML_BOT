#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
portfolio_backtest.py

Portfolio backtest (capital partagé) sur des sélections TEST "déjà prêtes"
(sel_test_sized.parquet par symbole).

Hypothèses simples mais réalistes:
- Chaque signal ouvre une position à t0, horizon fixe (ex 60 min) => fin à t0 + horizon
- Le trade retourne un R (ici R_base ou R_sized), pas de MFE/MAE
- Budget de risque total (risk_budget) + risque par trade (risk_per_trade)
- Limite max trades simultanés (max_concurrent)
- Si plusieurs signaux arrivent au même timestamp, on prend les meilleurs (score desc)

Entrées attendues par fichier:
- t0 (datetime)
- R_base (float) optionnel
- R_sized (float) optionnel
- pB (float) optionnel (score)
- symbol (sinon injecté via l'argument)

Sorties:
- portfolio_trades.parquet : trades pris (avec allocation, start/end, R contrib)
- portfolio_equity.parquet : equity curve en R
- portfolio_monthly.parquet : stats mensuelles
"""

import argparse
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs


# -------------------------
# IO helpers (S3 + local)
# -------------------------
def read_parquet_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_input_file(s3_path) as f:
            return pq.read_table(f).to_pandas()
    return pq.read_table(path).to_pandas()


def write_parquet_any(df: pd.DataFrame, path: str):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_output_stream(s3_path) as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, path)


def apply_fees_R(
    U: pd.DataFrame,
    *,
    rcol: str,
    fee_bps_rt: float,
    risk_bps_col: str = "R_bps",
    risk_bps_default: float | None = None,
) -> pd.DataFrame:
    """
    Convertit des fees en bps aller-retour en "R" et les soustrait au R du trade.

    fee_R = fee_bps_rt / risk_bps
    R_net = R_raw - fee_R

    - risk_bps_col doit contenir 1R en bps par trade (distance SL en bps).
    - si absent, on peut fallback avec risk_bps_default (moins propre).
    """
    out = U.copy()
    out[rcol] = pd.to_numeric(out[rcol], errors="coerce")

    if fee_bps_rt <= 0:
        out["R_net"] = out[rcol].astype(float)
        return out

    if risk_bps_col in out.columns:
        risk_bps = pd.to_numeric(out[risk_bps_col], errors="coerce").astype(float)
        # évite division par 0 / NaN
        bad = (~np.isfinite(risk_bps)) | (risk_bps <= 0)
        if bad.any():
            if risk_bps_default is None:
                raise RuntimeError(
                    f"fee-bps-rt={fee_bps_rt} but '{risk_bps_col}' has NaN/<=0 for {int(bad.sum())} rows "
                    f"and no --risk-bps-default provided."
                )
            risk_bps = risk_bps.where(~bad, float(risk_bps_default))
    else:
        if risk_bps_default is None:
            raise RuntimeError(
                f"fee-bps-rt={fee_bps_rt} requires column '{risk_bps_col}' or --risk-bps-default"
            )
        risk_bps = float(risk_bps_default)

    fee_R = float(fee_bps_rt) / risk_bps
    out["fee_R"] = fee_R
    out["R_net"] = out[rcol].astype(float) - out["fee_R"].astype(float)
    return out

# -------------------------
# Stats
# -------------------------
def stats_R(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0}
    win = float(np.mean(x > 0))
    loss = float(np.mean(x < 0))
    flat = float(np.mean(x == 0))
    exp = float(np.mean(x))
    gp = float(np.sum(x[x > 0]))
    gl = float(-np.sum(x[x < 0]))
    pf = (gp / gl) if gl > 0 else np.nan
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    mdd = float(np.min(dd)) if len(dd) else np.nan
    return {
        "n": int(n),
        "expectancy_R": exp,
        "sum_R": float(np.sum(x)),
        "win_rate": win,
        "loss_rate": loss,
        "flat_rate": flat,
        "profit_factor": pf,
        "max_drawdown_R": mdd,
    }


def monthly_stats(trades: pd.DataFrame, rcol: str) -> pd.DataFrame:
    tmp = trades.copy()
    tmp["t0"] = pd.to_datetime(tmp["t0"], utc=True)
    tmp["ym"] = tmp["t0"].dt.to_period("M").astype(str)
    out = (
        tmp.groupby("ym", sort=True)[rcol]
        .apply(lambda s: pd.Series(stats_R(s.to_numpy())))
        .reset_index()
    )
    return out


# -------------------------
# Portfolio engine
# -------------------------
@dataclass
class ActiveTrade:
    end_ts: pd.Timestamp
    risk_alloc: float


def parse_inputs(inputs: List[str]) -> List[Tuple[str, str]]:
    """
    inputs format (repeatable):
      --input BTCUSDT=s3://.../sel_test_sized.parquet
      --input ETHUSDT=s3://.../sel_test_sized.parquet
    """
    out = []
    for item in inputs:
        if "=" not in item:
            raise ValueError(f"Bad --input format: {item} (expected SYMBOL=PATH)")
        sym, path = item.split("=", 1)
        sym = sym.strip()
        path = path.strip()
        out.append((sym, path))
    return out


def load_universe(pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    dfs = []
    for sym, path in pairs:
        df = read_parquet_any(path)
        if "t0" not in df.columns:
            # fallback index
            df = df.copy()
            df["t0"] = pd.to_datetime(df.index, utc=True)
        else:
            df = df.copy()
            df["t0"] = pd.to_datetime(df["t0"], utc=True)

        if "symbol" not in df.columns:
            df["symbol"] = sym
        else:
            df["symbol"] = df["symbol"].fillna(sym)

        # choose columns we care about
        # tolerate missing pB; then score will fall back to 0
        keep = [c for c in ["t0", "symbol", "pA", "pB", "mult", "R", "R_base", "R_sized", "R_bps"] if c in df.columns]
        df = df[keep].copy()

        # normalize R columns
        if "R_base" not in df.columns:
            if "R" in df.columns:
                df["R_base"] = df["R"].astype(float)
        if "R_sized" not in df.columns:
            # if no sized provided, default to base
            df["R_sized"] = df["R_base"].astype(float)

        dfs.append(df)

    U = pd.concat(dfs, axis=0, ignore_index=True)
    U = U.sort_values("t0").reset_index(drop=True)
    return U


def run_portfolio(
    U: pd.DataFrame,
    *,
    horizon_min: int,
    risk_budget: float,
    risk_per_trade: float,
    max_concurrent: int,
    rcol: str,
    score_col: str,
    fee_bps_rt: float = 0.0,
    risk_bps_col: str = "R_bps",
    risk_bps_default: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      trades_taken (rows = trades)
      equity (time series by trade close)
    """
    U = U.copy()

    # apply fees in R-units (net)
    # NB: we keep rcol as input but use R_net internally when present
    if "fee_bps_rt" in U.columns:
        pass  # not used; kept for clarity

    U = apply_fees_R(
        U,
        rcol=rcol,
        fee_bps_rt=fee_bps_rt,
        risk_bps_col=risk_bps_col,
        risk_bps_default=risk_bps_default,
    )

    # on backteste toujours sur la colonne "net" si fees actives
    pnl_col = "R_net"

    # remove NaNs in R
    U[pnl_col] = pd.to_numeric(U[pnl_col], errors="coerce")
    U = U[np.isfinite(U[pnl_col])].copy()


    # score fallback
    if score_col not in U.columns:
        U[score_col] = 0.0
    U[score_col] = pd.to_numeric(U[score_col], errors="coerce").fillna(0.0)

    # group by timestamp to resolve “same t0”
    U = U.sort_values(["t0", score_col], ascending=[True, False]).reset_index(drop=True)

    active: List[ActiveTrade] = []
    taken_rows = []
    equity_points = []
    cum_R = 0.0

    horizon = pd.Timedelta(minutes=int(horizon_min))

    # iterate timestamps blocks
    for ts, block in U.groupby("t0", sort=True):
        ts = pd.Timestamp(ts)

        # 1) expire trades
        active = [a for a in active if a.end_ts > ts]
        used_risk = float(sum(a.risk_alloc for a in active))

        # 2) capacity left
        slots_left = max(0, int(max_concurrent) - len(active))
        risk_left = float(risk_budget - used_risk)

        if slots_left <= 0 or risk_left < risk_per_trade:
            continue

        # 3) take best signals in this block
        for _, row in block.iterrows():
            if slots_left <= 0 or risk_left < risk_per_trade:
                break

            # allocate fixed risk per trade
            alloc = float(risk_per_trade)

            # trade contribution in R-units weighted by risk alloc (optional)
            # Here: we interpret R as “per 1 risk unit”.
            # So portfolio delta_R = alloc * R
            r = float(row[pnl_col])
            delta = alloc * r

            end_ts = ts + horizon
            active.append(ActiveTrade(end_ts=end_ts, risk_alloc=alloc))

            cum_R += delta

            taken_rows.append({
                "t0": ts,
                "t1": end_ts,
                "symbol": row.get("symbol", ""),
                "score": float(row.get(score_col, 0.0)),
                "risk_alloc": alloc,
                "R_trade_raw": float(row.get(rcol, np.nan)),
                "fee_R": float(row.get("fee_R", 0.0)),
                "R_trade_net": r,
                "R_contrib": delta,
                "cum_R_after": cum_R,
            })

            equity_points.append({
                "t_close": end_ts,
                "delta_R": delta,
                "cum_R": cum_R,
            })

            slots_left -= 1
            risk_left -= alloc

    trades = pd.DataFrame(taken_rows)
    equity = pd.DataFrame(equity_points)

    if len(equity):
        equity = equity.sort_values("t_close").reset_index(drop=True)
    return trades, equity


def main():
    ap = argparse.ArgumentParser("Portfolio backtest multi-assets (capital partagé)")
    ap.add_argument("--input", action="append", required=True,
                    help="SYMBOL=PATH (repeatable). Example: --input BTCUSDT=s3://.../sel_test_sized.parquet")

    ap.add_argument("--horizon-min", type=int, default=60)
    ap.add_argument("--risk-budget", type=float, default=1.0,
                    help="Total risk budget for concurrent trades (1.0 = 100%)")
    ap.add_argument("--risk-per-trade", type=float, default=0.25,
                    help="Risk allocated per trade (0.25 => 4 trades max if budget=1.0)")
    ap.add_argument("--max-concurrent", type=int, default=4)

    ap.add_argument("--use-sized", action="store_true",
                    help="Use R_sized instead of R_base for portfolio PnL")
    ap.add_argument("--score-col", default="pB",
                    help="Column used to rank signals at same t0 (default pB)")
    
    ap.add_argument("--fee-bps-rt", type=float, default=0.0,
                help="Fees aller-retour en bps (ex: 6.0). 0 = ignore.")
    ap.add_argument("--risk-bps-col", default="R_bps",
                help="Colonne qui représente 1R en bps (distance SL) par trade. Ex: R_bps.")
    ap.add_argument("--risk-bps-default", type=float, default=None,
                help="Fallback si risk_bps_col absent. (pas idéal, mais évite crash)")

    ap.add_argument("--out-trades", required=True)
    ap.add_argument("--out-equity", required=True)
    ap.add_argument("--out-monthly", required=True)

    args = ap.parse_args()

    pairs = parse_inputs(args.input)
    U = load_universe(pairs)

    rcol = "R_sized" if args.use_sized else "R_base"

    trades, equity = run_portfolio(
        U,
        horizon_min=args.horizon_min,
        risk_budget=args.risk_budget,
        risk_per_trade=args.risk_per_trade,
        max_concurrent=args.max_concurrent,
        rcol=rcol,
        score_col=args.score_col,
        fee_bps_rt=args.fee_bps_rt,
        risk_bps_col=args.risk_bps_col,
        risk_bps_default=args.risk_bps_default,
    )

    print("===================================")
    print("PORTFOLIO RESULTS")
    print("===================================")
    print(f"signals_loaded: {len(U):,}")
    print(f"trades_taken : {len(trades):,}")
    if len(trades):
        print("stats (R_contrib):", stats_R(trades["R_contrib"].to_numpy()))
        monthly = monthly_stats(trades.rename(columns={"t0": "t0"}), "R_contrib")
        print("\nMONTHLY (R_contrib):")
        print(monthly)
    else:
        monthly = pd.DataFrame()

    write_parquet_any(trades, args.out_trades)
    write_parquet_any(equity, args.out_equity)
    write_parquet_any(monthly, args.out_monthly)

    print("\n[write]")
    print("trades :", args.out_trades)
    print("equity :", args.out_equity)
    print("monthly:", args.out_monthly)
    print("✅ Done.")


if __name__ == "__main__":
    main()