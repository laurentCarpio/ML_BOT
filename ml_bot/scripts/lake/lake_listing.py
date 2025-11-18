#!/usr/bin/env python3
# list_usdt_perp_symbols_full.py — Log exhaustif des symboles *-USDT-PERP par jour et récap final
from __future__ import annotations
import os, sys, argparse, datetime as dt, time, re
from typing import Set, List, Optional

import boto3
import pandas as pd
import lakeapi
from lakeapi.exceptions import NoFilesFound

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Lister les symboles disponibles (trades & book) filtrés par quote/suffixe, sortie exhaustive.")
    ap.add_argument("--exchange", default="BINANCE_FUTURES")
    ap.add_argument("--lake-bucket", default="qnt.data")
    ap.add_argument("--lake-region", default="eu-west-1")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end",   default="2024-01-01")
    ap.add_argument("--tables", default="trades,book", help="Ex: trades,book | ou uniquement trades")
    ap.add_argument("--step-days", type=int, default=3, help="Pas d’échantillonnage (jours)")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=1.6)
    # Filtre symboles
    ap.add_argument("--quote", default="USDT", help="Quote à garder (ex: USDT)")
    ap.add_argument("--perp-suffix", default="-PERP", help="Suffixe à garder (ex: -PERP)")
    # Fallback si l’API exige un symbole (facultatif)
    ap.add_argument("--batch-symbols", default="BTC-USDT-PERP,ETH-USDT-PERP,SOL-USDT-PERP,BNB-USDT-PERP,XRP-USDT-PERP,ADA-USDT-PERP,MATIC-USDT-PERP,LTC-USDT-PERP,AVAX-USDT-PERP,DOGE-USDT-PERP")
    # Affichage
    ap.add_argument("--per-day-limit", type=int, default=0, help="0 = tout afficher; sinon limite par jour")
    ap.add_argument("--per-day-columns", type=int, default=8, help="Nb de symboles par ligne pour l’affichage journalier")
    return ap.parse_args()

# -----------------------
# Utils
# -----------------------
def daterange(start: dt.date, end: dt.date, step_days: int):
    d = start
    step = dt.timedelta(days=step_days)
    while d < end:
        yield d
        d += step

def ensure_session(region: str):
    key = os.getenv("LAKEAPI_ACCESS_KEY_ID")
    sec = os.getenv("LAKEAPI_SECRET_ACCESS_KEY")
    if not key or not sec:
        print("⛔ LAKEAPI_ACCESS_KEY_ID / LAKEAPI_SECRET_ACCESS_KEY non présents", file=sys.stderr)
        sys.exit(2)
    return boto3.Session(aws_access_key_id=key, aws_secret_access_key=sec, region_name=region)

def try_load(table: str, day_start: dt.datetime, exchange: str, sess, bucket: str, max_retries: int, backoff: float, symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    tries = 0
    while tries <= max_retries:
        tries += 1
        try:
            df = lakeapi.load_data(
                table=table,
                start=day_start,
                end=day_start + dt.timedelta(days=1),
                symbols=symbols,            # None = sans filtre
                exchanges=[exchange],
                boto3_session=sess,
                bucket=bucket,
                cached=False,
            )
            if df is None or len(df) == 0:
                return None
            return df
        except NoFilesFound:
            return None
        except Exception as e:
            if tries > max_retries:
                print(f"❌ load_data {table} {day_start.date()} échec après {tries-1} retries: {e}", file=sys.stderr)
                return None
            time.sleep(backoff ** tries)

def filter_symbols(syms: List[str], quote: str, perp_suffix: str) -> List[str]:
    pat = re.compile(fr"-{re.escape(quote)}{re.escape(perp_suffix)}$", re.IGNORECASE)
    out = sorted({s for s in syms if s and pat.search(str(s))})
    return out

def print_symbols_block(prefix: str, syms: List[str], per_line: int = 8, limit: int = 0):
    """Affiche tous les symboles (ou 'limit' si > 0) en colonnes, sans ellipses."""
    if not syms:
        print(f"{prefix} (aucun)")
        return
    to_show = syms if limit <= 0 else syms[:limit]
    print(f"{prefix} ({len(to_show)}/{len(syms)}) :")
    for i in range(0, len(to_show), per_line):
        print("  " + ", ".join(to_show[i:i+per_line]))

def discover_symbols(table: str, exchange: str, start: dt.date, end: dt.date, sess, bucket: str, step_days: int,
                     max_retries: int, backoff: float, fallback_symbols: List[str],
                     quote: str, perp_suffix: str, per_day_limit: int, per_day_columns: int) -> List[str]:
    found: Set[str] = set()
    print(f"\n──────── Scan {table} — {exchange} — {start} → {end} (pas {step_days}j) ────────")

    # 1) Essai sans filtre symbol
    for day in daterange(start, end, step_days):
        df = try_load(table, dt.datetime.combine(day, dt.time.min), exchange, sess, bucket, max_retries, backoff, symbols=None)
        if df is None or len(df) == 0:
            continue
        cols_lower = [c.lower() for c in df.columns]
        if "symbol" in cols_lower:
            sym_col = df.columns[cols_lower.index("symbol")]
            syms = df[sym_col].astype("string").dropna().str.strip().unique().tolist()
            kept = filter_symbols(syms, quote, perp_suffix)
            if kept:
                kept_sorted = sorted(kept)
                found.update(kept_sorted)
                print_symbols_block(f"✅ {day} : symboles *-{quote.upper()}{perp_suffix.upper()}", kept_sorted,
                                    per_line=per_day_columns, limit=per_day_limit)
        else:
            print(f"ℹ️ {day} : données présentes mais pas de colonne 'symbol' — fallback batch…")

    if found:
        return sorted(found)

    # 2) Fallback : test de symboles connus si l’API exige un symbole
    print("↪️ Fallback par symboles connus…")
    for sym in fallback_symbols:
        if not sym.upper().endswith(f"-{quote.upper()}{perp_suffix.upper()}"):
            continue
        for day in daterange(start, end, step_days):
            df = try_load(table, dt.datetime.combine(day, dt.time.min), exchange, sess, bucket, max_retries, backoff, symbols=[sym])
            if df is not None and len(df) > 0:
                found.add(sym)
                print_symbols_block(f"✅ {table} {day} : détection", [sym], per_line=per_day_columns, limit=0)
                break
    return sorted(found)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    sess = ensure_session(args.lake_region)

    start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date   = dt.datetime.strptime(args.end,   "%Y-%m-%d").date()

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    fallback_symbols = [s.strip() for s in args.batch_symbols.split(",") if s.strip()]

    print("═══════════════════════════════════════════════════════════════════════════")
    print(f"Exchange : {args.exchange}")
    print(f"Tables   : {tables}")
    print(f"Date     : {args.start} → {args.end} (excl.) | pas = {args.step_days} j")
    print(f"Lake     : s3://{args.lake_bucket} | region = {args.lake_region}")
    print(f"Filtre   : *-{args.quote.upper()}{args.perp_suffix.upper()}")
    print(f"Affiche  : --per-day-limit={args.per_day_limit} (0 = tout) | --per-day-columns={args.per_day_columns}")
    print("═══════════════════════════════════════════════════════════════════════════")

    results = {}
    for table in tables:
        syms = discover_symbols(
            table=table,
            exchange=args.exchange,
            start=start_date,
            end=end_date,
            sess=sess,
            bucket=args.lake_bucket,
            step_days=args.step_days,
            max_retries=args.max_retries,
            backoff=args.backoff,
            fallback_symbols=fallback_symbols,
            quote=args.quote,
            perp_suffix=args.perp_suffix,
            per_day_limit=args.per_day_limit,
            per_day_columns=args.per_day_columns,
        )
        results[table] = syms

    print("\n==================== RÉSULTATS (filtrés) ====================")
    for table, syms in results.items():
        print_symbols_block(f"{table.upper()} — symboles *-{args.quote.upper()}{args.perp_suffix.upper()} détectés (ensemble total)",
                           syms, per_line=12, limit=0)
    print("\nTerminé.")

if __name__ == "__main__":
    main()