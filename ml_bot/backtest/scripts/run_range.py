from __future__ import annotations
import argparse, json

from ml_bot.backtest.lib.utils import months_between
from ml_bot.backtest.scripts.stage0b_evalready import build_evalready_month
from ml_bot.backtest.scripts.ms_edge_backtest import run_backtest


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml_bot/backtest/config/ms_edge_configs.json")
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--mode", choices=["tagged", "evalready"], default="tagged")
    ap.add_argument("--make-evalready", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    months = months_between(args.start, args.end)

    if args.make_evalready:
        for m in months:
            out = build_evalready_month(cfg, month=m)
            print("evalready wrote:", out)

    run_backtest(cfg, start=args.start, end=args.end, mode=args.mode)


if __name__ == "__main__":
    main()