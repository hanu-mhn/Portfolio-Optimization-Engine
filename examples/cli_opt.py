"""Simple CLI to run an optimizer using `src.api.run_optimization`.

Example:
  python examples/cli_opt.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2023-01-01 --method max_sharpe
"""
import os
import sys
# Ensure project root on path so `from src import ...` works when running the CLI directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
from pprint import pprint

from src.api import run_optimization


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", help="List of tickers (space separated)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--method", required=True, help="Optimization method (e.g. max_sharpe, min_variance, cvar, risk_parity, tracking_error, information_ratio, kelly, sortino, omega, min_max_drawdown, mvo_frontier)")
    p.add_argument("--risk-free", type=float, default=0.0)
    p.add_argument("--mar", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.95, help="Confidence level for CVaR")
    p.add_argument("--benchmark", type=str, default=None, help="Benchmark ticker (must be one of tickers)")
    p.add_argument("--long-only", dest="long_only", action="store_true", help="Force long-only (default)")
    p.add_argument("--long-short", dest="long_only", action="store_false", help="Allow long-short positions")
    p.add_argument("--min-weight", type=float, default=None)
    p.add_argument("--max-weight", type=float, default=None)
    p.add_argument("--no-sum-to-one", dest="sum_to_one", action="store_false", help="Do not enforce sum(weights)=1 (not yet supported)")
    p.add_argument("--n-frontier", type=int, default=50)
    p.set_defaults(long_only=True, sum_to_one=True)
    return p.parse_args()


def main():
    args = parse_args()
    params = dict(
        method=args.method,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        risk_free=args.risk_free,
        mar=args.mar,
        alpha=args.alpha,
        benchmark=args.benchmark,
        long_only=args.long_only,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        sum_to_one=args.sum_to_one,
        n_frontier=args.n_frontier,
    )
    result = run_optimization(**params)
    # Pretty print results
    if result.get("method") == "efficient_frontier":
        print("Efficient frontier computed:")
        print(f"Returns shape: {result['target_returns'].shape}")
        print(f"Vols shape: {result['vols'].shape}")
        print(f"Weights matrix shape: {result['weights'].shape}")
    else:
        print(json.dumps({"method": result.get("method"), "tickers": result.get("tickers"), "weights": [float(x) for x in result.get("weights")]}, indent=2))


if __name__ == "__main__":
    main()
