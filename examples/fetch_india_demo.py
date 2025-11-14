"""Demo: fetch OHLCV + Adj Close for a few Indian tickers and print summaries.

Run with the project's venv python, for example:

"""
import sys
from pathlib import Path

# allow running the example directly without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.india_data import get_nifty50_tickers, fetch_ohlcv, fetch_adjclose_df


def main():
    tickers = get_nifty50_tickers()[:4]
    print("Fetching the following tickers:", tickers)
    ohlcv = fetch_ohlcv(tickers, start="2023-01-01", end="2024-01-01")
    for t, df in ohlcv.items():
        print(f"\n--- {t} ---")
        if df.empty:
            print("No data returned")
        else:
            print(df.head().to_string())

    # show a combined adjusted close DataFrame
    adj = fetch_adjclose_df(tickers, start="2023-01-01", end="2024-01-01")
    print("\nAdjusted close (head):")
    print(adj.head().to_string())


if __name__ == "__main__":
    main()
