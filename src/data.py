import pandas as pd
import numpy as np


def load_prices_csv(path, index_col=None, date_col=None, parse_dates=True):
    """Load a CSV of prices. Returns a DataFrame indexed by date (if available).

    Parameters
    - path: str path to CSV
    - index_col: column to use as index (optional)
    - date_col: column containing date strings (optional). If provided and parse_dates=True, parse it.
    - parse_dates: bool, passed to pandas.read_csv
    """
    if date_col is not None:
        df = pd.read_csv(path, parse_dates=[date_col])
        df = df.set_index(date_col)
    else:
        df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_index()


def compute_returns(prices, kind="log"):
    """Compute returns from price DataFrame.

    Parameters
    - prices: pd.DataFrame, columns are tickers, index are dates
    - kind: 'log' or 'simple'

    Returns
    - pd.DataFrame of returns
    """
    if kind == "log":
        return np.log(prices / prices.shift(1)).dropna()
    elif kind == "simple":
        return prices.pct_change().dropna()
    else:
        raise ValueError("kind must be 'log' or 'simple'")
