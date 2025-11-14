"""Helpers to fetch Indian stock OHLCV + adjusted close using yfinance.

This module provides:
- `get_nifty50_tickers()` - returns a small NIFTY-like list (tickers with .NS suffix)
- `fetch_ohlcv()` - returns a dict[ticker] -> DataFrame with Open/High/Low/Close/Adj Close/Volume
- `fetch_adjclose_df()` - returns a DataFrame of adjusted close prices for all tickers (columns=ticker)

We keep this lightweight and dependency-free beyond `yfinance` which is already
in the project's requirements.
"""
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf


def get_nifty50_tickers() -> List[str]:
    """Return a small NIFTY50-like list of popular Indian tickers (Yahoo suffix .NS).

    This list is illustrative — feel free to replace with a full NIFTY50 list or
    supply your own tickers to the fetch functions.
    """
    # A compact set of common large-cap Indian tickers (Yahoo Finance uses .NS)
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "HDFC.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
        "LT.NS",
        "ITC.NS",
        "SBIN.NS",
    ]
    return tickers


def fetch_ohlcv(tickers: List[str], start: str, end: Optional[str] = None, interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV + Adj Close for each ticker and return a dict mapping ticker->DataFrame.

    Each DataFrame has columns ['Open','High','Low','Close','Adj Close','Volume'] and a DatetimeIndex.

    This function downloads each ticker in sequence to avoid multi-index parsing quirks and
    to give per-ticker DataFrames which are easy to work with.
    """
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, interval=interval, progress=False, threads=False, auto_adjust=False)
        if df.empty:
            # create an empty frame with expected columns to keep shapes predictable
            out[t] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]) 
            continue
        # ensure expected columns exist (some tickers or intervals may omit columns)
        cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        out[t] = df[cols].copy()
    return out


def fetch_adjclose_df(tickers: List[str], start: str, end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices for the provided tickers.

    Columns are tickers (e.g. 'RELIANCE.NS') and index is the date. Missing series are left as NaN.
    """
    series = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, interval=interval, progress=False, threads=False, auto_adjust=False)
        if df.empty or "Adj Close" not in df.columns:
            s = pd.Series(dtype=float)
            s.name = t
            series[t] = s
        else:
            s = df["Adj Close"].copy()
            # set series name to the ticker so concat produces nice column names
            s.name = t
            series[t] = s
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1)


__all__ = ["get_nifty50_tickers", "fetch_ohlcv", "fetch_adjclose_df"]
