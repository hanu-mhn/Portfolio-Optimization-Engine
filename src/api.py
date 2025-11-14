"""High-level API to run portfolio optimizations from user inputs.

This module exposes `run_optimization` which accepts either a pre-loaded
prices DataFrame or tickers + date range. It validates inputs and calls the
optimizers in `src.optimizers`.

Notes/assumptions:
- If `prices` is not provided, this will try to import `yfinance` to fetch
  adjusted close prices. If `yfinance` is not installed, an informative
  error is raised and the user should provide a `prices` DataFrame.
- Many optimizers expect weights summing to 1. The `sum_to_one` toggle is
  accepted but currently only honored when True; passing False will raise
  NotImplementedError (safe default). Support can be extended later.
"""
from typing import List, Optional, Sequence, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .data import compute_returns
from .optimizer import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    cvar_minimization,
    risk_parity_weights,
    tracking_error_minimization,
    information_ratio_maximization,
    kelly_weights,
    sortino_maximization,
    omega_maximization,
    min_max_drawdown,
)


def _fetch_prices_yfinance(tickers: Sequence[str], start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Fetch adjusted close prices per ticker using yfinance.

    This function downloads each ticker individually to (a) provide
    per-ticker error diagnostics and (b) be robust to yfinance returning
    different column shapes (MultiIndex vs flat). It returns a tuple
    (prices_df, errors) where `errors` maps ticker -> error message for
    tickers that failed to fetch. The caller may choose to continue with
    partially available data.
    """
    try:
        import yfinance as yf
    except Exception:
        raise RuntimeError("yfinance is required to fetch prices by ticker; please install it or pass a prices DataFrame")

    import traceback as _traceback
    frames: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}
    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=end, progress=False, threads=False, auto_adjust=False)
            if raw is None or raw.empty:
                errors[str(t)] = "no data returned"
                continue
            # raw may be a Series (single column) or DataFrame
            df = raw
            # try to extract adjusted close if present
            if isinstance(df, pd.DataFrame):
                if 'Adj Close' in df.columns:
                    s = df['Adj Close']
                elif 'Adj_Close' in df.columns:
                    s = df['Adj_Close']
                elif 'Close' in df.columns:
                    s = df['Close']
                else:
                    # take the first numeric column as best-effort
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        s = df[numeric_cols[0]]
                    else:
                        errors[str(t)] = 'no numeric price column found'
                        continue
            elif isinstance(df, pd.Series):
                s = df
            else:
                errors[str(t)] = 'unexpected yfinance return type'
                continue

            s = s.dropna()
            if s.empty:
                errors[f"{t}"] = 'all values are NaN'
                continue
            tickname = f"{t}"
            # set the series name safely (avoid passing a string as a positional
            # mapper to Series.rename which pandas may interpret as a callable)
            s2 = s.copy()
            s2.name = tickname
            frames[tickname] = s2

        except Exception as e:
            # capture full traceback to help debugging (includes exception type/message)
            errors[f"{t}"] = _traceback.format_exc()

    if not frames:
        # return empty DataFrame and errors
        return pd.DataFrame(), errors

    prices = pd.concat(frames.values(), axis=1, join='outer')
    # ensure columns are tickers (strings)
    prices.columns = [str(c) for c in prices.columns]
    return prices.dropna(how='all'), errors


def _build_bounds(n: int, long_only: bool = True, min_weight: Optional[Sequence[float]] = None,
                  max_weight: Optional[Sequence[float]] = None):
    if min_weight is None:
        min_w = 0.0 if long_only else -1.0
        min_weight = [min_w] * n
    if max_weight is None:
        max_weight = [1.0] * n
    if np.isscalar(min_weight):
        min_weight = [float(min_weight)] * n
    if np.isscalar(max_weight):
        max_weight = [float(max_weight)] * n
    if len(min_weight) != n or len(max_weight) != n:
        raise ValueError("min_weight and max_weight must be scalars or sequences of length n")
    return tuple((float(l), float(u)) for l, u in zip(min_weight, max_weight))


def run_optimization(
    method: str,
    tickers: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prices: Optional[pd.DataFrame] = None,
    risk_free: float = 0.0,
    mar: float = 0.0,
    alpha: float = 0.95,
    benchmark: Optional[str] = None,
    long_only: bool = True,
    min_weight: Optional[Sequence[float]] = None,
    max_weight: Optional[Sequence[float]] = None,
    sum_to_one: bool = True,
    n_frontier: int = 50,
    return_figures: bool = False,
    save_plots: Optional[str] = None,
    # regularization & transaction cost (turnover) support
    l1_reg: float = 0.0,
    transaction_cost: float = 0.0,
    prev_weights: Optional[Sequence[float]] = None,
    ensure_nse_suffix: bool = False,
)-> Dict[str, Any]:
    """Run requested optimization and return results.

    Parameters mirror the user's UI inputs. Returns a dict with keys:
    - weights: ndarray of resulting weights (or matrix for frontier)
    - mu: expected returns (per asset)
    - cov: covariance matrix
    - tickers: list of assets
    - method: method name
    """
    # Acquire price data
    if prices is None:
        if tickers is None or start is None or end is None:
            raise ValueError("Either provide `prices` DataFrame or tickers + start + end")
        # Optionally ensure Indian NSE tickers use the .NS suffix
        if ensure_nse_suffix:
            tickers = [t if str(t).upper().endswith('.NS') else f"{t}.NS" for t in tickers]
        prices, download_errors = _fetch_prices_yfinance(tickers, start, end)
        if download_errors:
            # make available to callers/UI so they can warn the user
            # (we'll still continue if there is some data)
            download_errors = {k: v for k, v in download_errors.items()}
        else:
            download_errors = {}
    else:
        prices = prices.copy()

    # Ensure columns/tickers ordering
    tickers = list(prices.columns)
    # surface download errors to result later
    maybe_download_errors = locals().get('download_errors', {})

    # If fetching returned no valid tickers, fail with diagnostics (before computing returns)
    if prices is None or prices.shape[1] == 0:
        err_msg = "No tickers could be fetched from data source."
        if maybe_download_errors:
            err_msg += f" See `download_errors` for per-ticker details."
        raise ValueError(err_msg)

    # Compute simple returns for scenario-based methods
    returns = compute_returns(prices, kind="simple")
    R = returns.values
    mu = returns.mean().values
    cov = returns.cov().values

    n = len(tickers)
    bounds = _build_bounds(n, long_only=long_only, min_weight=min_weight, max_weight=max_weight)

    # Validate bounds are compatible with sum-to-one constraint
    sum_max = sum(b[1] for b in bounds)
    sum_min = sum(b[0] for b in bounds)
    if not (sum_min <= 1.0 <= sum_max):
        raise ValueError(f"Provided bounds are incompatible with sum-to-one constraint (sum of lower bounds={sum_min:.4f}, sum of upper bounds={sum_max:.4f})")

    # Basic data sanity checks
    if returns.shape[0] < 2:
        raise ValueError("Not enough return observations to run optimization; provide a longer history")

    # sum_to_one toggle: currently only supported when True
    if not sum_to_one:
        raise NotImplementedError("sum_to_one=False is not yet supported; please set sum_to_one=True")

    method = method.lower()
    result: Dict[str, Any] = {"method": method, "tickers": tickers}

    if method in ("mvo_frontier", "efficient_frontier"):
        target_returns, vols, weights = efficient_frontier(mu, cov, n_points=n_frontier, bounds=bounds,
                                                          l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        result.update({"target_returns": target_returns, "vols": vols, "weights": weights})
        # optionally compute a default selected portfolio (max Sharpe)
        sel_w = max_sharpe_weights(mu, cov, risk_free=risk_free, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        result["selected_weights"] = sel_w
        # fall through to metrics/plots generation below
        portfolio_weights = sel_w

    elif method in ("max_sharpe", "sharpe", "max-sharpe"):
        w = max_sharpe_weights(mu, cov, risk_free=risk_free, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        result.update({"weights": w, "mu": mu, "cov": cov})
        portfolio_weights = w

    elif method in ("min_variance", "min-var", "minvariance"):
        w = min_variance_weights(cov, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        result.update({"weights": w, "mu": mu, "cov": cov})
        portfolio_weights = w

    elif method in ("cvar", "cvar_min", "es"):
        w = cvar_minimization(R, alpha=alpha, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        result.update({"weights": w})
        portfolio_weights = w

    elif method in ("risk_parity", "risk-parity"):
        w = risk_parity_weights(cov, bounds=bounds)
        result.update({"weights": w})
        portfolio_weights = w

    elif method in ("tracking_error", "tracking-error"):
        if benchmark is None:
            raise ValueError("benchmark is required for tracking error minimization")
        if benchmark not in tickers:
            raise ValueError("benchmark must be one of the provided tickers")
        wb = np.zeros(n)
        wb[tickers.index(benchmark)] = 1.0
        w = tracking_error_minimization(cov, benchmark_weights=wb, mu=mu, bounds=bounds)
        result.update({"weights": w})
        portfolio_weights = w
        benchmark_weights = wb

    elif method in ("information_ratio", "information-ratio"):
        if benchmark is None:
            raise ValueError("benchmark is required for information ratio maximization")
        if benchmark not in tickers:
            raise ValueError("benchmark must be one of the provided tickers")
        wb = np.zeros(n)
        wb[tickers.index(benchmark)] = 1.0
        w = information_ratio_maximization(mu, cov, wb, bounds=bounds)
        result.update({"weights": w})
        portfolio_weights = w
        benchmark_weights = wb

    elif method in ("kelly", "kelly_weights"):
        w = kelly_weights(R, bounds=bounds)
        result.update({"weights": w})
        portfolio_weights = w

    elif method in ("sortino", "sortino_max"):
        w = sortino_maximization(R, mar=mar, bounds=bounds)
        result.update({"weights": w})
        portfolio_weights = w

    elif method in ("omega", "omega_max"):
        w = omega_maximization(R, mar=mar, bounds=bounds)
        result.update({"weights": w})
        portfolio_weights = w

    elif method in ("min_max_drawdown", "minmaxdd"):
        w = min_max_drawdown(R, bounds=bounds, mar=mar)
        result.update({"weights": w})
        portfolio_weights = w

    else:
        raise ValueError(f"Unknown method: {method}")

    # expose any download errors to the caller
    if maybe_download_errors:
        result['download_errors'] = maybe_download_errors

    # Metrics computation
    from . import metrics
    metrics_dict = {}
    if 'weights' in result or 'selected_weights' in result:
        w = portfolio_weights
        metrics_dict['expected_return'] = metrics.expected_return(w, mu)
        metrics_dict['volatility'] = metrics.volatility(w, cov)
        metrics_dict['sharpe'] = metrics.sharpe(w, mu, cov, risk_free)
        metrics_dict['sortino'] = metrics.sortino(w, R, mar=mar)
        metrics_dict['omega'] = metrics.omega(w, R, mar=mar)
        metrics_dict['cvar'] = metrics.cvar(w, R, alpha=alpha)
        metrics_dict['max_drawdown'] = metrics.max_drawdown(w, R)
        if 'benchmark_weights' in locals():
            metrics_dict['tracking_error'] = metrics.tracking_error(w, benchmark_weights, R)
            metrics_dict['information_ratio'] = metrics.information_ratio(w, benchmark_weights, R)
        # risk contributions
        rc, rc_pct = metrics.risk_contributions(w, cov)
        metrics_dict['risk_contributions'] = {'absolute': rc.tolist(), 'percent': rc_pct.tolist()}
    result['metrics'] = metrics_dict

    # Plots
    figs = {}
    if return_figures or save_plots is not None:
        from . import plots
        if method in ("mvo_frontier", "efficient_frontier"):
            # plot frontier and selected
            # find index of selected in generated frontier by matching vol/return closest
            sel = result.get('selected_weights')
            sel_vol = float(np.sqrt(sel.T.dot(cov).dot(sel)))
            sel_ret = float(sel.dot(mu))
            dists = (result['vols'] - sel_vol) ** 2 + (result['target_returns'] - sel_ret) ** 2
            idx = int(np.nanargmin(dists))
            fig = plots.plot_efficient_frontier(result['target_returns'], result['vols'], result['weights'], chosen_idx=idx)
            figs['efficient_frontier'] = fig
            if save_plots:
                fig.savefig(f"{save_plots}_efficient_frontier.png")
        # risk contributions plot
        if 'risk_contributions' in metrics_dict:
            fig_rc = plots.plot_risk_contributions(np.array(metrics_dict['risk_contributions']['absolute']), np.array(metrics_dict['risk_contributions']['percent']), tickers=tickers)
            figs['risk_contributions'] = fig_rc
            if save_plots:
                fig_rc.savefig(f"{save_plots}_risk_contributions.png")
        # cumulative returns
        # result['weights'] may be a 1-D vector or a 2-D frontier matrix
        # prefer `selected_weights` when available; otherwise reduce 2-D
        # weights to a single 1-D vector (pick first point or take mean)
        w_for_perf = None
        if 'selected_weights' in result and result.get('selected_weights') is not None:
            w_for_perf = np.asarray(result.get('selected_weights'))
        elif 'weights' in result and result.get('weights') is not None:
            w_raw = np.asarray(result.get('weights'))
            if w_raw.ndim == 1:
                w_for_perf = w_raw
            elif w_raw.ndim == 2:
                # If shape aligns with (n_points, n_assets) pick a sensible row
                if w_raw.shape[1] == n:
                    # rows are frontier points, columns are assets -> pick the first point
                    w_for_perf = w_raw[0]
                elif w_raw.shape[0] == n:
                    # columns are frontier points, take first column
                    w_for_perf = w_raw[:, 0]
                else:
                    # fallback: average across frontier points to create a single allocation
                    w_for_perf = w_raw.mean(axis=0)

        if w_for_perf is not None:
            # ensure correct shape
            w_for_perf = np.asarray(w_for_perf).reshape(-1)
            if w_for_perf.size != n:
                # cannot compute cumulative returns if weight length mismatches asset count
                pass
            else:
                p = R.dot(w_for_perf)
                bench = None
                if 'benchmark_weights' in locals():
                    bench = R.dot(benchmark_weights)
                fig_cr = plots.plot_cumulative_returns(p, bench)
                figs['cumulative_returns'] = fig_cr
            if save_plots:
                fig_cr.savefig(f"{save_plots}_cumulative_returns.png")

    if return_figures:
        result['figures'] = figs
    if save_plots is not None:
        result['saved_plots_base'] = save_plots

    return result
