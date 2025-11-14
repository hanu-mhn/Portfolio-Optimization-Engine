"""Portfolio performance metrics.

Functions compute portfolio-level metrics from scenario returns or mu/cov.
All scenario-based functions expect `returns` as a T x n numpy array of
simple returns (not log returns).
"""
from typing import Optional
import numpy as np


def expected_return(weights: np.ndarray, mu: np.ndarray, periods_per_year: int = 252) -> float:
    return float(weights.dot(mu) * periods_per_year)


def volatility(weights: np.ndarray, cov: np.ndarray, periods_per_year: int = 252) -> float:
    var = float(weights.T.dot(cov).dot(weights))
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(periods_per_year))


def sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    er = expected_return(weights, mu, periods_per_year)
    vol = volatility(weights, cov, periods_per_year)
    if vol == 0:
        return float('nan')
    return float((er - risk_free) / vol)


def portfolio_returns_from_scenarios(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    return returns.dot(weights)


def sortino(weights: np.ndarray, returns: np.ndarray, mar: float = 0.0, periods_per_year: int = 252) -> float:
    p = portfolio_returns_from_scenarios(weights, returns)
    mean_excess = float(np.mean(p - mar) * periods_per_year)
    downside = p - mar
    downside = downside[downside < 0]
    if downside.size == 0:
        return float('inf')
    dd = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year))
    if dd == 0:
        return float('inf')
    return float(mean_excess / dd)


def omega(weights: np.ndarray, returns: np.ndarray, mar: float = 0.0) -> float:
    p = portfolio_returns_from_scenarios(weights, returns)
    gains = float(np.sum(np.maximum(p - mar, 0.0)))
    losses = float(np.sum(np.maximum(mar - p, 0.0)))
    if losses == 0:
        return float('inf')
    return float(gains / losses)


def cvar(weights: np.ndarray, returns: np.ndarray, alpha: float = 0.95) -> float:
    p = portfolio_returns_from_scenarios(weights, returns)
    losses = -p
    T = len(p)
    k = max(1, int(np.ceil((1 - alpha) * T)))
    worst = np.partition(losses, -k)[-k:]
    return float(np.mean(worst))


def max_drawdown(weights: np.ndarray, returns: np.ndarray) -> float:
    p = portfolio_returns_from_scenarios(weights, returns)
    wealth = np.cumprod(1 + p)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(-np.min(dd))


def tracking_error(weights: np.ndarray, benchmark_weights: np.ndarray, returns: np.ndarray, periods_per_year: int = 252) -> float:
    p = portfolio_returns_from_scenarios(weights, returns)
    pb = portfolio_returns_from_scenarios(benchmark_weights, returns)
    te = float(np.std(p - pb) * np.sqrt(periods_per_year))
    return te


def information_ratio(weights: np.ndarray, benchmark_weights: np.ndarray, returns: np.ndarray, periods_per_year: int = 252) -> float:
    p = portfolio_returns_from_scenarios(weights, returns)
    pb = portfolio_returns_from_scenarios(benchmark_weights, returns)
    active_ret = float(np.mean(p - pb) * periods_per_year)
    te = tracking_error(weights, benchmark_weights, returns, periods_per_year)
    if te == 0:
        return float('nan')
    return float(active_ret / te)


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    w = np.asarray(weights)
    cov = np.asarray(cov)
    port_var = float(w.T.dot(cov).dot(w))
    mc = cov.dot(w)
    rc = w * mc
    # return absolute contributions and percent
    return rc, rc / port_var


def compute_all_metrics(weights: np.ndarray,
                        mu: Optional[np.ndarray],
                        cov: Optional[np.ndarray],
                        returns: Optional[np.ndarray] = None,
                        risk_free: float = 0.0,
                        mar: float = 0.0,
                        alpha: float = 0.95) -> dict:
    """Compute a standard set of metrics and return as a dict.

    This convenience function tries to compute as many metrics as possible
    from the available inputs (mu/cov and/or scenario returns).
    """
    out = {}
    w = np.asarray(weights)
    if mu is not None and cov is not None:
        out['expected_return'] = expected_return(w, np.asarray(mu))
        out['volatility'] = volatility(w, np.asarray(cov))
        out['sharpe'] = sharpe(w, np.asarray(mu), np.asarray(cov), risk_free)
        rc, rc_pct = risk_contributions(w, np.asarray(cov))
        out['risk_contributions'] = {'absolute': rc.tolist(), 'percent': rc_pct.tolist()}
    if returns is not None:
        out['sortino'] = sortino(w, returns, mar=mar)
        out['omega'] = omega(w, returns, mar=mar)
        out['cvar'] = cvar(w, returns, alpha=alpha)
        out['max_drawdown'] = max_drawdown(w, returns)
        # tracking & info ratio require a benchmark weights in caller
    return out


def compute_factor_betas(returns: np.ndarray, factor_returns: np.ndarray):
    """Compute factor betas (exposures) by regressing each asset's returns on the factor returns.

    returns: T x n array (assets)
    factor_returns: T x k array (factors)

    Returns: betas as n x k array (asset rows, factor columns)
    """
    R = np.asarray(returns)
    F = np.asarray(factor_returns)
    if R.ndim != 2:
        raise ValueError("returns must be 2D (T x n)")
    if F.ndim != 2:
        raise ValueError("factor_returns must be 2D (T x k)")
    T, n = R.shape
    Tf, k = F.shape
    if Tf != T:
        raise ValueError("returns and factor_returns must have same number of rows (time)")
    # Add intercept to factors so betas include alpha if desired
    X = np.hstack([np.ones((T, 1)), F])  # T x (k+1)
    # Solve least squares for each asset
    betas = np.linalg.lstsq(X, R, rcond=None)[0]  # (k+1) x n
    # drop intercept row and return k x n -> transpose to n x k
    betas_no_intercept = betas[1:, :].T
    return betas_no_intercept


def portfolio_factor_exposure(weights: np.ndarray, betas: np.ndarray):
    """Compute portfolio-level factor exposures given asset betas.

    weights: n-vector
    betas: n x k array (asset rows, factor columns)
    Returns: k-vector of exposures
    """
    w = np.asarray(weights)
    B = np.asarray(betas)
    if w.ndim != 1:
        w = w.ravel()
    if B.ndim != 2:
        raise ValueError("betas must be 2D (n x k)")
    if B.shape[0] != w.shape[0]:
        raise ValueError("weights length must equal number of rows in betas")
    return w.dot(B)
