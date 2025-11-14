import numpy as np


def portfolio_return(weights, mu):
    """Return expected portfolio return (scalar).

    weights: 1d array-like, shape (n,)
    mu: 1d array-like of expected returns, shape (n,)
    """
    w = np.asarray(weights)
    mu = np.asarray(mu)
    return float(w.dot(mu))


def portfolio_volatility(weights, cov):
    """Return portfolio volatility (std dev).

    weights: 1d array-like
    cov: 2d array-like covariance matrix
    """
    w = np.asarray(weights)
    cov = np.asarray(cov)
    var = float(w.T.dot(cov).dot(w))
    return float(np.sqrt(max(var, 0.0)))


def sharpe_ratio(weights, mu, cov, risk_free=0.0):
    """Return Sharpe ratio = (portfolio return - rf) / volatility."""
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return np.nan
    return (ret - risk_free) / vol
