"""Demo script showing optimizer usage with synthetic data."""
import numpy as np
from src.optimizer import max_sharpe_weights, min_variance_weights
from src.utils import portfolio_return, portfolio_volatility, sharpe_ratio


def make_synthetic(n=4, seed=42):
    rng = np.random.default_rng(seed)
    base_mu = np.linspace(0.06, 0.14, n)
    A = rng.normal(scale=0.1, size=(n, n))
    cov = np.dot(A, A.T) * 0.05 + np.eye(n) * 0.01
    return base_mu, cov


def pretty_print(weights, mu, cov, rf=0.0):
    print("Weights:")
    for i, w in enumerate(weights):
        print(f"  Asset {i}: {w:.4f}")
    print(f"Expected return: {portfolio_return(weights, mu):.4%}")
    print(f"Volatility: {portfolio_volatility(weights, cov):.4%}")
    print(f"Sharpe: {sharpe_ratio(weights, mu, cov, rf):.4f}")


if __name__ == "__main__":
    mu, cov = make_synthetic(n=4)
    print("=== Max Sharpe Portfolio ===")
    w_sharpe = max_sharpe_weights(mu, cov, risk_free=0.01)
    pretty_print(w_sharpe, mu, cov, rf=0.01)

    print('\n=== Min Variance (no target) ===')
    w_minvar = min_variance_weights(cov)
    pretty_print(w_minvar, mu, cov)

    print('\n=== Min Variance (target return) ===')
    target = float(np.mean(mu))
    w_target = min_variance_weights(cov, mu=mu, target_return=target)
    pretty_print(w_target, mu, cov)
