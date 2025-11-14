import numpy as np
from src.optimizer import max_sharpe_weights, min_variance_weights


def test_max_sharpe_prefers_high_mu():
    # Identity covariance with small variance -> allocate to highest mu
    mu = np.array([0.05, 0.10, 0.01])
    cov = np.eye(3) * 0.01
    w = max_sharpe_weights(mu, cov, risk_free=0.0)
    # Best asset is index 1 with mu=0.10; optimizer should put the largest weight
    best_idx = int(np.argmax(mu))
    assert int(np.argmax(w)) == best_idx


def test_min_variance_equal_weights_for_identity_cov():
    cov = np.eye(4) * 0.02
    w = min_variance_weights(cov)
    # Expect roughly equal weights
    assert np.allclose(w, np.ones(4) / 4, atol=1e-6)
