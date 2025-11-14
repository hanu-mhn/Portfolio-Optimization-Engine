"""Optimizers: functional implementations and class wrappers.

This module provides a compact, single-file implementation of portfolio
optimizers (mean-variance, CVaR, risk parity, tracking error, etc.) and
class wrappers that expose a uniform `optimize()` API returning a dict with
keys: "weights", "metrics", "diagnostics".
"""
from typing import Optional, Dict, Any, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize


def _check_inputs(n: int, *arrays):
    for a in arrays:
        if a is None:
            continue
        arr = np.asarray(a)
        if arr.ndim == 1 and arr.shape[0] != n:
            raise ValueError("Input length mismatch")
        if arr.ndim == 2 and arr.shape[1] != n:
            raise ValueError("Scenario matrix columns must equal n assets")


def _penalty_term(w: np.ndarray, l1_reg: float = 0.0, transaction_cost: float = 0.0, prev_weights: Optional[np.ndarray] = None) -> float:
    w = np.asarray(w)
    penalty = 0.0
    if l1_reg and l1_reg > 0:
        penalty += float(l1_reg * np.sum(np.abs(w)))
    if transaction_cost and transaction_cost > 0:
        if prev_weights is None:
            prev = np.zeros_like(w)
        else:
            prev = np.asarray(prev_weights)
        penalty += float(transaction_cost * np.sum(np.abs(w - prev)))
    return penalty


def min_variance_weights(cov: np.ndarray,
                         mu: Optional[Sequence[float]] = None,
                         target_return: Optional[float] = None,
                         bounds: Optional[Sequence[tuple]] = None,
                         tol: float = 1e-10,
                         l1_reg: float = 0.0,
                         transaction_cost: float = 0.0,
                         prev_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    cov = np.asarray(cov)
    n = cov.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, mu)

    def obj(w):
        base = float(w.T.dot(cov).dot(w))
        pen = _penalty_term(w, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        return base + pen

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        if mu is None:
            raise ValueError("mu (expected returns) required when target_return provided")
        cons.append({"type": "eq", "fun": lambda w: float(w.dot(np.asarray(mu)) - target_return)})

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def max_sharpe_weights(mu: Sequence[float], cov: np.ndarray, risk_free: float = 0.0,
                       bounds: Optional[Sequence[tuple]] = None, tol: float = 1e-8,
                       l1_reg: float = 0.0,
                       transaction_cost: float = 0.0,
                       prev_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = mu.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    _check_inputs(n, mu)

    def neg_sharpe(w):
        ret = float(w.dot(mu) - risk_free)
        var = float(w.T.dot(cov).dot(w))
        vol = float(np.sqrt(max(var, 0.0)))
        if vol == 0:
            return 1e6 - ret
        pen = _penalty_term(w, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        return -ret / vol + pen

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def efficient_frontier(mu: Sequence[float], cov: np.ndarray, n_points: int = 50,
                       bounds: Optional[Sequence[tuple]] = None,
                       l1_reg: float = 0.0,
                       transaction_cost: float = 0.0,
                       prev_weights: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = mu.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    ret_min, ret_max = float(mu.min()), float(mu.max())
    target_returns = np.linspace(ret_min, ret_max, n_points)
    vols = []
    weights = []
    for tr in target_returns:
        try:
            w = min_variance_weights(cov, mu=mu, target_return=float(tr), bounds=bounds,
                                     l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
            v = float(np.sqrt(max(w.T.dot(cov).dot(w), 0.0)))
        except Exception:
            w = np.full(n, np.nan)
            v = np.nan
        weights.append(w)
        vols.append(v)
    return target_returns, np.array(vols), np.vstack(weights)


def cvar_minimization(returns: np.ndarray, alpha: float = 0.95,
                      bounds: Optional[Sequence[tuple]] = None,
                      tol: float = 1e-8,
                      l1_reg: float = 0.0,
                      transaction_cost: float = 0.0,
                      prev_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def obj(w):
        p = R.dot(w)
        losses = -p
        k = max(1, int(np.ceil((1 - alpha) * T)))
        worst = np.partition(losses, -k)[-k:]
        pen = _penalty_term(w, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        return float(np.mean(worst)) + pen

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"CVaR optimization failed: {res.message}")
    return res.x


def risk_parity_weights(cov: np.ndarray, bounds: Optional[Sequence[tuple]] = None,
                        tol: float = 1e-10) -> np.ndarray:
    cov = np.asarray(cov)
    n = cov.shape[0]
    if bounds is None:
        bounds = tuple((1e-12, 1.0) for _ in range(n))

    def obj(w):
        w = np.asarray(w)
        port_var = float(w.T.dot(cov).dot(w))
        mc = cov.dot(w)
        rc = w * mc
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Risk parity optimization failed: {res.message}")
    return res.x


def tracking_error_minimization(cov: np.ndarray, benchmark_weights: Optional[np.ndarray] = None,
                                 mu: Optional[Sequence[float]] = None,
                                 target_active_return: Optional[float] = None,
                                 bounds: Optional[Sequence[tuple]] = None,
                                 tol: float = 1e-10) -> np.ndarray:
    cov = np.asarray(cov)
    n = cov.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    if benchmark_weights is None:
        benchmark_weights = np.zeros(n)
    benchmark_weights = np.asarray(benchmark_weights)
    _check_inputs(n, mu, benchmark_weights)

    def obj(w):
        a = w - benchmark_weights
        var = float(a.T.dot(cov).dot(a))
        return float(np.sqrt(max(var, 0.0)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_active_return is not None:
        if mu is None:
            raise ValueError("mu required when target_active_return provided")
        cons.append({"type": "eq", "fun": lambda w: float((w - benchmark_weights).dot(np.asarray(mu)) - target_active_return)})

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Tracking error optimization failed: {res.message}")
    return res.x


def information_ratio_maximization(mu: Sequence[float], cov: np.ndarray, benchmark_weights: np.ndarray,
                                    bounds: Optional[Sequence[tuple]] = None,
                                    tol: float = 1e-8) -> np.ndarray:
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    wb = np.asarray(benchmark_weights)
    n = mu.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, mu, wb)

    def neg_info(w):
        active_ret = float(w.dot(mu) - wb.dot(mu))
        a = w - wb
        var = float(a.T.dot(cov).dot(a))
        vol = float(np.sqrt(max(var, 0.0)))
        if vol == 0:
            return 1e6 - active_ret
        return -active_ret / vol

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_info, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Information ratio optimization failed: {res.message}")
    return res.x


def kelly_weights(returns: np.ndarray, bounds: Optional[Sequence[tuple]] = None,
                  tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def obj(w):
        p = R.dot(w)
        if np.any(1 + p <= 0):
            return 1e6 + float(np.sum(np.minimum(0, 1 + p)))
        return -float(np.mean(np.log(1 + p)))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Kelly optimization failed: {res.message}")
    return res.x


def sortino_maximization(returns: np.ndarray, mar: float = 0.0,
                         bounds: Optional[Sequence[tuple]] = None,
                         tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def neg_sortino(w):
        p = R.dot(w)
        excess = p - mar
        downside = excess[excess < 0]
        if downside.size == 0:
            return -1e6
        dd = float(np.sqrt(np.mean(downside ** 2)))
        mean_excess = float(np.mean(excess))
        if dd == 0:
            return 1e6 - mean_excess
        return -mean_excess / dd

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_sortino, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Sortino optimization failed: {res.message}")
    return res.x


def omega_maximization(returns: np.ndarray, mar: float = 0.0,
                       bounds: Optional[Sequence[tuple]] = None,
                       tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def neg_omega(w):
        p = R.dot(w)
        gains = np.sum(np.maximum(p - mar, 0.0))
        losses = np.sum(np.maximum(mar - p, 0.0))
        if losses == 0:
            return -1e6
        return -float(gains / losses)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_omega, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Omega optimization failed: {res.message}")
    return res.x


def min_max_drawdown(returns: np.ndarray, bounds: Optional[Sequence[tuple]] = None,
                     mar: Optional[float] = None, tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def max_drawdown_of_p(p):
        wealth = np.cumprod(1 + p)
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak
        return float(np.min(dd))

    def obj(w):
        p = R.dot(w)
        mdd = max_drawdown_of_p(p)
        return float(-mdd)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    if mar is not None:
        cons = list(cons)
        cons.append({"type": "ineq", "fun": lambda w: float(np.mean(R.dot(w)) - mar)})

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol,
                   options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Min max-drawdown optimization failed: {res.message}")
    return res.x


# Import metrics module lazily (module must exist in package)
from . import metrics as metrics_mod


class OptimizerBase:
    def __init__(self, mu: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None, returns: Optional[np.ndarray] = None):
        self.mu = None if mu is None else np.asarray(mu)
        self.cov = None if cov is None else np.asarray(cov)
        self.returns = None if returns is None else np.asarray(returns)

    def optimize(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


class MeanVarianceOptimizer(OptimizerBase):
    def __init__(self, mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0):
        super().__init__(mu=mu, cov=cov)
        self.risk_free = risk_free

    def optimize(self, method: str = "max_sharpe", bounds=None, n_frontier: int = 50, **kwargs):
        method = method.lower()
        diagnostics = {}
        l1_reg = kwargs.get('l1_reg', 0.0)
        transaction_cost = kwargs.get('transaction_cost', 0.0)
        prev_weights = kwargs.get('prev_weights', None)
        if method in ("max_sharpe", "sharpe"):
            w = max_sharpe_weights(self.mu, self.cov, risk_free=self.risk_free, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        elif method in ("min_variance", "min-var"):
            w = min_variance_weights(self.cov, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        elif method in ("frontier", "efficient_frontier", "mvo_frontier"):
            tr, vols, wmat = efficient_frontier(self.mu, self.cov, n_points=n_frontier, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
            diagnostics['frontier'] = {'target_returns': tr, 'vols': vols, 'weights_matrix': wmat}
            w = max_sharpe_weights(self.mu, self.cov, risk_free=self.risk_free, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        else:
            raise ValueError(f"Unknown MVO method: {method}")

        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": diagnostics}


class CvarOptimizer(OptimizerBase):
    def optimize(self, alpha: float = 0.95, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for CVaR optimization")
        w = cvar_minimization(self.returns, alpha=alpha, bounds=bounds, l1_reg=kwargs.get('l1_reg', 0.0), transaction_cost=kwargs.get('transaction_cost', 0.0), prev_weights=kwargs.get('prev_weights', None))
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, alpha=alpha)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class RiskParityOptimizer(OptimizerBase):
    def optimize(self, bounds=None, **kwargs):
        if self.cov is None:
            raise ValueError("Covariance matrix required for risk parity")
        w = risk_parity_weights(self.cov, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class TrackingErrorOptimizer(OptimizerBase):
    def optimize(self, benchmark_weights: np.ndarray, target_active_return: Optional[float] = None, bounds=None, **kwargs):
        if self.cov is None:
            raise ValueError("Cov required for tracking error minimization")
        w = tracking_error_minimization(self.cov, benchmark_weights=benchmark_weights, mu=self.mu, target_active_return=target_active_return, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class InformationRatioOptimizer(OptimizerBase):
    def optimize(self, benchmark_weights: np.ndarray, bounds=None, **kwargs):
        w = information_ratio_maximization(self.mu, self.cov, benchmark_weights, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class KellyOptimizer(OptimizerBase):
    def optimize(self, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for Kelly optimization")
        w = kelly_weights(self.returns, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class SortinoOptimizer(OptimizerBase):
    def optimize(self, mar: float = 0.0, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for Sortino optimization")
        w = sortino_maximization(self.returns, mar=mar, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, mar=mar)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class OmegaOptimizer(OptimizerBase):
    def optimize(self, mar: float = 0.0, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for Omega optimization")
        w = omega_maximization(self.returns, mar=mar, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, mar=mar)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class MinMaxDrawdownOptimizer(OptimizerBase):
    def optimize(self, mar: Optional[float] = None, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for drawdown optimization")
        w = min_max_drawdown(self.returns, bounds=bounds, mar=mar)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, mar=mar)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


__all__ = [
    "MeanVarianceOptimizer",
    "CvarOptimizer",
    "RiskParityOptimizer",
    "TrackingErrorOptimizer",
    "InformationRatioOptimizer",
    "KellyOptimizer",
    "SortinoOptimizer",
    "OmegaOptimizer",
    "MinMaxDrawdownOptimizer",
    # keep functional helpers available as well
    "min_variance_weights",
    "max_sharpe_weights",
    "efficient_frontier",
    "cvar_minimization",
    "risk_parity_weights",
    "tracking_error_minimization",
    "information_ratio_maximization",
    "kelly_weights",
    "sortino_maximization",
    "omega_maximization",
    "min_max_drawdown",
]
"""Functional optimizers and class-based optimizer wrappers.

This module consolidates the optimizer implementations and class
wrappers into a single file. It provides the functional implementations
and the class wrappers that expose an `optimize()` method returning a
uniform result (weights, metrics, diagnostics).
"""
from typing import Optional, Dict, Any, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize


# Basic input checks
def _check_inputs(n: int, *arrays):
    for a in arrays:
        if a is None:
            continue
        arr = np.asarray(a)
        if arr.ndim == 1 and arr.shape[0] != n:
            raise ValueError("Input length mismatch")
        if arr.ndim == 2 and arr.shape[1] != n:
            raise ValueError("Scenario matrix columns must equal n assets")


# Penalty for regularization / transaction costs
def _penalty_term(w: np.ndarray, l1_reg: float = 0.0, transaction_cost: float = 0.0, prev_weights: Optional[np.ndarray] = None) -> float:
    w = np.asarray(w)
    penalty = 0.0
    if l1_reg and l1_reg > 0:
        penalty += float(l1_reg * np.sum(np.abs(w)))
    if transaction_cost and transaction_cost > 0:
        if prev_weights is None:
            prev = np.zeros_like(w)
        else:
            prev = np.asarray(prev_weights)
        penalty += float(transaction_cost * np.sum(np.abs(w - prev)))
    return penalty


def min_variance_weights(cov: np.ndarray,
                         mu: Optional[Sequence[float]] = None,
                         target_return: Optional[float] = None,
                         bounds: Optional[Sequence[tuple]] = None,
                         tol: float = 1e-10,
                         l1_reg: float = 0.0,
                         transaction_cost: float = 0.0,
                         prev_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    cov = np.asarray(cov)
    n = cov.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, mu)

    def obj(w):
        base = float(w.T.dot(cov).dot(w))
        pen = _penalty_term(w, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        return base + pen

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        if mu is None:
            raise ValueError("mu (expected returns) required when target_return provided")
        cons.append({"type": "eq", "fun": lambda w: float(w.dot(np.asarray(mu)) - target_return)})

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def max_sharpe_weights(mu: Sequence[float], cov: np.ndarray, risk_free: float = 0.0,
                       bounds: Optional[Sequence[tuple]] = None, tol: float = 1e-8,
                       l1_reg: float = 0.0,
                       transaction_cost: float = 0.0,
                       prev_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = mu.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    _check_inputs(n, mu)

    def neg_sharpe(w):
        ret = float(w.dot(mu) - risk_free)
        var = float(w.T.dot(cov).dot(w))
        vol = float(np.sqrt(max(var, 0.0)))
        if vol == 0:
            return 1e6 - ret
        pen = _penalty_term(w, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        return -ret / vol + pen

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def efficient_frontier(mu: Sequence[float], cov: np.ndarray, n_points: int = 50,
                       bounds: Optional[Sequence[tuple]] = None,
                       l1_reg: float = 0.0,
                       transaction_cost: float = 0.0,
                       prev_weights: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = mu.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    ret_min, ret_max = float(mu.min()), float(mu.max())
    target_returns = np.linspace(ret_min, ret_max, n_points)
    vols = []
    weights = []
    for tr in target_returns:
        try:
            w = min_variance_weights(cov, mu=mu, target_return=float(tr), bounds=bounds,
                                     l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
            v = float(np.sqrt(max(w.T.dot(cov).dot(w), 0.0)))
        except Exception:
            w = np.full(n, np.nan)
            v = np.nan
        weights.append(w)
        vols.append(v)
    return target_returns, np.array(vols), np.vstack(weights)


def cvar_minimization(returns: np.ndarray, alpha: float = 0.95,
                      bounds: Optional[Sequence[tuple]] = None,
                      tol: float = 1e-8,
                      l1_reg: float = 0.0,
                      transaction_cost: float = 0.0,
                      prev_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def obj(w):
        p = R.dot(w)
        losses = -p
        k = max(1, int(np.ceil((1 - alpha) * T)))
        worst = np.partition(losses, -k)[-k:]
        pen = _penalty_term(w, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        return float(np.mean(worst)) + pen

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"CVaR optimization failed: {res.message}")
    return res.x


def risk_parity_weights(cov: np.ndarray, bounds: Optional[Sequence[tuple]] = None,
                        tol: float = 1e-10) -> np.ndarray:
    cov = np.asarray(cov)
    n = cov.shape[0]
    if bounds is None:
        bounds = tuple((1e-12, 1.0) for _ in range(n))

    def obj(w):
        w = np.asarray(w)
        port_var = float(w.T.dot(cov).dot(w))
        mc = cov.dot(w)
        rc = w * mc
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Risk parity optimization failed: {res.message}")
    return res.x


def tracking_error_minimization(cov: np.ndarray, benchmark_weights: Optional[np.ndarray] = None,
                                 mu: Optional[Sequence[float]] = None,
                                 target_active_return: Optional[float] = None,
                                 bounds: Optional[Sequence[tuple]] = None,
                                 tol: float = 1e-10) -> np.ndarray:
    cov = np.asarray(cov)
    n = cov.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    if benchmark_weights is None:
        benchmark_weights = np.zeros(n)
    benchmark_weights = np.asarray(benchmark_weights)
    _check_inputs(n, mu, benchmark_weights)

    def obj(w):
        a = w - benchmark_weights
        var = float(a.T.dot(cov).dot(a))
        return float(np.sqrt(max(var, 0.0)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_active_return is not None:
        if mu is None:
            raise ValueError("mu required when target_active_return provided")
        cons.append({"type": "eq", "fun": lambda w: float((w - benchmark_weights).dot(np.asarray(mu)) - target_active_return)})

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Tracking error optimization failed: {res.message}")
    return res.x


def information_ratio_maximization(mu: Sequence[float], cov: np.ndarray, benchmark_weights: np.ndarray,
                                    bounds: Optional[Sequence[tuple]] = None,
                                    tol: float = 1e-8) -> np.ndarray:
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    wb = np.asarray(benchmark_weights)
    n = mu.shape[0]
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, mu, wb)

    def neg_info(w):
        active_ret = float(w.dot(mu) - wb.dot(mu))
        a = w - wb
        var = float(a.T.dot(cov).dot(a))
        vol = float(np.sqrt(max(var, 0.0)))
        if vol == 0:
            return 1e6 - active_ret
        return -active_ret / vol

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_info, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Information ratio optimization failed: {res.message}")
    return res.x


def kelly_weights(returns: np.ndarray, bounds: Optional[Sequence[tuple]] = None,
                  tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def obj(w):
        p = R.dot(w)
        if np.any(1 + p <= 0):
            return 1e6 + float(np.sum(np.minimum(0, 1 + p)))
        return -float(np.mean(np.log(1 + p)))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Kelly optimization failed: {res.message}")
    return res.x


def sortino_maximization(returns: np.ndarray, mar: float = 0.0,
                         bounds: Optional[Sequence[tuple]] = None,
                         tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def neg_sortino(w):
        p = R.dot(w)
        excess = p - mar
        downside = excess[excess < 0]
        if downside.size == 0:
            return -1e6
        dd = float(np.sqrt(np.mean(downside ** 2)))
        mean_excess = float(np.mean(excess))
        if dd == 0:
            return 1e6 - mean_excess
        return -mean_excess / dd

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_sortino, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Sortino optimization failed: {res.message}")
    return res.x


def omega_maximization(returns: np.ndarray, mar: float = 0.0,
                       bounds: Optional[Sequence[tuple]] = None,
                       tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def neg_omega(w):
        p = R.dot(w)
        gains = np.sum(np.maximum(p - mar, 0.0))
        losses = np.sum(np.maximum(mar - p, 0.0))
        if losses == 0:
            return -1e6
        return -float(gains / losses)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_omega, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol)
    if not res.success:
        raise RuntimeError(f"Omega optimization failed: {res.message}")
    return res.x


def min_max_drawdown(returns: np.ndarray, bounds: Optional[Sequence[tuple]] = None,
                     mar: Optional[float] = None, tol: float = 1e-8) -> np.ndarray:
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n)")
    T, n = R.shape
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    _check_inputs(n, R)

    def max_drawdown_of_p(p):
        wealth = np.cumprod(1 + p)
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak
        return float(np.min(dd))

    def obj(w):
        p = R.dot(w)
        mdd = max_drawdown_of_p(p)
        return float(-mdd)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    if mar is not None:
        cons = list(cons)
        cons.append({"type": "ineq", "fun": lambda w: float(np.mean(R.dot(w)) - mar)})

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=tol,
                   options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Min max-drawdown optimization failed: {res.message}")
    return res.x


# Import metrics module lazily (module must exist in package)
from . import metrics as metrics_mod


class OptimizerBase:
    def __init__(self, mu: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None, returns: Optional[np.ndarray] = None):
        self.mu = None if mu is None else np.asarray(mu)
        self.cov = None if cov is None else np.asarray(cov)
        self.returns = None if returns is None else np.asarray(returns)

    def optimize(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


class MeanVarianceOptimizer(OptimizerBase):
    def __init__(self, mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0):
        super().__init__(mu=mu, cov=cov)
        self.risk_free = risk_free

    def optimize(self, method: str = "max_sharpe", bounds=None, n_frontier: int = 50, **kwargs):
        method = method.lower()
        diagnostics = {}
        l1_reg = kwargs.get('l1_reg', 0.0)
        transaction_cost = kwargs.get('transaction_cost', 0.0)
        prev_weights = kwargs.get('prev_weights', None)
        if method in ("max_sharpe", "sharpe"):
            w = max_sharpe_weights(self.mu, self.cov, risk_free=self.risk_free, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        elif method in ("min_variance", "min-var"):
            w = min_variance_weights(self.cov, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        elif method in ("frontier", "efficient_frontier", "mvo_frontier"):
            tr, vols, wmat = efficient_frontier(self.mu, self.cov, n_points=n_frontier, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
            diagnostics['frontier'] = {'target_returns': tr, 'vols': vols, 'weights_matrix': wmat}
            w = max_sharpe_weights(self.mu, self.cov, risk_free=self.risk_free, bounds=bounds, l1_reg=l1_reg, transaction_cost=transaction_cost, prev_weights=prev_weights)
        else:
            raise ValueError(f"Unknown MVO method: {method}")

        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": diagnostics}


class CvarOptimizer(OptimizerBase):
    def optimize(self, alpha: float = 0.95, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for CVaR optimization")
        w = cvar_minimization(self.returns, alpha=alpha, bounds=bounds, l1_reg=kwargs.get('l1_reg', 0.0), transaction_cost=kwargs.get('transaction_cost', 0.0), prev_weights=kwargs.get('prev_weights', None))
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, alpha=alpha)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class RiskParityOptimizer(OptimizerBase):
    def optimize(self, bounds=None, **kwargs):
        if self.cov is None:
            raise ValueError("Covariance matrix required for risk parity")
        w = risk_parity_weights(self.cov, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class TrackingErrorOptimizer(OptimizerBase):
    def optimize(self, benchmark_weights: np.ndarray, target_active_return: Optional[float] = None, bounds=None, **kwargs):
        if self.cov is None:
            raise ValueError("Cov required for tracking error minimization")
        w = tracking_error_minimization(self.cov, benchmark_weights=benchmark_weights, mu=self.mu, target_active_return=target_active_return, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class InformationRatioOptimizer(OptimizerBase):
    def optimize(self, benchmark_weights: np.ndarray, bounds=None, **kwargs):
        w = information_ratio_maximization(self.mu, self.cov, benchmark_weights, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class KellyOptimizer(OptimizerBase):
    def optimize(self, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for Kelly optimization")
        w = kelly_weights(self.returns, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class SortinoOptimizer(OptimizerBase):
    def optimize(self, mar: float = 0.0, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for Sortino optimization")
        w = sortino_maximization(self.returns, mar=mar, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, mar=mar)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class OmegaOptimizer(OptimizerBase):
    def optimize(self, mar: float = 0.0, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for Omega optimization")
        w = omega_maximization(self.returns, mar=mar, bounds=bounds)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, mar=mar)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}


class MinMaxDrawdownOptimizer(OptimizerBase):
    def optimize(self, mar: Optional[float] = None, bounds=None, **kwargs):
        if self.returns is None:
            raise ValueError("Scenario returns required for drawdown optimization")
        w = min_max_drawdown(self.returns, bounds=bounds, mar=mar)
        metrics = metrics_mod.compute_all_metrics(w, self.mu, self.cov, returns=self.returns, mar=mar)
        return {"weights": w, "metrics": metrics, "diagnostics": {}}

__all__ = [
    "MeanVarianceOptimizer",
    "CvarOptimizer",
    "RiskParityOptimizer",
    "TrackingErrorOptimizer",
    "InformationRatioOptimizer",
    "KellyOptimizer",
    "SortinoOptimizer",
    "OmegaOptimizer",
    "MinMaxDrawdownOptimizer",
    # keep functional helpers available as well
    "min_variance_weights",
    "max_sharpe_weights",
    "efficient_frontier",
    "cvar_minimization",
    "risk_parity_weights",
    "tracking_error_minimization",
    "information_ratio_maximization",
    "kelly_weights",
    "sortino_maximization",
    "omega_maximization",
    "min_max_drawdown",
]