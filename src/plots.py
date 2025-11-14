"""Simple plotting utilities for portfolio outputs.

Produces matplotlib Figure objects for the efficient frontier, risk
contributions and cumulative returns.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_efficient_frontier(target_returns: np.ndarray, vols: np.ndarray, weights: np.ndarray,
                            chosen_idx: Optional[int] = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(vols, target_returns, '-o', label='Efficient frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected return')
    if chosen_idx is not None:
        ax.scatter([vols[chosen_idx]], [target_returns[chosen_idx]], color='red', zorder=5, label='Selected')
    ax.legend()
    ax.grid(True)
    return fig


def plot_risk_contributions(rc: np.ndarray, rc_pct: np.ndarray, tickers: Optional[list] = None) -> plt.Figure:
    n = len(rc)
    labels = tickers if tickers is not None else [f'Asset {i}' for i in range(n)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, rc)
    axes[0].set_title('Risk contributions')
    axes[0].set_ylabel('Contribution')
    axes[1].bar(labels, rc_pct)
    axes[1].set_title('Risk contribution (%)')
    axes[1].set_ylabel('Percent')
    plt.tight_layout()
    return fig


def plot_cumulative_returns(portfolio_returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> plt.Figure:
    wealth = np.cumprod(1 + portfolio_returns)
    fig, ax = plt.subplots()
    ax.plot(wealth, label='Portfolio')
    if benchmark_returns is not None:
        ax.plot(np.cumprod(1 + benchmark_returns), label='Benchmark')
    ax.set_title('Cumulative returns')
    ax.set_ylabel('Wealth')
    ax.set_xlabel('Period')
    ax.legend()
    ax.grid(True)
    return fig
