Portfolio Optimization Engine

A small, self-contained portfolio optimization engine in Python.

Features
- Load price CSVs and compute returns
- Portfolio metrics: return, volatility, Sharpe
- Optimizers: minimum-variance for a target return, maximum Sharpe (no shorting by default)

Quick start

1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the demo which uses synthetic data

```bash
python examples/run_demo.py
```

Files of interest
- `src/data.py` — data loading and return calculation
- `src/utils.py` — portfolio metrics
- `src/optimizer.py` — optimization routines (SLSQP via scipy)
- `examples/run_demo.py` — simple CLI demo
- `tests/test_optimizer.py` — basic pytest tests

Notes
- This project avoids heavyweight convex solvers to keep dependencies light (uses `scipy.optimize.minimize`).
- Extend with CVXPy or add data connectors (yfinance, local CSVs) as needed.
