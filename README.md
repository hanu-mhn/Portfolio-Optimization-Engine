# Portfolio Optimization Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A compact portfolio optimization engine implemented in Python. It provides
common portfolio construction methods (mean-variance, max Sharpe, CVaR
minimization, risk parity, and tracking-error minimization) exposed through a
single canonical module: `src/optimizer.py`.

Includes example apps and a small unit test suite to help you get started.

## Highlights

- Single-file canonical optimizer module: `src/optimizer.py` (class-based API
  with `optimize()` contract).
- Lightweight dependencies: numpy/pandas/scipy; optional Streamlit demo.
- Examples: CLI demo (`examples/run_demo.py`) and Streamlit demo
  (`examples/streamlit_app.py`).

## Quick start (using the project `.venv`)

1. Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies. If a `requirements.txt` exists, use it; otherwise
install the common packages directly:

```bash
pip install -r requirements.txt
# or (fallback)
pip install numpy pandas scipy pytest streamlit yfinance fastapi uvicorn matplotlib
```

3. Run the CLI demo (synthetic data):

```bash
source .venv/bin/activate
.venv/bin/python examples/run_demo.py
```

4. Run the Streamlit demo (optional):

```bash
source .venv/bin/activate
.venv/bin/streamlit run examples/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser after Streamlit reports it is
running. If the app doesn't respond, re-run Streamlit in the foreground so you
can inspect logs printed to the terminal.

## Running tests

Run the unit tests with the project's Python in `.venv`:

```bash
source .venv/bin/activate
.venv/bin/python -m pytest -q
```

## Important files

- `src/optimizer.py` — canonical optimizer implementations and class wrappers
  (each class exposes an `optimize()` method that returns (weights, metrics, diagnostics)).
- `examples/run_demo.py` — CLI demo that illustrates basic usage.
- `examples/streamlit_app.py` — interactive Streamlit demo (optional).
- `tests/` — unit tests for core functionality.

## Development notes

- Optimizers use `scipy.optimize.minimize` (SLSQP) for local constrained
  optimization; results depend on starting points and constraints.
- Avoid importing optimizer symbols at package import time to prevent circular
  imports; prefer `from src import optimizer` or direct imports from
  `src.optimizer`.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository and create a feature branch.
2. Add tests for new behavior and run the test suite.
3. Open a PR with a clear description and test results.

## License

This repository is licensed under the MIT License — see the `LICENSE` file.

---