Portfolio Optimization Engine
# Portfolio Optimization Engine

A compact portfolio optimization engine in Python. It includes common
portfolio construction methods (mean-variance, max Sharpe, CVaR minimization,
risk parity, tracking error minimization), example apps, and a small test
suite.

## Highlights
- Functional and class-based optimizers in a single canonical module: `src/optimizer.py`.
- Small dependency surface: numpy, scipy, pandas; Streamlit-based demo UI.
- Examples: CLI demo (`examples/run_demo.py`) and a Streamlit app (`examples/streamlit_app.py`).

## Quick start

Create and activate a virtualenv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the CLI demo (synthetic data):

```bash
python examples/run_demo.py
```

Run the Streamlit demo:

```bash
source .venv/bin/activate
.venv/bin/streamlit run examples/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser once Streamlit reports it is running.

## Running tests

Unit tests are under `tests/`. To run tests:

```bash
source .venv/bin/activate
.venv/bin/pytest -q
```

## Important files
- `src/optimizer.py` — canonical optimizer functions and class wrappers (optimize() API)
- `src/metrics.py` — metric calculators used by optimizer wrappers
- `examples/run_demo.py` — simple command-line demonstration
- `examples/streamlit_app.py` — interactive Streamlit demo
- `tests/test_optimizer.py` — small unit tests

## Development notes
- The optimizer implementations use `scipy.optimize.minimize` (SLSQP) with box constraints by default.
- Keep imports to the single `src.optimizer` module to avoid duplication or circular imports.
- If you need convex/QP solvers or exact constraints, consider wiring in CVXPy or OSQP.

## Contributing
PRs are welcome. Suggested workflow:

1. Fork this repository.
2. Create a feature branch.
3. Run tests and add unit tests for new behavior.
4. Open a PR with a clear description and test results.

## License
This repository does not include a license file. Add one (e.g., MIT) if you want to publish or share.

---
_Generated: updated README to include install, run, and test instructions._
