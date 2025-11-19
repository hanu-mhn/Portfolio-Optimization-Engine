"""Streamlit UI for running optimizations locally (imports run_optimization).

Run with: streamlit run examples/streamlit_app.py
"""
import os
import sys
# Ensure the project root is on sys.path so `from src import ...` works when
# running this example directly (e.g. `streamlit run examples/streamlit_app.py`).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import streamlit as st
except Exception:
    # Provide a helpful runtime message if streamlit is not installed.
    class _MissingStreamlit:
        def __getattr__(self, name):
            def _missing(*args, **kwargs):
                raise ImportError(
                    "streamlit is required to run this app; install it with "
                    "'pip install streamlit' and run via 'streamlit run examples/streamlit_app.py'"
                )
            return _missing
    st = _MissingStreamlit()
import pandas as pd
import numpy as np

import importlib.util

try:
    from src.api import run_optimization, _fetch_prices_yfinance
    from src.metrics import portfolio_returns_from_scenarios
except Exception:
    # Fallback: load modules directly from the project `src` directory by path
    api_path = os.path.join(ROOT, 'src', 'api.py')
    metrics_path = os.path.join(ROOT, 'src', 'metrics.py')
    spec = importlib.util.spec_from_file_location('src.api', api_path)
    api_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_mod)
    run_optimization = api_mod.run_optimization
    _fetch_prices_yfinance = api_mod._fetch_prices_yfinance

    spec_m = importlib.util.spec_from_file_location('src.metrics', metrics_path)
    metrics_mod = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(metrics_mod)
    portfolio_returns_from_scenarios = metrics_mod.portfolio_returns_from_scenarios


def _coerce_weights_to_1d(w_in, n_assets):
    """Coerce various weight representations into a 1-D numpy array of length n_assets.

    Handles:
    - 1-D arrays
    - 2-D frontier matrices with shape (n_points, n_assets) or (n_assets, n_points)
    - other arrays by flattening and taking the first n_assets values
    Returns None when coercion is not possible.
    """
    try:
        arr = np.asarray(w_in)
    except Exception:
        return None
    if arr is None:
        return None
    if arr.ndim == 1 and arr.size == n_assets:
        return arr.astype(float)
    if arr.ndim == 2:
        if arr.shape[1] == n_assets:
            return arr[0].astype(float)
        if arr.shape[0] == n_assets:
            return arr[:, 0].astype(float)
        try:
            avg = arr.mean(axis=0)
            if avg.size == n_assets:
                return avg.astype(float)
        except Exception:
            pass
    try:
        flat = arr.ravel().astype(float)
        if flat.size >= n_assets:
            return flat[:n_assets]
    except Exception:
        return None
    return None

st.title('Portfolio Optimization Engine')

with st.sidebar.form('inputs'):
    tickers = st.text_input('Tickers (space separated)', 'AAPL MSFT GOOGL')
    start = st.text_input('Start date', '2020-01-01')
    end = st.text_input('End date', '2023-01-01')
    method = st.selectbox('Method', ['mvo_frontier','max_sharpe','min_variance','cvar','risk_parity','tracking_error','information_ratio','kelly','sortino','omega','min_max_drawdown'])
    risk_free = st.number_input('Risk-free rate', value=0.01)
    mar = st.number_input('MAR (for Sortino/Omega)', value=0.0)
    alpha = st.number_input('CVaR confidence', value=0.95)
    long_only = st.checkbox('Long only', value=True)
    min_weight_input = st.text_input('Min weight (scalar or comma-separated per ticker)', '')
    max_weight_input = st.text_input('Max weight (scalar or comma-separated per ticker)', '')
    l1_reg = st.number_input('L1 regularization (sparsity)', value=0.0, step=0.0001, format="%f")
    transaction_cost = st.number_input('Transaction cost (per unit turnover)', value=0.0, step=0.0001, format="%f")
    use_nse = st.checkbox('Indian NSE tickers (.NS suffix)', value=True, help='If checked, tickers without .NS will get the .NS suffix appended')
    n_frontier = st.number_input('Frontier points', value=50, min_value=10)
    benchmark = st.text_input('Benchmark (ticker from list)', '')
    submit = st.form_submit_button('Run')


def _parse_bounds_input(inp: str, n: int):
    """Parse a user-provided bounds input string.

    Accepts:
    - empty string -> None
    - scalar float -> returns scalar float
    - comma-separated floats -> returns list of length n
    Raises ValueError on malformed input or length mismatch.
    """
    if inp is None:
        return None
    s = str(inp).strip()
    if s == '':
        return None
    # try scalar
    try:
        val = float(s)
        return val
    except Exception:
        pass
    # try comma-separated list
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if not parts:
        return None
    if len(parts) not in (1, n):
        raise ValueError(f"Expected {n} values or a single scalar, got {len(parts)} entries")
    try:
        vals = [float(p) for p in parts]
    except Exception:
        raise ValueError('Could not parse numeric values from bounds input')
    if len(vals) == 1:
        return vals[0]
    return vals

if submit:
    # prepare tickers list and optionally append NSE suffix
    tlist = tickers.split()
    if use_nse:
        tlist = [t if str(t).upper().endswith('.NS') else f"{t}.NS" for t in tlist]
    # normalize benchmark to same naming convention if provided
    if benchmark and use_nse:
        if not str(benchmark).upper().endswith('.NS'):
            benchmark = f"{benchmark}.NS"

    # pre-fetch prices so we can display per-ticker download errors and avoid
    # crashing the app when no tickers are available
    try:
        prices, dl_errors = _fetch_prices_yfinance(tlist, start, end)
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        prices = pd.DataFrame()
        dl_errors = {t: str(e) for t in tlist}

    if dl_errors:
        # Show per-ticker problems but continue if some tickers succeeded
        st.warning('Some tickers failed to download. See details below.')
        for tk, msg in dl_errors.items():
            st.write(f"{tk}: {msg}")

    # parse min/max weight inputs now that we know how many tickers we have
    min_weight = None
    max_weight = None
    try:
        n_t = len(tlist)
        min_weight = _parse_bounds_input(min_weight_input, n_t)
        max_weight = _parse_bounds_input(max_weight_input, n_t)
    except Exception as e:
        st.error(f"Invalid min/max weight input: {e}")
        prices = pd.DataFrame()

    # Provide a per-ticker editor for bounds so users can set Min/Max for each asset
    per_ticker_min = None
    per_ticker_max = None
    if prices is not None and not prices.empty:
        try:
            # normalize any scalar/list min/max into lists of length n_t for defaults
            def _to_list(val, n, default):
                if val is None:
                    return [default] * n
                if isinstance(val, (list, tuple)):
                    if len(val) == n:
                        return [float(v) for v in val]
                    if len(val) == 1:
                        return [float(val[0])] * n
                return [float(val)] * n

            # sensible defaults: long-only -> min 0.0, max 1.0; otherwise -1.0 to 1.0
            default_min = 0.0 if long_only else -1.0
            default_max = 1.0
            default_min_list = _to_list(min_weight, n_t, default_min)
            default_max_list = _to_list(max_weight, n_t, default_max)

            per_ticker_min = []
            per_ticker_max = []
            with st.expander('Per-asset weight constraints (min / max)', expanded=True):
                st.write('Set per-ticker minimum and maximum weight bounds. These override the global Min/Max inputs if provided.')
                # layout three columns: ticker, min, max
                cols = st.columns([3, 2, 2])
                cols[0].markdown('**Ticker**')
                cols[1].markdown('**Min weight**')
                cols[2].markdown('**Max weight**')
                for i, tk in enumerate(tlist):
                    c0, c1, c2 = st.columns([3, 2, 2])
                    c0.write(tk)
                    # use ticker-safe keys so Streamlit preserves values across interactions
                    key_min = f"per_min_{i}_{tk}"
                    key_max = f"per_max_{i}_{tk}"
                    mv = c1.number_input('', value=float(default_min_list[i]), format='%f', key=key_min)
                    Mv = c2.number_input('', value=float(default_max_list[i]), format='%f', key=key_max)
                    per_ticker_min.append(float(mv))
                    per_ticker_max.append(float(Mv))

            # validate per-ticker bounds
            for i in range(n_t):
                if per_ticker_min[i] > per_ticker_max[i]:
                    st.error(f"For ticker {tlist[i]}: min ({per_ticker_min[i]}) > max ({per_ticker_max[i]})")
                    prices = pd.DataFrame()
                    break
        except Exception as e:
            st.error(f"Error preparing per-ticker bounds editor: {e}")
            prices = pd.DataFrame()

    if prices is None or prices.empty:
        st.error('No price data could be fetched for the requested tickers/date range. Please check tickers, your network, or provide a prices DataFrame.')
        # ensure `res` exists so downstream UI code does not raise NameError
        res = {}
    else:
        with st.spinner('Running optimization...'):
            # if method requires a benchmark, validate
            if method in ("tracking_error", "tracking-error", "information_ratio", "information-ratio"):
                if not benchmark:
                    st.error('Benchmark ticker is required for tracking-error / information-ratio methods. Please enter a benchmark in the sidebar.')
                    res = {}
                elif benchmark not in tlist:
                    st.error('Benchmark must be one of the requested tickers')
                    res = {}
                else:
                    # prefer per-ticker bounds if user supplied them via the editor
                    mw = per_ticker_min if per_ticker_min is not None else min_weight
                    Mw = per_ticker_max if per_ticker_max is not None else max_weight
                    res = run_optimization(method=method, prices=prices, risk_free=risk_free, mar=mar, alpha=alpha, benchmark=benchmark, long_only=long_only, min_weight=mw, max_weight=Mw, n_frontier=n_frontier, return_figures=True, l1_reg=float(l1_reg), transaction_cost=float(transaction_cost), prev_weights=None)
            else:
                mw = per_ticker_min if per_ticker_min is not None else min_weight
                Mw = per_ticker_max if per_ticker_max is not None else max_weight
                res = run_optimization(method=method, prices=prices, risk_free=risk_free, mar=mar, alpha=alpha, long_only=long_only, min_weight=mw, max_weight=Mw, n_frontier=n_frontier, return_figures=True, l1_reg=float(l1_reg), transaction_cost=float(transaction_cost), prev_weights=None)
            # persist latest results and prices so other UI actions (backtest, exposures)
            # can access them across Streamlit reruns triggered by button clicks.
            try:
                st.session_state['last_res'] = res
                st.session_state['last_prices'] = prices
            except Exception:
                # session_state may not be available in some import-fallback contexts
                pass
    st.header('Weights')
    # prefer selected_weights (single portfolio) over a full frontier matrix
    n_assets = len(res.get('tickers', []))
    w = None
    if res.get('selected_weights') is not None:
        w = _coerce_weights_to_1d(res.get('selected_weights'), n_assets)
    elif res.get('weights') is not None:
        w = _coerce_weights_to_1d(res.get('weights'), n_assets)

    if w is not None and w.size == n_assets:
        df = pd.DataFrame({'ticker': res['tickers'], 'weight': np.round(np.asarray(w), 6)})
        st.table(df)
    else:
        st.info('No single portfolio weights available to display.')
    st.header('Metrics')
    # render metrics in tabular form (summary metrics + risk contributions)
    metrics = res.get('metrics', {}) or {}
    import pandas as _pd
    if metrics:
        scalar_items = []
        for k, v in metrics.items():
            if k == 'risk_contributions':
                continue
            try:
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    scalar_items.append((k, float(v[0])))
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    scalar_items.append((k, float(v)))
                else:
                    scalar_items.append((k, str(v)))
            except Exception:
                scalar_items.append((k, str(v)))
        if scalar_items:
            sm = _pd.DataFrame(scalar_items, columns=['metric', 'value']).set_index('metric')
            st.table(sm)

        rc = metrics.get('risk_contributions')
        if isinstance(rc, dict):
            abs_vals = rc.get('absolute') or []
            pct_vals = rc.get('percent') or []
            if abs_vals:
                rc_df = _pd.DataFrame({'ticker': res.get('tickers', []), 'absolute': abs_vals})
                if pct_vals and len(pct_vals) == len(abs_vals):
                    rc_df['percent'] = pct_vals
                st.subheader('Risk contributions')
                st.dataframe(rc_df)
    else:
        st.info('No metrics available')
    st.header('Charts')
    figs = res.get('figures', {})
    if figs.get('cumulative_returns') is not None:
        st.pyplot(figs['cumulative_returns'])
    if figs.get('efficient_frontier') is not None:
        st.pyplot(figs['efficient_frontier'])
    if figs.get('risk_contributions') is not None:
        st.pyplot(figs['risk_contributions'])

    # Tabs for additional features: backtest, factor exposures, downloads
    st.header('Additional tools')
    tab1, tab2, tab3 = st.tabs(['Multi-period Backtest', 'Factor Exposure', 'Download'])

    with tab1:
        st.subheader('Rolling rebalancing backtest')
        lookback_days = st.number_input('Lookback (days) for estimation', value=126)
        rebalance_days = st.number_input('Rebalance every N days', value=21)
        if st.button('Run backtest'):
            with st.spinner('Running backtest...'):
                # perform simple rolling backtest: estimate mu/cov on lookback and optimize at each rebalance
                # prefer cached prices stored in session_state (set when optimization was run)
                prices = None
                if 'last_prices' in st.session_state:
                    prices = st.session_state.get('last_prices')
                else:
                    prices = res.get('prices') if res.get('prices') is not None else None
                # If run_optimization returned no prices, fetch again
                if prices is None:
                    try:
                        from src.api import _fetch_prices_yfinance
                        prices, backtest_dl_errors = _fetch_prices_yfinance(tlist, start, end)
                        if backtest_dl_errors:
                            st.warning('Some tickers failed to download for backtest:')
                            for tk, msg in backtest_dl_errors.items():
                                st.write(f"{tk}: {msg}")
                    except Exception as e:
                        st.error(f"Unable to fetch prices for backtest: {e}")
                        prices = None
                if prices is None or prices.empty:
                    st.info('Not enough price history available for backtest. Run optimization first or expand the date range.')
                if prices is not None and not prices.empty:
                    # compute returns DataFrame
                    from src.data import compute_returns
                    price_series = prices.copy()
                    returns_df = compute_returns(price_series, kind='simple')
                    dates = returns_df.index
                    n = len(tlist)

                    # Now run the rolling rebalancing backtest using the historical windows
                    try:
                        # initialize backtest state
                        n = len(tlist)
                        prev_w = np.zeros(n)
                        port_values = []
                        port_dates = []
                        current_value = 1.0

                        # iterate over rebalance dates: estimate on lookback window and apply until next rebalance
                        for i in range(lookback_days, len(dates), rebalance_days):
                            train = returns_df.iloc[i - lookback_days:i]
                            est_prices = price_series.loc[train.index]
                            # call run_optimization using historical window; pass prev_w for turnover-aware penalties
                            try:
                                out = run_optimization(method=method, prices=est_prices, risk_free=risk_free, mar=mar, alpha=alpha, long_only=long_only, min_weight=min_weight, max_weight=max_weight, n_frontier=n_frontier, l1_reg=float(l1_reg), transaction_cost=float(transaction_cost), prev_weights=prev_w)
                            except Exception as e:
                                st.error(f"Optimization failed during backtest at index {i}: {e}")
                                break

                            # coerce weights to 1-D for simulation
                            w = None
                            if out.get('selected_weights') is not None:
                                w = _coerce_weights_to_1d(out.get('selected_weights'), n)
                            elif out.get('weights') is not None:
                                w = _coerce_weights_to_1d(out.get('weights'), n)
                            if w is None:
                                st.warning(f"Optimizer returned no usable single portfolio at rebalance {i}; stopping backtest.")
                                break

                            # apply weights until next rebalance or end
                            start_idx = i
                            end_idx = min(i + rebalance_days, len(dates))
                            window_returns = returns_df.iloc[start_idx:end_idx].values

                            # compute transaction cost based on turnover at rebalance
                            turnover = float(np.sum(np.abs(w - prev_w)))
                            tc = float(transaction_cost) * turnover

                            # compute portfolio returns for the window and account for transaction cost on first period
                            pouts = window_returns.dot(w)
                            if pouts.size > 0:
                                pouts[0] = pouts[0] - tc

                            for rr, d in zip(pouts, returns_df.index[start_idx:end_idx]):
                                current_value *= (1 + float(rr))
                                port_values.append(current_value)
                                port_dates.append(d)

                            prev_w = np.array(w)

                        import pandas as _pd
                        if port_values:
                            ser = _pd.Series(port_values, index=port_dates)
                            st.line_chart(ser)
                        else:
                            st.info('Not enough data to run backtest with given lookback/rebalance settings')
                    except Exception as e:
                        st.error(f'Error running backtest: {e}')

    with tab2:
        st.subheader('Factor exposure report')
        st.markdown('Upload a CSV of factor returns (index=Date) with columns as factor names, or leave blank to skip.')
        f = st.file_uploader('Factor returns CSV', type=['csv'])
        if f is not None and st.button('Compute exposures'):
            import pandas as _pd
            fac_df = _pd.read_csv(f, parse_dates=True, index_col=0)
            # align dates with asset returns
            try:
                from src.data import compute_returns
                prices = res.get('prices')
                if prices is None:
                    try:
                        prices, fac_dl_errors = _fetch_prices_yfinance(tlist, start, end)
                        if fac_dl_errors:
                            st.warning('Some tickers failed to download for factor exposure:')
                            for tk, msg in fac_dl_errors.items():
                                st.write(f"{tk}: {msg}")
                    except Exception as e:
                        st.error(f"Error fetching prices for factor exposure: {e}")
                        prices = None
                r_df = compute_returns(prices, kind='simple')
                # align on dates intersection
                common = r_df.index.intersection(fac_df.index)
                if len(common) == 0:
                    st.error('No overlapping dates between asset returns and factor returns')
                else:
                    from src import metrics as metrics_mod
                    betas = metrics_mod.compute_factor_betas(r_df.loc[common].values, fac_df.loc[common].values)
                    # coerce portfolio weights to 1-D before computing exposures
                    n_assets = len(res.get('tickers', []))
                    w_for_exposure = None
                    if res.get('selected_weights') is not None:
                        w_for_exposure = _coerce_weights_to_1d(res.get('selected_weights'), n_assets)
                    elif res.get('weights') is not None:
                        w_for_exposure = _coerce_weights_to_1d(res.get('weights'), n_assets)
                    if w_for_exposure is None:
                        st.error('Cannot compute portfolio factor exposures: no single portfolio weights available')
                    else:
                        exposures = metrics_mod.portfolio_factor_exposure(np.asarray(w_for_exposure), betas)
                    import pandas as _pd
                    bet_df = _pd.DataFrame(betas, index=res.get('tickers'), columns=fac_df.columns)
                    st.write('Asset factor betas')
                    st.dataframe(bet_df)
                    st.write('Portfolio factor exposures')
                    st.write(dict(zip(fac_df.columns, exposures.tolist())))
            except Exception as e:
                st.error(f"Error computing factor exposures: {e}")

    with tab3:
        st.subheader('Download results')
        import io
        import pandas as _pd
        weights = None
        n_assets = len(res.get('tickers', []))
        if res.get('selected_weights') is not None:
            weights = _coerce_weights_to_1d(res.get('selected_weights'), n_assets)
        elif res.get('weights') is not None:
            weights = _coerce_weights_to_1d(res.get('weights'), n_assets)
        if weights is not None:
            dfw = _pd.DataFrame({'ticker': res['tickers'], 'weight': np.round(np.array(weights), 6)})
            csv = dfw.to_csv(index=False).encode('utf-8')
            st.download_button('Download weights CSV', data=csv, file_name='weights.csv', mime='text/csv')
            # simple PDF report via matplotlib
            if st.button('Download PDF report'):
                import matplotlib.pyplot as plt
                from io import BytesIO
                fig = plt.figure(figsize=(6, 8))
                txt = fig.text(0.01, 0.99, f"Portfolio optimization report\nMethod: {method}\n", va='top')
                metrics_text = str(res.get('metrics', {}))
                fig.text(0.01, 0.8, metrics_text, va='top', wrap=True)
                buf = BytesIO()
                fig.savefig(buf, format='pdf', bbox_inches='tight')
                buf.seek(0)
                st.download_button('Download PDF', data=buf, file_name='report.pdf', mime='application/pdf')
                plt.close(fig)
