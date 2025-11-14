from src.api import _fetch_prices_yfinance, run_optimization

prices, errs = _fetch_prices_yfinance(['INFY.NS','TCS.NS','WIPRO.NS'], '2020-01-01', '2024-01-01')
print('errs:', errs)
print('prices shape', prices.shape)
res = run_optimization(method='mvo_frontier', prices=prices, n_frontier=50, return_figures=True)
print('result keys:', list(res.keys()))
if 'figures' in res and 'cumulative_returns' in res['figures']:
    print('cumulative fig present')
else:
    print('no cumulative fig')
