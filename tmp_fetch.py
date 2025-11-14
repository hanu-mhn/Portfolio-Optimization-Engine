from src.api import _fetch_prices_yfinance
p, e = _fetch_prices_yfinance(['INFY.NS','TCS.NS','WIPRO.NS'], '2020-01-01', '2024-01-01')
print('ERRS:', e)
print('PRICES SHAPE:', None if p is None else p.shape)
print(p.head())
