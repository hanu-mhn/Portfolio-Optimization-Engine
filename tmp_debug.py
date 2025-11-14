import traceback
import yfinance as yf

try:
    raw = yf.download('INFY.NS', start='2020-01-01', end='2024-01-01', progress=False, threads=False, auto_adjust=False)
    print(type(raw))
    print(raw.head())
except Exception as e:
    print('EXC:', repr(e))
    traceback.print_exc()
