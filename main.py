import numpy as np
import pandas as pd
import requests 
import xlsxwriter 
import yfinance as yf


tickers = ['AAPL', 'MSFT']
data = yf.Tickers(tickers)

for ticker in tickers:
    info = data.tickers[ticker].info
    market_cap = info.get('marketCap', 0)
    print(market_cap)

