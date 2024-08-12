
import requests
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

from alpaca.trading.requests import GetAssetsRequest, GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import ta
import pandas as pd
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  
import ta.trend
import ta.volatility
from datetime import datetime, time, timezone

# Load environment variables from .env file
load_dotenv()

class Trading:

    def __init__(self, live=False):
        


        if live:
            self.apiKey = os.getenv("API_KEY_ID")
            self.secretKey = os.getenv("API_SECRET_KEY")
            self.apiBase = "https://api.alpaca.markets"

        else:
            self.apiKey = os.getenv("PAPER_API_KEY_ID")
            self.secretKey = os.getenv("PAPER_API_SECRET_KEY")
            self.apiBase = "https://paper-api.alpaca.markets"

        
        self.trading_client = TradingClient(self.apiKey, self.secretKey, paper = not live)

        self.historical_client = StockHistoricalDataClient(self.apiKey, self.secretKey)


    
        # Set up the headers with your API keys
        self.headers = {
            "APCA-API-KEY-ID": self.apiKey ,
            "APCA-API-SECRET-KEY": self.secretKey
        }
        



    def requestAccount(self):
        accountUrl = f"{self.apiBase}/v2/account"
        # Make a GET request to the account endpoint
        response = requests.get(accountUrl, headers=self.headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            account_data = response.json()
            print("Account Information:")
            print(account_data)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    def getBars(self, tickers):
        ur = f"{self.apiBase}v2/stocks/bars"

    def getBalanceChange(self):
        account = self.trading_client.get_account()

        # Check our current balance vs. our balance at the last market close
        balance_change = float(account.equity) - float(account.last_equity)
        print(f'Today\'s portfolio balance change: ${balance_change}')

    def checkTradable(self, asset):
        found_asset = self.trading_client.get_asset(asset)

        if found_asset.tradable:
            print(f'We can trade {asset}.')
            return True

        return False

    def setMarketOrder(self, ticker, quantity):
        if self.checkTradable(ticker):
            market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=quantity,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                        )

            # Market order
            market_order = self.trading_client.submit_order(
                        order_data=market_order_data
                        )

            print(market_order)

        

    def getOrders(self):
        get_orders_data = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        limit=5,
        nested=True  # show nested multi-leg orders
    )

        orders = self.trading_client.get_orders(filter=get_orders_data)
        print(f"Orders: {orders}")

    
    def test(self):
        request = StockBarsRequest(
            symbol_or_symbols="AAPL",  # Single symbol or list of symbols
            timeframe=TimeFrame.Day,   # TimeFrame options: Minute, Hour, Day, Week, Month
            start="2023-02-14",        # Start date for the data
            end="2023-08-01"           # End date for the data
        )


        bars = self.historical_client.get_stock_bars(request)
        data = []
        for bar in bars['AAPL']:
            data.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            })

        data = pd.DataFrame(data)

        data['Bollinger_High'] = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2).bollinger_hband()
        data['Bollinger_Med'] = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2).bollinger_mavg()
        data['Bollinger_Low'] = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2).bollinger_lband()
        data["SMA"] = ta.trend.SMAIndicator(data['close'], window=14).sma_indicator()
        
        with open('out.txt', 'w') as file:
            print(data, file = file)

        print(data)

    def getStockData(self, ticker):
        requestDaily = StockBarsRequest(
            symbol_or_symbols=ticker,  
            timeframe=TimeFrame.Day,   
            start=datetime.now()-datetime.timedelta(days=14, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0),        
            end=datetime.now()         
        )

        barsDaily = self.historical_client.get_stock_bars(requestDaily)

        dataDaily = []
        for bar in barsDaily[ticker]:
            dataDaily.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            })


        dataDaily = pd.DataFrame(dataDaily)
        dataDaily['Bollinger_High'] = ta.volatility.BollingerBands(close=dataDaily['close'], window=20, window_dev=2).bollinger_hband()
        dataDaily['Bollinger_Med'] = ta.volatility.BollingerBands(close=dataDaily['close'], window=20, window_dev=2).bollinger_mavg()
        dataDaily['Bollinger_Low'] = ta.volatility.BollingerBands(close=dataDaily['close'], window=20, window_dev=2).bollinger_lband()
        dataDaily["SMA"] = ta.trend.SMAIndicator(dataDaily['close'], window=14).sma_indicator()

        return dataDaily
    
    def getBollinger(self, ticker):
        request = StockBarsRequest(
        symbol_or_symbols=ticker,  
        timeframe=TimeFrame.Hour,   
        start=datetime.now()-datetime.timedelta(days=14, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0),        
        end=datetime.now()         
        )

        barsDaily = self.historical_client.get_stock_bars(request)

        dataHourly = []
        for bar in barsDaily[ticker]:
            dataHourly.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            })


        dataHourly = pd.DataFrame(dataHourly)
        dataHourly['Bollinger_High'] = ta.volatility.BollingerBands(close=dataHourly['close'], window=20, window_dev=2).bollinger_hband()
        dataHourly['Bollinger_Med'] = ta.volatility.BollingerBands(close=dataHourly['close'], window=20, window_dev=2).bollinger_mavg()
        dataHourly['Bollinger_Low'] = ta.volatility.BollingerBands(close=dataHourly['close'], window=20, window_dev=2).bollinger_lband()

        
        return dataHourly


    def start(self):
        tickers = ['AAPL', 'XOM', 'CVX', 'JNJ', 'PFE', 'JPM', 'GS', 'NVDA']

        while True:
            time.sleep(10)






if __name__ == "__main__":
    trading = Trading()
    #trading.requestAccount()
    #trading.setMarketOrder("AAPL", 1)
    #trading.getOrders()
   # trading.getBalanceChange()

    trading.test()
