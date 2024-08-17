
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
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)        # Auto-adjust the width based on the content
pd.set_option('display.max_colwidth', None) # Show full content of each column
import ta.momentum
import ta.trend
import ta.volatility
from datetime import datetime, time, timezone, timedelta

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



    # def getStockData(self, ticker):
    #     requestDaily = StockBarsRequest(
    #         symbol_or_symbols=ticker,  
    #         timeframe=TimeFrame.Day,   
    #         start=datetime.now()-timedelta(days=14, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0),        
    #         end=datetime.now()         
    #     )

    #     barsDaily = self.historical_client.get_stock_bars(requestDaily)

    #     dataDaily = []
    #     for bar in barsDaily[ticker]:
    #         dataDaily.append({
    #             'timestamp': bar.timestamp,
    #             'open': bar.open,
    #             'high': bar.high,
    #             'low': bar.low,
    #             'close': bar.close,
    #             'volume': bar.volume,
    #         })


    #     dataDaily = pd.DataFrame(dataDaily)
    #     dataDaily['Bollinger_High'] = ta.volatility.BollingerBands(close=dataDaily['close'], window=20, window_dev=2).bollinger_hband()
    #     dataDaily['Bollinger_Med'] = ta.volatility.BollingerBands(close=dataDaily['close'], window=20, window_dev=2).bollinger_mavg()
    #     dataDaily['Bollinger_Low'] = ta.volatility.BollingerBands(close=dataDaily['close'], window=20, window_dev=2).bollinger_lband()
    #     dataDaily["SMA"] = ta.trend.SMAIndicator(dataDaily['close'], window=14).sma_indicator()

    #     return dataDaily
    
    def getBollinger(self, ticker):
        offset = 240
        request = StockBarsRequest(
        symbol_or_symbols=ticker,  
        timeframe=TimeFrame.Hour,   
        start=datetime.now()-timedelta(days=365) -timedelta(minutes=offset),        
        end=datetime.now() -timedelta(minutes=offset)        
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
        bollinger = ta.volatility.BollingerBands(close=dataHourly['close'], window=20, window_dev=2)

        #Calculate Bolllinger Bands for each hour
        dataHourly['Bollinger_High'] = bollinger.bollinger_hband()
        dataHourly['Bollinger_Med'] = bollinger.bollinger_mavg()
        dataHourly['Bollinger_Low'] = bollinger.bollinger_lband()
        dataHourly["SMA200"] = ta.trend.SMAIndicator(dataHourly['close'], window=2000).sma_indicator()
        dataHourly["RSI"] = ta.momentum.RSIIndicator(dataHourly['close'], window=200).rsi()

        #dataHourly["MACD"] = ta.

        #Drop NaNs created from not enough data to calc bollinger band
        dataHourly.dropna(inplace=True) 

        def identify_market_regime(row):
            if row['close'] > row['SMA200']: #and row['RSI'] > 50:  #row['close'] > row['200DMA'] and row['RSI'] > 50 and row['MACD'] > row['Signal Line']:
                return 'Uptrend'
            elif row['close'] < row['SMA200']: #and row['RSI'] < 50:   ##row['close'] < row['200DMA'] and row['RSI'] < 50 and row['MACD'] < row['Signal Line']:
                return 'Downtrend'
            else:
                return 'Range'
            
        dataHourly['Regime'] = dataHourly.apply(identify_market_regime, axis=1)

        dataHourly['timestamp'] = pd.to_datetime(dataHourly['timestamp'])
        dataHourly = dataHourly.set_index('timestamp', drop=False)

        # Calculate the daily market regime by taking the mode for each day
        def daily_market_regime(group):
            return group['Regime'].mode().iloc[0] if not group['Regime'].mode().empty else 'Unknown'

        # Group by day and calculate daily regime
        daily_regimes = dataHourly.groupby(dataHourly.index.date).apply(daily_market_regime).reset_index()
        daily_regimes.columns = ['Date', 'Daily_Regime']

        # Merge daily regimes back into hourly data
        dataHourly['Date'] = dataHourly.index.date
        dataHourly = dataHourly.merge(daily_regimes, left_on='Date', right_on='Date', how='left')

        # Clean up DataFrame
        dataHourly.drop(columns=['Date'], inplace=True)
        
        #Create buy/sell signals for each hour
        #shift by one so no lookahead bias
        dataHourly['Buy_Signal'] = (dataHourly['close'] < dataHourly['Bollinger_Low']) & (dataHourly['Daily_Regime']=='Uptrend')
        dataHourly['Sell_Signal'] = (dataHourly['close'] > dataHourly['Bollinger_High']) & (dataHourly['Daily_Regime']=='Downtrend')

        dataHourly['Buy_Signal_Bollinger'] = dataHourly['close'] < dataHourly['Bollinger_Low']
        dataHourly['Buy_Signal_Regime'] = dataHourly['Daily_Regime']=='Uptrend'

        dataHourly['Sell_Signal_Bollinger'] = (dataHourly['close'] > dataHourly['Bollinger_High'])
        dataHourly['Sell_Signal_Regime'] = (dataHourly['Daily_Regime']=='Downtrend')


        # # Create a new column for buy signals considering the regime
        # # If you want to keep RSI filtering but allow more flexibility
        # dataHourly['Buy_Signal'] = (dataHourly['Buy_Signal_Bollinger']) & (dataHourly['Regime'] == 'Uptrend')

        # # Optionally, you could have a less restrictive buy signal
        # # Buy if below Bollinger Low, regardless of RSI
        # dataHourly['Buy_Signal_Less_Restrictive'] = (dataHourly['close'] < dataHourly['Bollinger_Low'])


        #filter consective
        dataHourly['Buy_Signal'] = dataHourly['Buy_Signal'] & ~dataHourly['Buy_Signal'].shift(1).fillna(False)
        dataHourly['Sell_Signal'] = dataHourly['Sell_Signal'] & ~dataHourly['Sell_Signal'].shift(1).fillna(False)

   #     print(dataHourly)
        with open('out.txt', 'w') as file:
            print(dataHourly, file = file)


        return dataHourly

    def backtest_strategy(self, dataHourly, initial_cash=10000):
        cash = initial_cash  # Starting cash
        position = 0  # No position initially
        times = 0
        percentage = 0.5
        # Track trades and portfolio value
        trades = []
        
        for index, row in dataHourly.iterrows():

            if row['Buy_Signal'] and times < 2:
                # Calculate the amount to invest
                invest_amount = cash * percentage
                # Calculate number of shares to buy
                shares_to_buy = invest_amount / row['close']
                # Update cash and position
                cash -= invest_amount
                position += shares_to_buy
                times+=1
                trades.append({'Type': 'Buy', 'Price': row['close'], 'Shares': shares_to_buy, 'Timestamp': row['timestamp']})
                print(f"Buying {shares_to_buy:.2f} shares at {row['close']} on {row['timestamp']}")
            
            elif row['Sell_Signal'] and position > 0:
                # Sell all shares
                sell_amount = position * row['close'] 
                cash += sell_amount
                trades.append({'Type': 'Sell', 'Price': row['close'], 'Shares': position, 'Timestamp': row['timestamp']})
                print(f"Selling {position:.2f} shares at {row['close']} on {row['timestamp']}")
                position = 0  # Reset position after selling
                times = 0
        
        # Final portfolio value
        final_value = cash + position * dataHourly.iloc[-1]['close']
        print(f"Final portfolio value: {final_value:.2f}")
        
        # Convert trades list to DataFrame for further analysis
        trades_df = pd.DataFrame(trades)
        
        return trades_df, final_value

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

    dataHourly = trading.getBollinger("AAPL")

    x = dataHourly["timestamp"]
    plt.plot(x, dataHourly["close"], label="price")
    plt.plot(x, dataHourly["Bollinger_Low"], label="bol low")
    plt.plot(x, dataHourly["Bollinger_High"], label="bol high")
    plt.scatter(dataHourly.loc[dataHourly['Buy_Signal'], 'timestamp'], dataHourly.loc[dataHourly['Buy_Signal'], 'close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.legend()
    plt.show()

    trades_df, final_value = trading.backtest_strategy(dataHourly)

    # Optionally save trades to a file
    trades_df.to_csv('trades.csv', index=False)
