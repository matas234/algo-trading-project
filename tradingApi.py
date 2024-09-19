
from collections import deque
import json
import math
import random
import pytz
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

import time as t

from datetime import datetime, time, timezone, timedelta

# Load environment variables from .env file
load_dotenv()

class Trading:

    def __init__(self, live=False):
        

        if live:
            self.apiKey = os.getenv("API_KEY_ID")
            self.secretKey = os.getenv("API_SECRET_KEY")
            self.apiBase = "https://data.alpaca.markets"

        else:
            self.apiKey = os.getenv("PAPER_API_KEY_ID")
            self.secretKey = os.getenv("PAPER_API_SECRET_KEY")
            self.apiBase = "https://data.sandbox.alpaca.markets"

        self.feed = "iex"
            
        self.finnKey = os.getenv("FINN_HUB_KEY")

        self.trading_client = TradingClient(self.apiKey, self.secretKey, paper = not live)

        self.historical_client = StockHistoricalDataClient(self.apiKey, self.secretKey)

        self.window_size = 200
        self.totalCash = 100_000

        self.stop_loss_threshold = 0.94 # 10% loss threshold
        self.minPercentage = 0 

        self.loadFromFile()


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
    

    def getBollinger(self, ticker):
        offset = 240
        request = StockBarsRequest(
        symbol_or_symbols=ticker,  
        timeframe=TimeFrame.Minute,   
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
        dataHourly["SMA200"] = ta.trend.SMAIndicator(dataHourly['close'], window=200).sma_indicator()
        dataHourly["RSI"] = ta.momentum.RSIIndicator(dataHourly['close'], window=200).rsi()

        #dataHourly["MACD"] = ta.

        #Drop NaNs created from not enough data to calc bollinger band
        dataHourly.dropna(inplace=True) 

        def identify_market_regime(row):
            if row['close'] > row['SMA200'] and row['RSI'] > 40:  #row['close'] > row['200DMA'] and row['RSI'] > 50 and row['MACD'] > row['Signal Line']:
                return 'Uptrend'
            elif row['close'] < row['SMA200'] and row['RSI'] < 60:   ##row['close'] < row['200DMA'] and row['RSI'] < 50 and row['MACD'] < row['Signal Line']:
                return 'Downtrend'
            else:
                return 'Range'

        def identify_market_regime_RSI(row):
            if row['RSI'] > 50:  #row['close'] > row['200DMA'] and row['RSI'] > 50 and row['MACD'] > row['Signal Line']:
                return 'Uptrend'
            elif row['RSI'] < 50:   ##row['close'] < row['200DMA'] and row['RSI'] < 50 and row['MACD'] < row['Signal Line']:
                return 'Downtrend'
            else:
                return 'Range'
                
        dataHourly['Regime'] = dataHourly.apply(identify_market_regime, axis=1)

        average_volume = dataHourly['volume'].rolling(window=20).mean()

        min_rows = 0 #Default

        dataHourly['Buy_Signal'] =  (dataHourly['close'] < dataHourly['Bollinger_Low'])  & (dataHourly['Regime']=='Downtrend') & (dataHourly.index >= min_rows)#| (dataHourly['volume'] > 1.5*average_volume))
        dataHourly['Sell_Signal'] = (dataHourly['close'] > dataHourly['Bollinger_High']) & (dataHourly['Regime']=='Uptrend') &   (dataHourly.index >= min_rows)#| (dataHourly['volume'] > 1.5*average_volume))

        dataHourly['Buy_Signal_Bollinger'] = dataHourly['close'] < dataHourly['Bollinger_Low']
        dataHourly['Buy_Signal_Regime'] = dataHourly['Regime']=='Uptrend' 

        dataHourly['Sell_Signal_Bollinger'] = (dataHourly['close'] > dataHourly['Bollinger_High'])
        dataHourly['Sell_Signal_Regime'] = (dataHourly['Regime']=='Downtrend')



        #filter consective
        dataHourly['Buy_Signal'] = dataHourly['Buy_Signal'] & ~dataHourly['Buy_Signal'].shift(1).fillna(False)
        dataHourly['Sell_Signal'] = dataHourly['Sell_Signal'] & ~dataHourly['Sell_Signal'].shift(1).fillna(False)

   #     print(dataHourly)
        # with open('out.txt', 'w') as file:
        #     print(dataHourly, file = file)


        return dataHourly
        
    def backtest_strategy(self, dataHourly, initial_cash=10000):
        cash = initial_cash  # Starting cash
        position = 0  # No position initially
        times = 0
        # Track trades and portfolio value
        trades = []
        buy_price = None
        
        for index, row in dataHourly.iterrows():

            if row['Buy_Signal'] and times < 1:
                buy_price = row['close']

                invest_amount = cash * (1 + (self.minPercentage/100))
                # Calculate number of shares to buy
                shares_to_buy = invest_amount / row['close']
                # Update cash and position
                cash -= invest_amount
                position += shares_to_buy
                times+=1
                trades.append({'Type': 'Buy', 'Price': row['close'], 'Shares': shares_to_buy, 'Timestamp': row['timestamp']})
                print(f"Buying {shares_to_buy:.2f} shares at {row['close']} on {row['timestamp']}")
            
            elif row['Sell_Signal'] and position > 0:
                    
                    minimum_selling_price = buy_price 
                    stop_loss_price = buy_price * self.stop_loss_threshold
        
                    current_price =  row['close']

                    if current_price >= minimum_selling_price or current_price <= stop_loss_price:
                        sell_amount = position * row['close'] 
                        cash += sell_amount
                        trades.append({'Type': 'Sell', 'Price': row['close'], 'Shares': position, 'Timestamp': row['timestamp']})
                        print(f"Selling {position:.2f} shares at {row['close']} on {row['timestamp']}")
                        position = 0  # Reset position after selling
                        times = 0
            
                    # else:
                    #     print(f"Holding. Current price {current_price} is below buy price {minimum_selling_price}.")


        
        # Final portfolio value
        final_value = cash + position * dataHourly.iloc[-1]['close']
        print(f"Final portfolio value: {final_value:.2f}")
        
        # Convert trades list to DataFrame for further analysis
        trades_df = pd.DataFrame(trades)
        
        return trades_df, final_value

    def start(self, enableCrypto=False, enableGraph = True):
        stocks = ["BILI","TSLA", "SBUX", "AAPL", "MSFT", "GOOGL", "AMZN", "NFLX", "JPM", "V", "DIS", "KO", "BRK.B", "JNJ", "PG", "XOM", "UNH"]

        average = 0

        with open("out.txt", "w+") as file:

            for stock in stocks:
                print(stock)
                dataHourly = self.getBollinger(stock)


                trades_df, final_value = self.backtest_strategy(dataHourly)
                if enableGraph:
                    showGraph(dataHourly)

                average +=final_value

                print(stock, final_value, file=file)
            
            average /= len(stocks)

            print(f"Average value: {average}")
    

    def loadFromFile(self):
        file_path = "data.json"
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.buy_price = data.get('buy_price', {})
                self.cash = data.get('cash', self.totalCash)
                self.position = data.get('position', {})
                self.times = data.get('times', {})
                
                # Convert the JSON string back to DataFrames
                self.stockData = {key: pd.read_json(value, orient='records') for key, value in data.get('stockData', {}).items()}
                
                print("Data loaded from data.json")
            else:
                raise FileNotFoundError("File not found. Initializing with default values.")
        
        except (json.JSONDecodeError, KeyError, FileNotFoundError, ValueError) as e:
            # Handle any error by resetting to default values
            print(f"Error loading data: {e}. Initializing with default values and deleting {file_path}.")
            
            self.buy_price = {}
            self.stockData = {}
            self.cash = self.totalCash
            self.position = {}
            self.times = {}
            
            # If the file exists and is corrupted, delete it
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted corrupted file: {file_path}")

    def saveToFile(self):
        # Convert DataFrames in stockData to JSON using the orient='records' format
        stockData_jsonable = {key: df.to_json(orient='records') for key, df in self.stockData.items()}
        
        data = {
            'buy_price': self.buy_price,
            'stockData': stockData_jsonable,
            'cash': self.cash,
            'position': self.position,
            'times': self.times
        }
        
        with open("data.json", 'w') as f:
            json.dump(data, f, indent=4)  

        print("Data saved to data.json")


    def removeFile(self):
        file_path = "data.json"
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} has been deleted.")
        else:
            print(f"{file_path} does not exist.")


    def wipeLog(self):
        with open("log.txt", "w") as file:
            pass

    def wait_until_market_open(self):
        clock = self.trading_client.get_clock()

        next_opening = clock.next_open.astimezone(pytz.timezone('America/New_York'))

        est_now = datetime.now(pytz.timezone('America/New_York'))

        sleep_duration = (next_opening - est_now).total_seconds()

        if sleep_duration > 0:
            print(f"Market opens at {next_opening}. Sleeping for {sleep_duration} seconds.")
            t.sleep(sleep_duration)
        else:
            print("Market is already open.")

    def is_market_open(self):

        clock = self.trading_client.get_clock()

        if clock.is_open:
            return True
        else:
            return False

    def get_latest_stock_data(self, ticker):

        url = f'https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnKey}'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'timestamp': datetime.fromtimestamp(data['t'] , tz = pytz.utc), 
                'close': data['c'],
                'high': data['h'],
                'low': data['l'],
                'open': data['pc']
            }
        else:
            print(f"Error fetching data: {response.status_code}")
            return None

    def initialize_stock_data(self, tickers):
        for ticker in tickers:
            offset = 90

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(days=5) - timedelta(minutes=offset),
                end=datetime.now() - timedelta(minutes=offset),
            )

            # Fetch the stock bars data
            bars = self.historical_client.get_stock_bars(request)

            data = []
            for bar in bars[ticker]:
                data.append({
                    'timestamp': bar.timestamp,
                    'close': bar.close,
                    'high': bar.high,
                    'low': bar.low,
                    'open': bar.open
                })

            df = pd.DataFrame(data)
            
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            
            cutoff_time = datetime.strptime('20:00:01', '%H:%M:%S').time()

            df['time_only'] = df['timestamp'].dt.time

            df_filtered = df[df['time_only'] < cutoff_time]

            df_filtered.drop(columns=['time_only'], inplace=True)

            croppedDF = df_filtered.tail(self.window_size).copy()

            self.stockData[ticker] = croppedDF

            

            
    def formatDF(self, dataFrame):

            dataFrame = dataFrame.copy()

            bollinger = ta.volatility.BollingerBands(close=dataFrame['close'], window=20, window_dev=2)

            #Calculate Bolllinger Bands for each hour
            dataFrame['Bollinger_High'] = bollinger.bollinger_hband()
            dataFrame['Bollinger_Med'] = bollinger.bollinger_mavg()
            dataFrame['Bollinger_Low'] = bollinger.bollinger_lband()
            dataFrame["SMA200"] = ta.trend.SMAIndicator(dataFrame['close'], window=200).sma_indicator()
            dataFrame["RSI"] = ta.momentum.RSIIndicator(dataFrame['close'], window=200).rsi()

            #dataFrame["MACD"] = ta.

            #Drop NaNs created from not enough data to calc bollinger band
            dataFrame.dropna(inplace=True) 

            def identify_market_regime(row):
                if row['close'] > row['SMA200'] and row['RSI'] > 40:  #row['close'] > row['200DMA'] and row['RSI'] > 50 and row['MACD'] > row['Signal Line']:
                    return 'Uptrend'
                elif row['close'] < row['SMA200'] and row['RSI'] < 60:   ##row['close'] < row['200DMA'] and row['RSI'] < 50 and row['MACD'] < row['Signal Line']:
                    return 'Downtrend'
                else:
                    return 'Range'

            dataFrame['Regime'] = dataFrame.apply(identify_market_regime, axis=1)

            dataFrame['Buy_Signal'] =  (dataFrame['close'] < dataFrame['Bollinger_Low'])  & (dataFrame['Regime']=='Downtrend') #| (dataFrame['volume'] > 1.5*average_volume))
            dataFrame['Sell_Signal'] = (dataFrame['close'] > dataFrame['Bollinger_High']) & (dataFrame['Regime']=='Uptrend') #| (dataFrame['volume'] > 1.5*average_volume))

            dataFrame['Buy_Signal_Bollinger'] = dataFrame['close'] < dataFrame['Bollinger_Low']
            dataFrame['Buy_Signal_Regime'] = dataFrame['Regime']=='Uptrend' 

            dataFrame['Sell_Signal_Bollinger'] = (dataFrame['close'] > dataFrame['Bollinger_High'])
            dataFrame['Sell_Signal_Regime'] = (dataFrame['Regime']=='Downtrend')


            return dataFrame


    def fetchLatestAndSignal(self, tickers):

        for ticker in tickers:
            # Ensure the ticker has been initialized
            if ticker not in self.stockData:
                self.initialize_stock_data([ticker])
                df = self.stockData[ticker]
                print(f"Initialised for {ticker}:")
            
            latest = self.get_latest_stock_data(ticker)
            latest_df = pd.DataFrame([latest])

            if latest:
                df = self.stockData[ticker]


                combined_df = pd.concat([df, latest_df], ignore_index=True)
                combined_df = combined_df.dropna(axis=1, how='all')


                # Update the window with the combined DataFrame
                df = combined_df.tail(self.window_size)

                self.stockData[ticker] = df
                
                print(f"Updated data for {ticker}:")

                formattedDF = self.formatDF(df)

                selectedColumns = ["Buy_Signal", "Sell_Signal", "Buy_Signal_Regime", "Buy_Signal_Bollinger", "Sell_Signal_Regime", "Sell_Signal_Bollinger"]
                print(formattedDF[selectedColumns])

                self.makeTrade(ticker, formattedDF["Buy_Signal"][200], formattedDF["Sell_Signal"][200],  formattedDF["close"][200], formattedDF["timestamp"][200])


    def makeTrade(self, ticker, buy, sell, close, timestamp):    

        close = float(close)

        if ticker not in self.times:
            self.times[ticker] = 0

        if ticker not in self.position:
            self.position[ticker] = 0

        if buy and self.times[ticker] < 1:
            
            invest_amount = min(self.totalCash * (1/len(self.stocks)), self.cash)

            shares_to_buy = invest_amount / close

            self.cash -= invest_amount

            self.position[ticker] += shares_to_buy

            self.times[ticker] +=1

            self.buy_price[ticker] = close

            self.alpacaBuy(ticker, shares_to_buy)

            with open("log.txt", "a") as file:
                print(f"Buying {shares_to_buy:.2f} shares of {ticker} at {close} on {timestamp}", file=file)
        

        elif sell and self.position[ticker] > 0:
            
            minimum_selling_price = self.buy_price[ticker] 
            stop_loss_price = self.buy_price[ticker] * self.stop_loss_threshold

            current_price =  close

            if current_price >= minimum_selling_price or current_price <= stop_loss_price:
                sell_amount = self.position[ticker] * close 
                self.cash += sell_amount
                with open("log.txt", "a") as file:
                    print(f"Selling {self.position[ticker]:.2f} shares of {ticker} at {close} on {timestamp}", file=file)


                self.alpacaSell(ticker)

                self.position[ticker] = 0  # Reset position after selling
                self.times[ticker] = 0
        

    def alpacaBuy(self, ticker, sharesToBuy):
        try:
            # Ensure the asset is tradable
            if self.checkTradable(ticker) and sharesToBuy > 0:

                market_order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=sharesToBuy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )

                # Submit the buy order
                market_order = self.trading_client.submit_order(
                    order_data=market_order_data
                )

                print(market_order)
            else:
                print(f"{ticker} is not tradable or there is no money to buy the stock.")

        except Exception as e:
            print(f"An error occurred while trying to buy {ticker}: {str(e)}")


    def alpacaSell(self, ticker):
        try:
            # Get current position for the ticker
            position = self.trading_client.get_open_position(ticker)
            if position:
                quantity = float(position.qty)

                market_order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )

                # Submit the sell order
                market_order = self.trading_client.submit_order(
                    order_data=market_order_data
                )

                print(market_order)
            else:
                print(f"No open position in {ticker} to sell.")
        
        except Exception as e:
            print(f"An error occurred while trying to sell {ticker}: {str(e)}")


    def alpacaSellAll(self):
        try:
            # Get all open positions
            positions = self.trading_client.get_all_positions()

            # Loop through each position and sell all shares
            for position in positions:
                ticker = position.symbol
                qty = float(position.qty) 

                # Create the market order request to sell the full quantity
                market_order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force="day"
                )

                # Submit the sell order
                self.trading_client.submit_order(order_data=market_order_data)
                print(f"Submitted sell order for {qty} shares of {ticker}.")
        
        except Exception as e:
            print(f"Error in selling positions: {e}")


    def liveStart(self, reset = False):
        self.stocks = [
            "BILI", "LLY", "AAPL", "MSFT", "GOOGL", "AMZN", "NFLX", "JPM", "V", "DIS", 
            "KO", "BRK.B", "JNJ", "PG", "XOM", "UNH", "TSLA", "NVDA", "ADBE", "META"
        ]
        
        if reset:
            trading.alpacaSellAll()
            trading.wipeLog()
            trading.removeFile()
            t.sleep(60)

            trading.loadFromFile()

        
        while True:
            if not self.is_market_open():
                self.saveToFile()
                print("Market is closed. Waiting until market opens.")
                self.wait_until_market_open()
                t.sleep(60)
            else:
                self.fetchLatestAndSignal(self.stocks)
                t.sleep(60)
    


if __name__ == "__main__":
    trading = Trading()
    # trading.wipeLog()

    #trading.requestAccount()
    #trading.setMarketOrder("AAPL", 1)
    #trading.getOrders()
    # trading.getBalanceChange()
    # trading.start()

    # trading.fetchLatestAndSignal(["TSLA"])

    # trading.alpacaSell("TSLA")
    # trading.alpacaBuy("TSLA", 2)

    # trading.alpacaSell("AAPL")

    trading.liveStart()

    #trading.wipeLog()
    # trading.makeTrade("AAPL", "True", "False" ,"220.91", "2024-09-10 09:30:00-04:00")
    # trading.makeTrade("AAPL", "True", "False" ,"220.91", "2024-09-10 09:30:00-04:00")
    # trading.makeTrade("AAPL", "False", "True" ,"220.91", "2024-09-10 09:30:00-04:00")

    ## AGI



def showGraph(dataHourly):
    

    x = dataHourly["timestamp"]

    useBolinger = False

    plt.plot(x, dataHourly["close"], label="price")
    plt.plot(x, dataHourly["Bollinger_Low"], label="bol low")
    plt.plot(x, dataHourly["Bollinger_High"], label="bol high")

    if not useBolinger:
        plt.scatter(dataHourly.loc[dataHourly['Buy_Signal'], 'timestamp'], dataHourly.loc[dataHourly['Buy_Signal'], 'close'], marker='^', color='green', s=100, label='Buy Signal')
        plt.scatter(dataHourly.loc[dataHourly['Sell_Signal'], 'timestamp'], dataHourly.loc[dataHourly['Sell_Signal'], 'close'], marker='v', color='red', s=100, label='Sell Signal')
    else:
        plt.scatter(dataHourly.loc[dataHourly['Buy_Signal_Bollinger'], 'timestamp'], dataHourly.loc[dataHourly['Buy_Signal_Bollinger'], 'close'], marker='^', color='green', s=100, label='Buy Signal')
        plt.scatter(dataHourly.loc[dataHourly['Sell_Signal_Bollinger'], 'timestamp'], dataHourly.loc[dataHourly['Sell_Signal_Bollinger'], 'close'], marker='v', color='red', s=100, label='Sell Signal')

    plt.legend()
    plt.show()