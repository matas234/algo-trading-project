
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

        self.buy_price = {}
        self.stop_loss_threshold = 0.90  # 10% loss threshold

    
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
        percentage = 1
        # Track trades and portfolio value
        trades = []
        buy_price = None
        
        for index, row in dataHourly.iterrows():

            if row['Buy_Signal'] and times < 1:
                buy_price = row['close']

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

# def sell(self,asset, quantity, current_price):

    def start(self):
        stocks = ["BILI","TSLA", "SBUX", "AAPL", "MSFT", "GOOGL", "AMZN", "NFLX", "JPM", "V", "DIS", "KO", "BRK.B", "JNJ", "PG", "XOM", "UNH"]
        
        average = 0
        
        with open("out.txt", "w+") as file:

            for stock in stocks:
                print(stock)
                dataHourly = self.getBollinger(stock)


                trades_df, final_value = self.backtest_strategy(dataHourly)
                # showGraph(dataHourly)

                average +=final_value

                print(stock, final_value, file=file)
            
            average /= len(stocks)

            print(f"Average value: {average}")

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




if __name__ == "__main__":
    trading = Trading()
    #trading.requestAccount()
    #trading.setMarketOrder("AAPL", 1)
    #trading.getOrders()
   # trading.getBalanceChange()
    trading.start()


    ## AGI


