
import requests
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# Load environment variables from .env file
load_dotenv()

class Trading:

    def __init__(self, live=False):
        


        if live:
            self.apiKey = os.getenv("API_KEY_ID")
            self.secretKey = os.getenv("API_SECRET_KEY")

            # Alpaca API endpoint
            self.apiBase = "https://api.alpaca.markets"

        else:
            self.apiKey = os.getenv("PAPER_API_KEY_ID")
            self.secretKey = os.getenv("PAPER_API_SECRET_KEY")

            # Alpaca API endpoint
            self.apiBase = "https://paper-api.alpaca.markets"


    
        # Set up the headers with your API keys
        self.headers = {
            "APCA-API-KEY-ID": self.apiKey ,
            "APCA-API-SECRET-KEY": self.secretKey
        }
        
        self.trading_client = TradingClient('api-key', 'secret-key')


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

    def getOrders(self):
        get_orders_data = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=100,
            nested=True  # show nested multi-leg orders
        )

        orders = self.trading_client.get_orders(filter=get_orders_data)

        print(orders)



if __name__ == "__main__":
    trading = Trading()
    trading.requestAccount()
    trading.getOrders()
    # trading.checkTradable("AAPL")