from numpy._typing import _VoidDTypeLike
import requests
import pandas as pd
from src.data.providers.base_provider import MarketDataProvider

class AlphaVantageProvider(MarketDataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_historical_data(self, symbol: str, interval: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
            }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()

        if "Time Series (Daily)" not in data:
            raise ValueError(f"Alpha Vantage error: {data}")

        df = (
            pd.DataFrame.from_dict(data["Time Series (Daily)"], orient = "index")
            .astype(float)
            .reset_index()
            .rename(columns={
                "index": "timestamp",
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "6. volume": "volume"
                })
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]


    



