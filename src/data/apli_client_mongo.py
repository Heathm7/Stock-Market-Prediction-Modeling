from http import client
import time
from typing import Collection
import pandas as pd
from pymongo import MongoClient, ASCENDING
from datetime import datetime
from features.indicators import TechnicalIndicators
from regime.regime_detector import RegimeDetector
from data.providers.alpha_vantage import AlphaVantageProvider


DB_URI = "mongodb+srv://Heathm7:uTuzdXKHNfoqYNL8@stockprediction.34qeeah.mongodb.net/?appName=StockPrediction"
DB_NAME = "stock_prediction"
COLLECTION_NAME = "market_data"
SLEEP_BETWEEN_CALLS = 12

client = MongoClient(DB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

collection.create_index([("symbol", ASCENDING), ("timestamp", ASCENDING)], unique=True)

def fetch_symbol(symbol: str, provider: AlphaVantageProvider = None) -> pd.DataFrame:
    if provider is None:
        provider = AlphaVantageProvider()
    
    df = provider.get_daily(symbol)
    return df

def compute_indicators_and_regimes(df: pd.DataFrame) -> pd.DataFrame:
    ti = TechnicalIndicators(df)
    df_ind = ti.generate_all_indicators()
    rd = RegimeDetector(df_ind)
    df_ind["regime"] = rd.classify_regimes()
    return df_ind

def save_to_mongo(df: pd.DataFrame):
    for _, row in df.iterrows():
        doc = row.to_dict()
        
        if not isinstance(doc["timestamp"], datetime):
            doc["timestamp"] = pd.to_datetime(doc["timestamp"])
        
        collection.update_one({"symbol": doc["symbol"], "timestamp": doc["timestamp"]}, {"$set": doc}, upsert=True)

def update_symbol(symbol: str):
    print(f"[API -> MongoDB] Updated symbol: {symbol}")

    df = fetch_symbol(symbol)
    df_processed = compute_indicators_and_regimes(df)
    save_to_mongo(df_processed)
    time.sleep(SLEEP_BETWEEN_CALLS)
    print(f"[API -> MongoDB] Finish {symbol}, {len(df_processed)} rows saved.")

def update_symbols(symbols: list):
    for symbol in symbols:
        update_symbol(symbol)

if __name__ == "__main__":
    symbols_to_update = ["AAPL", "MSFT", "GOOG"]
    update_symbols(symbols_to_update)