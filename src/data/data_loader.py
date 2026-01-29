import pandas as pd
from typing import List, Optional
from pymongo.collection import Collection
from datetime import datetime
from features.indicators import TechnicalIndicators
from regime.regime_detector import RegimeDetector
from data.apli_client_mongo import collection as mongo_collection

class MarketDataLoader:
    def __init__(self, price_col: str = "Close", timestamp_col: str = "timestamp", mongo_collection: Collection = mongo_collection):
        self.price_col = price_col
        self.timestamp_col = timestamp_col
        self.mongo_collection = mongo_collection

    def load_from_mongo(self, symbols: Optional[List[str]] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        query = {}
        
        if symbols:
            query["symbol"] = {"$in": symbols}
        
        if start_date:
            query[self.timestamp_col]["#gte"] = start_date

            if start_date:
                query[self.timestamp_col]["$gte"] = start_date
            if end_date:
                query[self.timestamp_col]["$lte"] = end_date

        cursor = self.mongo_collection.find(query).sort(self.timestamp_col, 1)
        df = pd.DataFrame(list(cursor))

        if df.empty:
            raise ValueError(f"No data returned from MongoDB for given filters.")

        return self._prepare_dataframe(df)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df)

        df = self._process_timestamps(df)
        df = self._clean_data(df)

        indicator_cols = ["SMA_20", "EMA_20", "Volatility_20", "Momentum_10", "RSI_14", "MACD", "MACD_signal"]
        missing_indicators = [c for c in indicator_cols if c not in df.columns]

        if missing_indicators:
            ti = TechnicalIndicators(df)
            df_ind = ti.generate_all_indicators()

            for c in df_ind.columns:
                if c not in df.columns:
                    df[c] = df_ind[c]
        
        if "regime" not in df.columns:
            rd = RegimeDetector(df)
            df["regime"] = rd.classify_regimes()

        return df
        
    def _validate_columns(self, df: pd.DataFrame):
        required = {self.price_col, self.timestamp_col}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.sort_values(self.timestamp_col)
        df = df.reset_index(drop=True)
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[self.price_col])
        df = df.ffill()
        return df

def load_market_data(self, symbols: Optional[List[str]] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        return self.load_from_mongo(symbols=symbols, start_date=start_date, end_date=end_date)





