import pandas as pd
from typing import Optional

from src.features.indicators import add_technical_indicators
from src.regime.regime_detector import detect_market_regime

class MarketDataLoader:
    def __init__(self, price_col: str = "Close", timestamp_col: str = "timestamp"):
        self.price_col = price_col
        self.timestamp_col = timestamp_col

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        return self._prepare_dataframe(df)

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_dataframe(df.copy())

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df)

        df = self._process_timestamps(df)
        df = self._clean_data(df)
        df = add_technical_indicators(df, price_col=self.price_col)
        df = detect_market_regime(df)
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

def load_market_data(filepath: Optional[str] = "data/sample.csv") -> pd.DataFrame:
    loader = MarketDataLoader()
    return loader.load_from_csv(filepath)





