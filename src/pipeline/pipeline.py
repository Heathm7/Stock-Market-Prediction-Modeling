from curses import raw
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from data import api_client
from data.apli_client_mongo import update_symbols
from data.data_loader import MarketDataLoader
from features.indicators import TechnicalIndicators
from regime.regime_detector import RegimeDetector
from stacking.stack import RegimeStack
from models.base_model import BaseModel

class PredictionPipeline:
    def __init__(self, stack: RegimeStack, indicator_engine: TechnicalIndicators = None, regime_detector: RegimeDetector = None):
        self.data_loader = MarketDataLoader()
        self.indicator_engine = indicator_engine or TechnicalIndicators()
        self.regime_detector = regime_detector or RegimeDetector()
        self.stack = stack

    def run(self, symbol: str, interval: str = "5min", lookback: int = 200):
        from data.api_client import MarketAPIClient

        api_client = MarketAPIClient()
        raw_data = self.api_client.fetch_intraday(symbol=symbol, interval=interval)

        if not raw_data or len(raw_data) < lookback:
            raise ValueError(f"Not enough data returned from API")

        df = self.data_loader.to_dataframe(raw_data)
        df = df.tail(lookback)

        feature_df = self.indicator_engine.compute(df)
        feature_df = feature_df.dropna()

        if feature_df.empty:
            raise ValueError(f"Feature dataframe is empty after indicator computation")

        X = feature_df.values.astype(np.float32)

        regime_score = self.regime_detector.detect(feature_df)
        regime_label = self._label_regime(regime_score)
        
        prediction = self.stack.predict(X, regime_score)
        return {"symbol": symbol, "interval:": interval, "regime_score": float(regime_score), "regime_label": regime_label, "prediction": float(prediction), "num_features": X.shape[1], "num_samples": X.shape[0]}

    def load_from_mongo(self, symbols: List[str] = None) -> pd.DataFrame:
        print(f"[Pipeline] Loading data from MongoDB...")
        df = self.data_loader.load_market_data(symbols=symbols)
        print(f"[Pipeline] Loaded {len{df}} rows from MongoDB")
        return df

    @staticmethod
    def _label_regime(regime_score: float) -> str:
        if regime_score < 1:
            return "0_stagnant"
        elif regime_score < 2:
            return "1_low_vol"
        elif 2.0 <= regime_score <= 2.5:
            return "2a_med_low_vol"
        elif 2.5 <= regime_score <= 3.0:
            return "2b_med_high_vol"
        elif regime_score <= 4:
            return "3_high_vol"
        else:
            return "4_extreme_vol"

       