import pandas as pd
import numpy as np

class RegimeDetector:
    def __init__(self, df: pd.DataFrame, vol_column: str = 'Volatility_20'):
        self.df = df.copy()
        if vol_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain column '{vol_column}'")
        self.vol_column = vol_column

    def classify_regimes(self) -> pd.Series:
        vol = self.df[self.vol_column]

        thresholds = np.percentile(vol.dropna(), [20,40,60,80])
        regimes = pd.Series(index=vol.index, dtype=int)

        regimes[vol <= thresholds[0]] = 0
        regimes[(vol > thresholds[0]) & (vol <thresholds[1])] = 1
        regimes[(vol > thresholds[1]) & (vol <thresholds[2])] = 2
        regimes[(vol > thresholds[2]) & (vol <thresholds[3])] = 3
        regimes[vol > thresholds[3]] = 4
        return regimes

    def add_regime_column(self, column_name: str = 'Regime') -> pd.DataFrame:
        self.df[column_name] = self.classify_regimes()
        return self.df

def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    detector = RegimeDetector(df)
    return detector.add_regime_column()





