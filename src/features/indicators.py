import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'Close' not in self.df.columns:
            raise ValueError("DataFrame must constain 'Close' column")

    def moving_average(self, window: int = 20) -> pd.Series:
        return self.df['Close'].rolling(window=window).mean()

    def exponential_moving_average(self, span: int = 20) -> pd.Series:
        return self.df['Close'].ewm(span=span, adjust=False).mean()

    def rolling_volatility(self, window: int = 20) -> pd.Series:
        returns = self.df['Close'].pct_change()
        return returns.rolling(window=window).std()

    def momentum(self, window: int = 14) -> pd.Series:
        delta = self.df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def macd(self, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9):
        ema_fast = self.exponential_moving_average(span=fast_span)
        ema_slow = self.exponential_moving_average(span=slow_span)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        return macd_line, signal_line

    def generate_all_indicators(self) -> pd.DataFrame:
        df_ind = self.df.copy()
        df_ind['SMA_20'] = self.moving_average(20)
        df_ind['EMA_20'] = self.exponential_moving_average(20)
        df_ind['Volatility_20'] = self.rolling_volatility(20)
        df_ind['Momentum_10'] = self.momentum(10)
        df_ind['RSI_14'] = self.rsi(14)
        macd_line, signal_line = self.macd()
        df_ind['MACD'] = macd_line
        df_ind['MACD_signal'] = signal_line
        return df_ind
                                      
def compute_indicators(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    ti = TechnicalIndicators(df)
    return ti.generate_all_indicators()

