from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import pandas as pd

class BaseModel(ABC):
    def __init__(self, name: str, supported_regimes: List[int]):
        self.name = name
        self.supported_regimes = supported_regimes
        self.is_trained = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support predict_proba")

    def supports_regime(self, regime_id: int) -> bool:
        return regime_id in self.supported_regimes



