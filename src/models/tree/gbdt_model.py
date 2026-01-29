import joblib
import numpy as np
from typing import Union
from models.base_model import BaseModel

class GBDTModel(BaseModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try: 
            model = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load GBDT model from {self.model_path}") from e

        return model

    def predict(self, X: Union[np.ndarray, list]) -> float:
        if isinstance(X, list):
            X = np.array(X)

        if X.ndim != 2:
            raise ValueError(f"Input X must be 2D")

        if X.shape[0] < 1:
            raise ValueError(f"X must contain at least one row")

        x_latest = X[-1].reshape(1, -1)

        try:
            prediction = self.model.predict(x_latest)
        except Exception as e:
            raise RuntimeError(f"GBDT prediction failed") from e

        return float(prediction[0])
