import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from data.data_loader import load_market_data
from stacking.stack import RegimeStack
from models.linear.ridge_model import RidgeModel

MODEL_DIR = "models/linear"
TARGET_COLUMN = "Close"
N_SPLITS = 5
RANDOM_STATE = 42

RIDGE_PARAMS = {"alpha": 1.0, "fit_intercept": True, "normalize": False, "random_state": RANDOM_STATE}

os.makedirs(MODEL_DIR, exists_ok=True)

print(f"Loading market data...")
df = load_market_data("data/your_prices.csv")

features = df.drop(columns=[TARGET_COLUMN, "Regime"])
target = df[TARGET_COLUMN]

train_size = 0.8
split_index = int(len(df) * train_size)
X_train, X_test = features[:split_index], features[split_index:]
y_train, y_test = target[:split_index], target[split_index:]
regimes_train = df["Regime"][:split_index]
regimes_test = df["Regime"][split_index:]

linear_stack = RegimeStack(primary_model_cls=RidgeModel, primary_model_params=RIDGE_PARAMS, secondary_model_cls=None, regimes=[0, 1, 2, 3, 4])
linear_stack.fit(X_train, y_train, regimes_train)

y_pred = linear_stack.predict(X_test, regimes_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Stack Test MSE: {mse:.6f}, R2: {r2:.6f}")

model_path = os.path.join(MODEL_DIR, "linear_stack.pkl")
joblib.dump(linear_stack, model_path)
print(f"Linear stack saved to {model_path}")



