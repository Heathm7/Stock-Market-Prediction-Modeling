import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from data.data_loader import load_market_data
from stacking.stack import RegimeStack
from models.linear.ridge_model import RidgeModel

MODEL_DIR = "models/linear"
TARGET_COLUMN = "target"
RANDOM_STATE = 42

RIDGE_PARAMS = {"alpha": 1.0, "fit_intercept": True, "normalize": False, "random_state": RANDOM_STATE}

def prepare_dataset(df):
    X = df.drop(columns=[TARGET_COLUMN, "symbol", "timestamp", "Regime"])
    y = df[TARGET_COLUMN]
    regimes = df["Regime"]
    return X, y, regimes

def save_model(model, name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Linear model saved to {path}")

def train_pipeline(df):
    print("[Linear Trainer] Preparing dataset...")
    X, y, regimes = prepare_dataset(df)

    print("[Linear Trainer] Initializing RegimeStack...")
    linear_stack = RegimeStack(primary_model_cls=RidgeModel, primary_model_params=RIDGE_PARAMS, secondary_model_cls=None, regimes=[0, 1, 2, 3, 4])

    print("[Linear Trainer] Fitting RegimeStack...")
    linear_stack.fit(X, y, regimes)

    save_model(linear_stack, "linear_stack")
    return linear_stack


