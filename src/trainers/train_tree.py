import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from data.data_loader import load_market_data
from features.indicators import compute_indicators

MODEL_DIR = "models/tree"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
N_SPLITS = 5

GBDT_PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8, "random_state": RANDOM_STATE}

def prepare_dataset(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN, "symbol", "timestamp", "regime"])
    y = df[TARGET_COLUMN]
    return X, y

def train_gbdt(X: pd.DataFrame, y: pd.Series):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    models = []
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = GradientBoostingRegressor(**GBDT_PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, square=False)

        print(f"[Fold {fold}] RMSE: {rmse:.6f}")

        models.append(model)
        scores.append(rmse)

    best_model = models[np.argmin(scores)]
    print(f"Best RMSE: {min(scores):.6f}")
    return best_model

def save_model(model, name:str):
    os.makedirs(MODEL_DIR, exists_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def train_pipeline(df):
    print("[Tree Trainer] Preparing dataset")
    X, y = prepare_dataset(df)

    print("[Tree Trainer] Training GBDT model...")
    model = train_gbdt(X, y)
    save_model(model, "gbdt_base")
    return model

