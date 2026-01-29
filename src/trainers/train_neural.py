import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.data_loader import load_market_data
from models.neural.neural_model import LSTMModel
from trainers.train_tree import TARGET_COLUMN

MODEL_DIR = "models/neural"
TARGET_COLUMN = "Close"
SEQ_LEN = 10
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
EPOCHS = 100
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exists_ok=True)

print(f"Loading market data...")
df = load_market_data(symbols=["AAPL", "MSFT", "GOOG"])

features = df.drop(columns=[TARGET_COLUMN, "Regime", "symbol", "timestamp"])
target = df[TARGET_COLUMN]
regimes = df["Regime"]

high_vol_mask = regimes >= 3
features_hv = features[high_vol_mask].reset_index(drop=True)
target_hv = target[high_vol_mask].reset_index(drop=True)

if len(features_hv) < SEQ_LEN:
    raise ValueError(f"Not enough high-volatility data for LSTM sequences.")

scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_hv), columns=features_hv.columns)

input_dim = features_scaled.shape[1]
lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

print(f"Training LSTM neural model on high-volatility regimes...")
lstm_model.fit(features_scaled, target_hv)

model_path = os.path.join(MODEL_DIR, "neural_stack_lstm_pkl")
joblib.dump({"model": lstm_model, "scaler": scaler}, model_path)
print(f"LSTM neural stack saved to {model_path}")
