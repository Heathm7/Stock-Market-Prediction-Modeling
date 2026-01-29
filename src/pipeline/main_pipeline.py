import os
import joblib
from data.api_client_mongo import update_symbols
from data.data_loader import load_market_data
from features.indicators import TechnicalIndicators
from regime.regime_detector import RegimeDetector
from trainers import train_linear, train_neural, train_tree

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN" "TSLA"]

os.makedirs("models/tree", exist_ok=True)
os.makedirs("models/linear", exist_ok=True)
os.makedirs("models/neural", exist_ok=True)

def main_pipeline(symbols=SYMBOLS):
    print("[Pipeline] Updating symbols in MongoDB...")
    update_symbols(symbols)
    print("[Pipeline] Data update complete.\n")

    print("[Pipeline] Loading market data from MongoDB...")
    df = load_market_data(symbols=symbols)
    print("[Pipeline] Loaded {len(df)} rows from MongoDB.\n")

    print("[Pipeline] Training Tree (GBDT) model...")
    tree_model = train_tree.train_pipeline(df)
    print("[Pipeline] Tree model training complete.\n")

    print("[Pipeline] Training Linear (Ridge) model...")
    linear_model = train_linear.train_pipeline(df)
    print("[Pipeline] Linear model training complete\n")

    print("[Pipeline] Training Neural (LSTM) model...")
    neural_model = train_neural.train_pipeline(df)
    print("[Pipeline] Neural model training complete\n")
    print("[Pipeline] All models trained successfully.")

if __name__ == "__main__":
    main_pipeline()

    