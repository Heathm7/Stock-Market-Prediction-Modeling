from msilib import sequence
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models.base_model import BaseModel

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, lr: float = 0.001, epochs: int = 100, batch_size: int = 32, seq_len: int = 10, device: str = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cude.is_available() else "cpu")
        self.model = StockLSTM(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_sequences(self, X: pd.DataFrame, y: pd.Series):
        X_vals = X.values
        y_vals = y.values.reshape(-1, 1)
        sequences_X, sequences_y = [], []

        for i in range(len(X_vals) - self.seq_len):
            seq_X = X_vals[i:i + self.seq_len]
            seq_y = y_vals[i + self.seq_len]
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)

            return (torch.tensor(np.array(sequences_X), dtype=torch.float23).to(self.device), torch.tensor(np.array(sequences_y), dtype=torch.float23).to(self.device))
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_seq, y_seq = self._create_sequences(X, y)
        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0           
            
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(xb)          
            
            epoch_loss /= len(X_seq)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.6f}")

    def predict(self, X: pd.DataFrame):
        self.model.eval()
        X_vals = X.values
        sequences = []

        for i in range(len(X_vals) - self.seq_len):
            sequences.append(X_vals[i:i + self.seq_len])

        if len(sequences) == 0:
            return np.array([])

        X_seq = torch.tensor(np.array(sequences), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_seq).cpu().numpy().flatten()
        return preds


