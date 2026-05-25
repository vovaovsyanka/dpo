import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "artifacts" / "final_models"

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=1, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class ModelService:
    def __init__(self, horizon=20, seq_len=20):
        self.horizon = horizon
        self.seq_len = seq_len

        # Загрузка списка валидных тикеров
        self.valid_tickers = joblib.load(DATA_DIR / "valid_tickers.pkl")

        # Загрузка готового минимального датасета в PCA-пространстве
        self.df_pca = pd.read_parquet(DATA_DIR / "pca_features.parquet")
        self.df_pca = self.df_pca.sort_values(['ticker', 'date'])

        # Определяем размерность PCA
        pca_cols = [c for c in self.df_pca.columns if c.startswith('pca_')]
        input_dim = len(pca_cols)

        # Загрузка модели GRU
        model_path = MODEL_DIR / f"gru_pca_{horizon}d.pt"
        state = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = GRUModel(input_dim=input_dim)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, ticker):
        if ticker not in self.valid_tickers:
            return {"error": f"Ticker '{ticker}' not found. Available tickers count: {len(self.valid_tickers)}"}

        ticker_data = self.df_pca[self.df_pca['ticker'] == ticker].sort_values('date')
        if len(ticker_data) < self.seq_len:
            # На всякий случай, хотя в датасете уже ровно seq_len строк
            return {"error": f"Not enough data for '{ticker}': need {self.seq_len} rows, got {len(ticker_data)}"}

        # Данные уже в PCA-пространстве и масштабированы, берём все строки (их ровно seq_len)
        pca_cols = [c for c in ticker_data.columns if c.startswith('pca_')]
        X = ticker_data[pca_cols].values.astype(np.float32)   # (seq_len, n_components)
        X_tensor = torch.tensor(X).unsqueeze(0)               # (1, seq_len, n_components)

        with torch.no_grad():
            logits = self.model(X_tensor).squeeze().item()
            proba = 1.0 / (1.0 + np.exp(-logits))

        return {"binary_proba": proba}