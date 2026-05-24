import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent.parent / "artifacts" / "final_models"

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

        # Загрузка списка исходных признаков (109 признаков)
        with open(MODEL_DIR / "feature_columns.txt") as f:
            self.raw_feature_cols = [line.strip() for line in f if line.strip()]

        # Загрузка PCA-трансформеров
        self.scaler = joblib.load(MODEL_DIR / "pca_scaler.pkl")
        self.pca = joblib.load(MODEL_DIR / "pca_model.pkl")

        # Загрузка модели (ожидает input_dim = число компонент PCA)
        self.model = GRUModel(input_dim=self.pca.n_components_)
        state = torch.load(MODEL_DIR / f"gru_pca_{horizon}d.pt", map_location=torch.device('cpu'))
        self.model.load_state_dict(state)
        self.model.eval()

    def _preprocess(self, df_seq: pd.DataFrame) -> np.ndarray:
        """
        Применяет scaler и PCA к окну признаков.
        df_seq: (seq_len, n_features) — исходные признаки.
        Возвращает: (seq_len, n_components) — проекция PCA.
        """
        # Заполняем отсутствующие колонки (если вход неполный)
        for col in self.raw_feature_cols:
            if col not in df_seq.columns:
                df_seq[col] = np.nan
        df_seq = df_seq[self.raw_feature_cols]
        # Заполняем пропуски (ffill) — как в обучении
        df_seq = df_seq.ffill().bfill().fillna(0)
        X = df_seq.values.astype(np.float32)
        # Стандартизация и PCA
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def predict(self, features_list):
        """
        features_list: list of list of dict
        Каждый внутренний список – последовательность из seq_len словарей.
        Возвращает список dict с ключом "binary_proba".
        """
        predictions = []
        for seq_dicts in features_list:
            if len(seq_dicts) != self.seq_len:
                raise ValueError(f"Each sequence must have length {self.seq_len}, got {len(seq_dicts)}")

            df_seq = pd.DataFrame(seq_dicts)
            X_pca = self._preprocess(df_seq)          # (seq_len, n_components)
            X_pca_norm = self._normalize_sequence(X_pca)   # z-score внутри окна (как при обучении)
            X_tensor = torch.tensor(X_pca_norm).unsqueeze(0)  # (1, seq_len, n_components)

            with torch.no_grad():
                logits = self.model(X_tensor).squeeze().item()
                proba = 1.0 / (1.0 + np.exp(-logits))
            predictions.append({"binary_proba": proba})
        return predictions

    @staticmethod
    def _normalize_sequence(seq):
        """seq: np.array (seq_len, n_features) -> нормализация z-score по столбцам"""
        mean = seq.mean(axis=0, keepdims=True)
        std = seq.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (seq - mean) / std