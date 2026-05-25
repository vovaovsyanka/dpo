# prepare_inference_data.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========== Конфигурация ==========
BASE_DIR = Path.cwd().parent.parent   # корень проекта (если скрипт в src/)
PROCESSED_DIR = BASE_DIR / "bigdata" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "final_models"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SEQ_LEN = 20
VAL_START = '2022-01-01'
MIN_TRAIN_DAYS = SEQ_LEN + 10

# ========== Загрузка данных и списка признаков ==========
df = pd.read_parquet(PROCESSED_DIR / "combined_features.parquet")
with open(PROCESSED_DIR / "feature_columns.txt") as f:
    all_feature_cols = [line.strip() for line in f if line.strip()]

# ========== Фильтрация тикеров (как в обучении) ==========
valid_tickers = []
for ticker in df['ticker'].unique():
    ticker_df = df[df['ticker'] == ticker]
    train_len = len(ticker_df[ticker_df['date'] < VAL_START])
    if train_len >= MIN_TRAIN_DAYS:
        valid_tickers.append(ticker)

print(f"Оставлено тикеров: {len(valid_tickers)}")

# ========== Загрузка обученных трансформеров ==========
scaler = joblib.load(ARTIFACTS_DIR / "pca_scaler.pkl")
pca = joblib.load(ARTIFACTS_DIR / "pca_model.pkl")

# ========== Применяем скейлер и PCA ко всем строкам (без разбиения на окна) ==========
all_rows = []
for ticker in valid_tickers:
    ticker_data = df[df['ticker'] == ticker].sort_values('date').copy()
    # Заполняем пропуски (ffill) как в обучении
    ticker_data[all_feature_cols] = ticker_data[all_feature_cols].ffill()
    ticker_data = ticker_data.dropna(subset=all_feature_cols)
    if len(ticker_data) < SEQ_LEN:
        continue
    # Берём только последние SEQ_LEN строк (минимально необходимое)
    last_rows = ticker_data.iloc[-SEQ_LEN:].copy()
    # Применяем scaler и PCA
    X_df = last_rows[all_feature_cols]          # оставляем DataFrame
    X_scaled = scaler.transform(X_df)
    pca_vals = pca.transform(X_scaled)
    # Создаём DataFrame с PCA-компонентами
    pca_df = pd.DataFrame(pca_vals, columns=[f'pca_{i}' for i in range(pca.n_components_)])
    pca_df['ticker'] = ticker
    # Сохраняем даты (опционально, для отладки)
    pca_df['date'] = last_rows['date'].values
    all_rows.append(pca_df)

df_pca_minimal = pd.concat(all_rows, ignore_index=True)
df_pca_minimal = df_pca_minimal.sort_values(['ticker', 'date']).reset_index(drop=True)

# ========== Сохраняем ==========
df_pca_minimal.to_parquet(DATA_DIR / "pca_features.parquet", index=False)
joblib.dump(valid_tickers, DATA_DIR / "valid_tickers.pkl")

print(f"Сохранено: {DATA_DIR / 'pca_features.parquet'}, форма: {df_pca_minimal.shape}")
print(f"Сохранён список тикеров: {DATA_DIR / 'valid_tickers.pkl'}")