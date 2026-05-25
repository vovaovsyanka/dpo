import requests
import random
import joblib
from pathlib import Path

# ========== Конфигурация ==========
API_URL = "http://localhost:8000/predict"   # адрес вашего сервера
DATA_DIR = Path(__file__).parent.parent / "data"   # папка с данными

# ========== Загрузка списка валидных тикеров ==========
valid_tickers = joblib.load(DATA_DIR / "valid_tickers.pkl")
print(f"Загружено валидных тикеров: {len(valid_tickers)}")

# ========== Выбор 5 случайных тикеров ==========
random.seed(42)  # для воспроизводимости
sample_tickers = random.sample(valid_tickers, min(5, len(valid_tickers)))

# Добавляем один заведомо несуществующий тикер
test_tickers = sample_tickers + ["FAKETICKER"]

print("\nТестируемые тикеры:")
for t in test_tickers:
    print(f"  - {t}")

# ========== Отправка запросов ==========
print("\n" + "="*50)
for ticker in test_tickers:
    print(f"\nЗапрос для тикера: {ticker}")
    response = requests.post(API_URL, json={"ticker": ticker})
    
    if response.status_code == 200:
        result = response.json()
        print(f"  ✅ Успех: вероятность роста = {result['binary_proba']:.4f}")
    else:
        print(f"  ❌ Ошибка {response.status_code}: {response.json().get('detail', response.text)}")

print("\n" + "="*50)
print("Тестирование завершено.")