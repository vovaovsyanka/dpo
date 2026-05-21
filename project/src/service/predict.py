import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor

MODEL_DIR = Path(__file__).parent.parent.parent / "artifacts" / "final_models"

class ModelService:
    def __init__(self, horizon=20):
        self.horizon = horizon
        with open(MODEL_DIR / "feature_columns.txt") as f:
            self.feature_cols = [line.strip() for line in f if line.strip()]
        # загрузка моделей
        self.binary_model = CatBoostClassifier()
        self.binary_model.load_model(str(MODEL_DIR / f"catboost_binary_{horizon}d.cbm"))
        # если есть регрессия:
        self.return_model = CatBoostRegressor()
        self.return_model.load_model(str(MODEL_DIR / f"catboost_return_{horizon}d.cbm"))
    
    def predict(self, features_list):
        df = pd.DataFrame(features_list)
        # убеждаемся, что все нужные колонки есть, недостающие заполняем NaN (CatBoost сам обработает)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_cols]
        proba = self.binary_model.predict_proba(df)[:, 1]
        returns = self.return_model.predict(df)
        return [{"binary_proba": p, "return_pred": r} for p, r in zip(proba, returns)]