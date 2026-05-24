from pydantic import BaseModel
from typing import List, Dict, Any

class PredictRequest(BaseModel):
    features: List[Dict[str, Any]]   # список объектов с признаками

class PredictResponse(BaseModel):
    predictions: List[Dict[str, float]]   # [{"binary_proba": 0.7}]