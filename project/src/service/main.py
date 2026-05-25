from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import ModelService

app = FastAPI(title="Investment Advisor API", version="1.0")
service = ModelService(horizon=20)

class TickerRequest(BaseModel):
    ticker: str

class PredictResponse(BaseModel):
    binary_proba: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_ticker(request: TickerRequest):
    result = service.predict(request.ticker)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return PredictResponse(binary_proba=result["binary_proba"])