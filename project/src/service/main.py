from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse
from predict import ModelService

app = FastAPI(title="Investment Advisor API", version="1.0")
service = ModelService(horizon=20)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        preds = service.predict(request.features)
        return PredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))