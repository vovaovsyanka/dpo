import logging
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from predict import ModelService

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Investment Advisor API", version="1.0")
service = ModelService(horizon=20)

class TickerRequest(BaseModel):
    ticker: str

class PredictResponse(BaseModel):
    binary_proba: float

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} status={response.status_code} latency={latency:.0f}ms")
    return response

@app.get("/health")
def health():
    logger.info("GET /health -> ok")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_ticker(request: TickerRequest):
    logger.info(f"POST /predict ticker={request.ticker}")
    result = service.predict(request.ticker)
    if "error" in result:
        logger.warning(f"POST /predict ticker={request.ticker} failed: {result['error']}")
        raise HTTPException(status_code=404, detail=result["error"])
    logger.info(f"POST /predict ticker={request.ticker} proba={result['binary_proba']:.4f}")
    return PredictResponse(binary_proba=result["binary_proba"])