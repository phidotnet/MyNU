import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from .ml_train_models import FEATURE_COLUMNS, MODELS_DIR

LOGS_DIR = Path(__file__).resolve().parents[1] / "runtime/logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
_info_handler = RotatingFileHandler(LOGS_DIR / "api.log", maxBytes=5_000_000, backupCount=3)
_info_handler.setLevel(logging.INFO)
_info_handler.setFormatter(_fmt)
_error_handler = RotatingFileHandler(LOGS_DIR / "api-error.log", maxBytes=5_000_000, backupCount=3)
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_info_handler, _error_handler, _console_handler])
logger = logging.getLogger(__name__)

app = FastAPI(title="Retail Average Purchase Value API")

HTTP_REQUEST_COUNTER = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status_code"]
)
HTTP_REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "HTTP request latency", ["method", "path"]
)
PREDICTION_COUNTER = Counter(
    "model_predictions_total", "Total prediction requests per model", ["model_name", "status"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds", "Prediction latency per model", ["model_name"]
)


@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """Record per-request count and latency for Prometheus scraping."""
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    HTTP_REQUEST_LATENCY.labels(method=request.method, path=path).observe(time.perf_counter() - start)
    HTTP_REQUEST_COUNTER.labels(method=request.method, path=path, status_code=response.status_code).inc()
    return response


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


MODEL_PATHS = {
    "random_forest_100": Path(os.getenv("RANDOM_FOREST_100_MODEL_PATH", MODELS_DIR / "random_forest_100" / "model.joblib")),
    "random_forest_200": Path(os.getenv("RANDOM_FOREST_200_MODEL_PATH", MODELS_DIR / "random_forest_200" / "model.joblib")),
    "gradient_boosting_100": Path(os.getenv("GRADIENT_BOOSTING_100_MODEL_PATH", MODELS_DIR / "gradient_boosting_100" / "model.joblib")),
    "gradient_boosting_200": Path(os.getenv("GRADIENT_BOOSTING_200_MODEL_PATH", MODELS_DIR / "gradient_boosting_200" / "model.joblib")),
}
MODELS = {}


class RetailFeatures(BaseModel):
    age: float
    zipcode: float
    year: float
    ratings: float
    transaction_hour: float
    income: str
    customer_segment: str
    product_category: str
    product_brand: str


def load_models() -> None:
    """Load the configured, versioned model artifacts once at application startup."""
    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.is_file():
            raise RuntimeError(f"Model file does not exist: {model_path}")
        MODELS[model_name] = joblib.load(model_path)
        logger.info("model=%s status=loaded path=%s", model_name, model_path)


@app.on_event("startup")
def startup() -> None:
    load_models()


def predict(model_name: str, features: RetailFeatures) -> dict[str, float | str]:
    """Return an average-purchase-value prediction from one loaded model."""
    model = MODELS.get(model_name)
    if model is None:
        PREDICTION_COUNTER.labels(model_name=model_name, status="unavailable").inc()
        logger.error("model=%s status=unavailable", model_name)
        raise HTTPException(status_code=503, detail=f"{model_name} model is not loaded")

    input_data = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)
    try:
        with PREDICTION_LATENCY.labels(model_name=model_name).time():
            prediction = float(model.predict(input_data)[0])
    except Exception:
        PREDICTION_COUNTER.labels(model_name=model_name, status="error").inc()
        logger.exception("model=%s status=error", model_name)
        raise HTTPException(status_code=500, detail=f"{model_name} prediction failed")

    PREDICTION_COUNTER.labels(model_name=model_name, status="success").inc()
    logger.info("model=%s status=success prediction=%.4f", model_name, prediction)
    return {"model": model_name, "prediction-average-purchase-value": prediction}


@app.get("/health")
def health() -> dict[str, list[str]]:
    return {"loaded_models": sorted(MODELS)}


@app.post("/predict/random-forest-100")
def predict_random_forest_100(features: RetailFeatures) -> dict[str, float | str]:
    return predict("random_forest_100", features)


@app.post("/predict/random-forest-200")
def predict_random_forest_200(features: RetailFeatures) -> dict[str, float | str]:
    return predict("random_forest_200", features)


@app.post("/predict/gradient-boosting-100")
def predict_gradient_boosting_100(features: RetailFeatures) -> dict[str, float | str]:
    return predict("gradient_boosting_100", features)


@app.post("/predict/gradient-boosting-200")
def predict_gradient_boosting_200(features: RetailFeatures) -> dict[str, float | str]:
    return predict("gradient_boosting_200", features)
