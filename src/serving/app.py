"""FastAPI model serving for NYC Taxi fare prediction."""

import logging
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import List

import fsspec
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from prometheus_client import Gauge
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import config
from src.features.core import TemporalFeatureEngineer, FEATURE_COLS
from src.monitoring.drift import generate_drift_report

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nyc-taxi-api")

MODEL_PATH = config.model_path
MAX_BATCH_SIZE = config.max_batch_size

model = None
inference_mode = "sklearn"
feature_engineer = TemporalFeatureEngineer()

# Prometheus gauge: 1 when model is loaded and serving, 0 otherwise.
model_loaded_gauge = Gauge(
    "model_loaded_bool", "1 if model is loaded and serving, 0 otherwise"
)

PREDICTION_LOG_PATH = "data/predictions"


def _attempt_model_load(path: str):
    """Load the model once. Returns (model_obj, mode_str) or raises."""
    if path.endswith(".onnx"):
        if ort is None:
            logger.warning(
                "ONNX model path set but `onnxruntime` is missing. "
                "Install it to use fast inference."
            )
        else:
            logger.info(f"Loading ONNX model from {path}...")
            with fsspec.open(path, "rb") as f:
                model_bytes = f.read()
            session = ort.InferenceSession(model_bytes)
            logger.info("✅ Loaded ONNX model (mode: onnx)")
            return session, "onnx"

    logger.info(f"Loading sklearn pipeline from {path}...")
    with fsspec.open(path, "rb") as f:
        pipeline = joblib.load(f)
    logger.info("✅ Loaded sklearn pipeline (mode: sklearn)")
    return pipeline, "sklearn"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: load model with retry on startup."""
    global model, inference_mode

    for attempt in range(3):
        try:
            model, inference_mode = _attempt_model_load(MODEL_PATH)
            break
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt  # 1s, then 2s
                logger.warning(
                    f"Model load attempt {attempt + 1}/3 failed: {e}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"❌ Model load failed after 3 attempts: {e}\n"
                    "Tip: Run `make train` or update MODEL_PATH env var."
                )

    model_loaded_gauge.set(1 if model is not None else 0)
    warmup_model()
    yield


app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts green taxi fare amount given trip features.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


def _log_prediction(features: dict, predicted_fare: float):
    """Append one prediction row to a date-partitioned Parquet dataset."""
    row = {**features, "predicted_fare": predicted_fare, "log_date": str(date.today())}
    table = pa.Table.from_pydict({k: [v] for k, v in row.items()})
    pq.write_to_dataset(table, root_path=PREDICTION_LOG_PATH, partition_cols=["log_date"])


def _log_predictions_batch(features_list: list, fares: list):
    """Batch-write all prediction rows in a single Parquet write."""
    today = str(date.today())
    rows = [
        {**feat, "predicted_fare": fare, "log_date": today}
        for feat, fare in zip(features_list, fares)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_to_dataset(table, root_path=PREDICTION_LOG_PATH, partition_cols=["log_date"])


def warmup_model():
    """Run one sample inference to warm up latency-critical pathways."""
    if model is None:
        return

    sample = pd.DataFrame(
        [
            {
                "trip_distance": 1.0,
                "passenger_count": 1,
                "PULocationID": 1,
                "DOLocationID": 1,
                "pickup_hour": 12,
                "pickup_dayofweek": 2,
                "pickup_month": 1,
                "RatecodeID": 1,
            }
        ]
    )

    try:
        if inference_mode == "onnx":
            x = feature_engineer.transform(sample)
            x = x[FEATURE_COLS].astype("float32").to_numpy()
            input_name = model.get_inputs()[0].name
            label_name = model.get_outputs()[0].name
            _ = model.run([label_name], {input_name: x})
        else:
            model.predict(sample)
        logger.info("Warmup inference completed.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "inference_mode": inference_mode,
    }


@app.get("/monitoring/drift", response_class=HTMLResponse)
async def drift_report():
    """Generate and return an Evidently data drift report as HTML."""
    try:
        report_path = generate_drift_report()
        with open(report_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Drift report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift report failed: {e}")


class TripFeatures(BaseModel):
    trip_distance: float = Field(..., gt=0, example=2.5, description="Miles")
    passenger_count: int = Field(..., ge=1, le=6, example=1)
    PULocationID: int = Field(..., example=236, description="TLC pickup zone ID")
    DOLocationID: int = Field(..., example=237, description="TLC dropoff zone ID")
    pickup_hour: int = Field(..., ge=0, le=23, example=14)
    pickup_dayofweek: int = Field(
        ..., ge=0, le=6, example=2, description="0=Monday, 6=Sunday"
    )
    pickup_month: int = Field(..., ge=1, le=12, example=3)
    RatecodeID: int = Field(
        1, ge=1, le=6, example=1, description="1=Standard, 2=JFK, 3=Newark"
    )


class PredictionResponse(BaseModel):
    predicted_fare: float
    currency: str = "USD"


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    currency: str = "USD"


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: TripFeatures, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    feature_df = pd.DataFrame([features.model_dump()])

    if inference_mode == "onnx":
        feature_df = feature_engineer.transform(feature_df)
        inputs = feature_df[FEATURE_COLS].astype("float32").to_numpy()
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        res = model.run([label_name], {input_name: inputs})
        predicted_fare = float(res[0][0])
    else:
        predicted_fare = float(model.predict(feature_df)[0])

    predicted_fare = max(2.5, round(predicted_fare, 2))

    # Fire-and-forget: logging runs after the response is sent
    background_tasks.add_task(_log_prediction, features.model_dump(), predicted_fare)

    return PredictionResponse(predicted_fare=predicted_fare)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
    features_list: List[TripFeatures], background_tasks: BackgroundTasks
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    if not features_list:
        return BatchPredictionResponse(predictions=[])

    if len(features_list) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size exceeds limit ({MAX_BATCH_SIZE}).",
        )

    feature_df = pd.DataFrame([f.model_dump() for f in features_list])

    if inference_mode == "onnx":
        feature_df = feature_engineer.transform(feature_df)
        inputs = feature_df[FEATURE_COLS].astype("float32").to_numpy()
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        results = model.run([label_name], {input_name: inputs})[0]
        predicted_fares = [float(x) for x in results.flatten()]
    else:
        predicted_fares = model.predict(feature_df).tolist()

    processed_fares = [max(2.5, round(fare, 2)) for fare in predicted_fares]

    # Single batch Parquet write, fire-and-forget
    background_tasks.add_task(
        _log_predictions_batch,
        [f.model_dump() for f in features_list],
        processed_fares,
    )

    return BatchPredictionResponse(predictions=processed_fares)
