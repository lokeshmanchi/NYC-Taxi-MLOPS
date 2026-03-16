"""FastAPI model serving for NYC Taxi fare prediction."""

import logging
from typing import List
from contextlib import asynccontextmanager
from datetime import date

import fsspec
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Import feature logic for the Hybrid ONNX approach
from src.config import config
from src.features.core import TemporalFeatureEngineer, FEATURE_COLS
from src.monitoring.drift import generate_drift_report

# Try importing ONNX Runtime for edge inference
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nyc-taxi-api")

MODEL_PATH = config.model_path
MAX_BATCH_SIZE = config.max_batch_size

model = None
inference_mode = "sklearn"  # Options: 'sklearn' or 'onnx'
feature_engineer = TemporalFeatureEngineer()  # Used for ONNX hybrid mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (load model on startup)."""
    global model, inference_mode
    try:
        # 1. Fault Tolerant Loading: Check for ONNX first (High Performance / Edge)
        if MODEL_PATH.endswith(".onnx"):
            if ort is None:
                logger.warning(
                    "ONNX model found but `onnxruntime` is missing. "
                    "Install it to use fast inference."
                )
            else:
                logger.info(f"Loading ONNX model from {MODEL_PATH}...")
                # Load bytes using fsspec (supports s3://, gs://, local)
                with fsspec.open(MODEL_PATH, "rb") as f:
                    model_bytes = f.read()

                model = ort.InferenceSession(model_bytes)
                inference_mode = "onnx"
                logger.info(f"✅ Loaded ONNX model (Mode: {inference_mode})")

        # 2. Fallback: Load Scikit-Learn Pipeline (Standard Cloud/Dev)
        if model is None:
            logger.info(f"Loading Pickle pipeline from {MODEL_PATH}...")
            with fsspec.open(MODEL_PATH, "rb") as f:
                model = joblib.load(f)
            inference_mode = "sklearn"
            logger.info(f"✅ Loaded Sklearn pipeline (Mode: {inference_mode})")

    except Exception as e:
        logger.error(f"❌ Critical Error: Load failed {MODEL_PATH}: {e}")
        logger.info("Tip: Run `make train` or update MODEL_PATH env var.")

    warmup_model()
    yield
    # Cleanup code can go here if needed


app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts green taxi fare amount given trip features.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

PREDICTION_LOG_PATH = "data/predictions"


def _log_prediction(features: dict, predicted_fare: float):
    """Append one prediction row to a date-partitioned Parquet dataset."""
    row = {**features, "predicted_fare": predicted_fare, "log_date": str(date.today())}
    table = pa.Table.from_pydict({k: [v] for k, v in row.items()})
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
    # Note: is_weekend is no longer needed. The model pipeline will generate it
    # from the raw features below.
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
async def predict(features: TripFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    # Create a DataFrame from the input features. The column names must match
    # what the training pipeline expects.
    feature_df = pd.DataFrame([features.model_dump()])

    if inference_mode == "onnx":
        # Hybrid Approach: Python Feature Eng -> ONNX Inference
        # 1. Apply features (weekend, sin/cos) using the imported class
        feature_df = feature_engineer.transform(feature_df)
        # 2. Prepare float32 numpy array for ONNX
        inputs = feature_df[FEATURE_COLS].astype("float32").to_numpy()
        # 3. Run inference
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        res = model.run([label_name], {input_name: inputs})
        predicted_fare = float(res[0][0])
    else:
        # Sklearn Approach: Pipeline handles everything
        predicted_fare = float(model.predict(feature_df)[0])

    # Enforce business rule (minimum fare)
    predicted_fare = max(2.5, round(predicted_fare, 2))

    try:
        _log_prediction(features.model_dump(), predicted_fare)
    except Exception as e:
        logger.warning(f"Prediction logging failed: {e}")

    return PredictionResponse(predicted_fare=predicted_fare)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(features_list: List[TripFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    if not features_list:
        return BatchPredictionResponse(predictions=[])

    if len(features_list) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size exceeds limit ({MAX_BATCH_SIZE}).",
        )

    # Create a DataFrame from the list of input features
    feature_df = pd.DataFrame([f.model_dump() for f in features_list])

    if inference_mode == "onnx":
        # Hybrid Batch Inference
        feature_df = feature_engineer.transform(feature_df)
        inputs = feature_df[FEATURE_COLS].astype("float32").to_numpy()

        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        # Output might be shape (N, 1) or (N,) depending on export
        results = model.run([label_name], {input_name: inputs})[0]
        predicted_fares = [float(x) for x in results.flatten()]
    else:
        predicted_fares = model.predict(feature_df).tolist()

    # Apply business rule to each prediction
    processed_fares = [max(2.5, round(fare, 2)) for fare in predicted_fares]

    try:
        for feat, fare in zip(features_list, processed_fares):
            _log_prediction(feat.model_dump(), fare)
    except Exception as e:
        logger.warning(f"Batch prediction logging failed: {e}")

    return BatchPredictionResponse(predictions=processed_fares)
