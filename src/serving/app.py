"""FastAPI model serving for NYC Taxi fare prediction."""

import logging
import os
from typing import List
from contextlib import asynccontextmanager

import fsspec
import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

# Import feature logic for the Hybrid ONNX approach
from src.features.transform import TemporalFeatureEngineer, FEATURE_COLS

# Try importing ONNX Runtime for edge inference
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nyc-taxi-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

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

    except (FileNotFoundError, Exception) as e:
        logger.error(f"❌ Critical Error: Load failed {MODEL_PATH}: {e}")
        logger.info("Tip: Run `make train` or update MODEL_PATH env var.")

    yield
    # Cleanup code can go here if needed


app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts green taxi fare amount given trip features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "inference_mode": inference_mode,
    }


class TripFeatures(BaseModel):
    trip_distance: float = Field(..., gt=0, example=2.5, description="Miles")
    passenger_count: int = Field(..., ge=1, le=6, example=1)
    PULocationID: int = Field(
        ..., example=236, description="TLC pickup zone ID"
    )
    DOLocationID: int = Field(
        ..., example=237, description="TLC dropoff zone ID"
    )
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
        raise HTTPException(
            status_code=503, detail="Model not loaded. Run training first."
        )

    # Create a DataFrame from the input features. The column names must match
    # what the training pipeline expects.
    feature_df = pd.DataFrame([features.dict()])

    if inference_mode == "onnx":
        # Hybrid Approach: Python Feature Eng -> ONNX Inference
        # 1. Apply features (weekend, sin/cos) using the imported class
        feature_df = feature_engineer.transform(feature_df)
        # 2. Prepare float32 numpy array for ONNX
        inputs = feature_df[FEATURE_COLS].astype("float32").to_numpy()
        # 3. Run inference
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        predicted_fare = float(
            model.run([label_name], {input_name: inputs})[0][0]
        )
    else:
        # Sklearn Approach: Pipeline handles everything
        predicted_fare = float(model.predict(feature_df)[0])

    # Enforce business rule (minimum fare)
    predicted_fare = max(2.5, round(predicted_fare, 2))

    return PredictionResponse(predicted_fare=predicted_fare)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(features_list: List[TripFeatures]):
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Run training first."
        )

    if not features_list:
        return BatchPredictionResponse(predictions=[])

    # Create a DataFrame from the list of input features
    feature_df = pd.DataFrame([f.dict() for f in features_list])

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

    return BatchPredictionResponse(predictions=processed_fares)
