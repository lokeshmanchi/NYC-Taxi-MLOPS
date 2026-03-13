"""FastAPI model serving for NYC Taxi fare prediction."""

import os
from typing import List

import fsspec
import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts green taxi fare amount given trip features.",
    version="1.0.0",
)

model = None

# At petabyte scale, the model artifact is not just the estimator (e.g., XGBoost)
# but a full scikit-learn Pipeline that includes feature transformation steps.
# This prevents training-serving skew by ensuring transformations are identical.

@app.on_event("startup")
async def load_model():
    global model
    try:
        # In a production environment with multiple model frameworks, consider
        # standardizing on a format like ONNX (Open Neural Network Exchange).
        # The scikit-learn pipeline could be converted to ONNX (e.g., using skl2onnx)
        # and served with a high-performance runtime like ONNX Runtime.
        # This decouples the serving environment from the training framework.
        # For now, we load the native scikit-learn object.
        with fsspec.open(MODEL_PATH, "rb") as f:
            model = joblib.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load model from {MODEL_PATH}: {e}. Run `make train` first.")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


class TripFeatures(BaseModel):
    trip_distance: float = Field(..., gt=0, example=2.5, description="Miles")
    passenger_count: int = Field(..., ge=1, le=6, example=1)
    PULocationID: int = Field(..., example=236, description="TLC pickup zone ID")
    DOLocationID: int = Field(..., example=237, description="TLC dropoff zone ID")
    # Note: is_weekend is no longer needed. The model pipeline will generate it
    # from the raw features below.
    pickup_hour: int = Field(..., ge=0, le=23, example=14)
    pickup_dayofweek: int = Field(..., ge=0, le=6, example=2, description="0=Monday, 6=Sunday")
    pickup_month: int = Field(..., ge=1, le=12, example=3)
    RatecodeID: int = Field(1, ge=1, le=6, example=1, description="1=Standard, 2=JFK, 3=Newark")


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
    feature_df = pd.DataFrame([features.dict()])
    
    # The pipeline handles all transformations (e.g., cyclical features, one-hot encoding)
    # and then passes the result to the model for prediction.
    predicted_fare = float(model.predict(feature_df)[0])
    
    # Enforce business rule (minimum fare)
    predicted_fare = max(2.5, round(predicted_fare, 2))

    return PredictionResponse(predicted_fare=predicted_fare)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(features_list: List[TripFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    if not features_list:
        return BatchPredictionResponse(predictions=[])

    # Create a DataFrame from the list of input features
    feature_df = pd.DataFrame([f.dict() for f in features_list])

    # Get predictions for the entire batch
    predicted_fares = model.predict(feature_df).tolist()

    # Apply business rule to each prediction
    processed_fares = [max(2.5, round(fare, 2)) for fare in predicted_fares]

    return BatchPredictionResponse(predictions=processed_fares)
