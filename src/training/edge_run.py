"""
Standalone Edge Inference Script for Robotic Platforms.

This script demonstrates how to run the NYC Taxi model on an edge device
without the overhead of a web server or Docker container.

Prerequisites:
    pip install onnxruntime pandas numpy

Usage:
    python -m src.inference.edge_run --model model.onnx
"""

import argparse
import logging
import time
import sys
import numpy as np
import onnxruntime as ort
import pandas as pd

# Ensure we can import the project modules
sys.path.append(".")
from src.features.core import (  # noqa: E402
    FEATURE_COLS,
    TemporalFeatureEngineer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("EdgeInfer")


def load_model_session(model_path: str):
    """Loads the ONNX model with the best available hardware accelerator."""
    # ONNX Runtime automatically detects CUDA (NVIDIA), CoreML (Apple), or CPU
    providers = ort.get_available_providers()
    logger.info(f"Loading from {model_path} with providers: {providers}")
    session = ort.InferenceSession(model_path, providers=providers)
    return session


def predict_fare(session, feature_engineer, raw_features: dict):
    """
    Hybrid Inference Step:
    1. Python CPU: Lightweight feature engineering (dates, cyclics).
    2. ONNX Runtime: Heavy model inference (XGBoost).
    """
    t0 = time.time()

    # 1. Feature Engineering
    # Convert raw dictionary to DataFrame
    df = pd.DataFrame([raw_features])
    # Apply transformations (same logic as training)
    df_processed = feature_engineer.transform(df)

    # 2. Prepare ONNX Input
    # XGBoost ONNX typically expects Float32 inputs
    inputs = df_processed[FEATURE_COLS].astype(np.float32).to_numpy()

    # 3. Inference
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    result = session.run([label_name], {input_name: inputs})
    predicted_fare = float(result[0][0])

    # Apply business logic (minimum fare)
    final_fare = max(2.5, round(predicted_fare, 2))

    latency_ms = (time.time() - t0) * 1000
    return final_fare, latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="model.onnx", help="Path to ONNX model"
    )
    args = parser.parse_args()

    # Initialize
    session = load_model_session(args.model)
    engineer = TemporalFeatureEngineer()

    # Simulated Robot Stream (Example Input)
    # In production, this loop would consume from a ROS topic or sensor buffer
    dummy_input = {
        "trip_distance": 3.2,
        "passenger_count": 2,
        "PULocationID": 142,  # Lincoln Square
        "DOLocationID": 239,  # Upper West Side
        "pickup_hour": 18,
        "pickup_dayofweek": 4,  # Friday
        "pickup_month": 5,
        "RatecodeID": 1,
    }

    logger.info("Starting inference loop...")
    for _ in range(5):
        fare, ms = predict_fare(session, engineer, dummy_input)
        logger.info(
            f"Trip: {dummy_input['trip_distance']}mi | "
            f"Pred: ${fare:.2f} | Latency: {ms:.2f}ms"
        )
