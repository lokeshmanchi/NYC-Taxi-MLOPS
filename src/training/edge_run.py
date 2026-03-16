"""
Standalone Edge Inference Script.

Optimised for sub-millisecond latency on NVIDIA Jetson / Intel NUC / CPU-only nodes.
Improvements over the baseline:
  - TensorRT execution provider (NVIDIA Jetson) with CUDA / CPU fallback
  - ORT_ENABLE_ALL graph optimisation at load time
  - Warm-up call on startup to trigger ONNX Runtime JIT
  - Direct NumPy feature computation — no pandas DataFrame overhead
  - LRU cache for frequently repeated route queries
  - SLA-bounded inference with distance-based heuristic fallback
  - Background thread for zero-downtime OTA model hot-swap

Prerequisites:
    pip install onnxruntime numpy

Usage:
    python -m src.training.edge_run --model models/model_int8.onnx
"""

import argparse
import logging
import math
import os
import threading
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import onnxruntime as ort

from src.features.core import FEATURE_COLS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("EdgeInfer")

# --- Configuration -----------------------------------------------------------
LATENCY_BUDGET_MS = 10.0        # SLA: breach triggers heuristic fallback
OTA_POLL_INTERVAL_S = 300       # Check for updated model file every 5 minutes
LRU_CACHE_SIZE = 1024           # Cache most frequent routes to skip ONNX.

# Module-level session state (safe for CPython GIL; hot-swapped atomically)
_session: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None
_label_name: Optional[str] = None


# --- Model loading -----------------------------------------------------------

def load_model_session(model_path: str) -> ort.InferenceSession:
    """
    Load ONNX model with the best available execution provider.

    Provider priority:
      1. TensorrtExecutionProvider — fastest on NVIDIA Jetson (fp16 enabled)
      2. CUDAExecutionProvider      — generic GPU fallback
      3. CPUExecutionProvider       — always available

    ORT_ENABLE_ALL performs constant folding, node fusion, and shape inference
    once at load time so none of that work happens per-inference.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    active = session.get_providers()
    logger.info(f"Loaded {model_path} | active providers: {active}")
    return session


def _init_session(model_path: str):
    """Load, warm up, and register the session as the active global."""
    global _session, _input_name, _label_name
    sess = load_model_session(model_path)
    _warmup(sess)
    _run_cached.cache_clear()   # invalidate stale cache entries on hot-swap
    _session = sess
    _input_name = sess.get_inputs()[0].name
    _label_name = sess.get_outputs()[0].name


def _warmup(session: ort.InferenceSession):
    """
    Run one dummy inference to trigger ONNX Runtime's lazy JIT compilation.
    Without this, the very first real prediction spikes 50–200 ms.
    """
    dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
    iname = session.get_inputs()[0].name
    session.run(None, {iname: dummy})
    logger.info("Warmup inference complete.")


# --- OTA hot-swap ------------------------------------------------------------

def _watch_model(model_path: str):
    """
    Background daemon thread: polls model_path every OTA_POLL_INTERVAL_S seconds.
    If the file mtime changes, reloads the session in-place — zero downtime.
    """
    last_mtime = os.path.getmtime(model_path)
    while True:
        time.sleep(OTA_POLL_INTERVAL_S)
        try:
            mtime = os.path.getmtime(model_path)
            if mtime > last_mtime:
                logger.info(f"Model update detected at {model_path}. Hot-swapping...")
                _init_session(model_path)
                last_mtime = mtime
                logger.info("Model hot-swap complete.")
        except Exception as e:
            logger.warning(f"OTA watch error (non-fatal): {e}")


# --- Inference ---------------------------------------------------------------

@lru_cache(maxsize=LRU_CACHE_SIZE)
def _run_cached(
    trip_distance: float,
    passenger_count: int,
    PULocationID: int,
    DOLocationID: int,
    pickup_hour: int,
    pickup_dayofweek: int,
    pickup_month: int,
    RatecodeID: int,
) -> float:
    """
    Cached inference kernel — all args are primitives so they are hashable.

    Derives the 3 temporal features inline (no pandas / TemporalFeatureEngineer
    overhead) and runs the ONNX session. Result is cached by input signature.
    Cache is cleared automatically on OTA model hot-swap.
    """
    features = np.array([[
        trip_distance,
        passenger_count,
        PULocationID,
        DOLocationID,
        pickup_hour,
        pickup_dayofweek,
        pickup_month,
        RatecodeID,
        int(pickup_dayofweek >= 5),                          # is_weekend
        math.sin(2 * math.pi * pickup_hour / 24),            # hour_sin
        math.cos(2 * math.pi * pickup_hour / 24),            # hour_cos
    ]], dtype=np.float32)
    result = _session.run([_label_name], {_input_name: features})
    return float(result[0][0])


def predict_fare(raw_features: dict) -> tuple:
    """
    Public inference entry point.

    Returns (predicted_fare, latency_ms).
    Falls back to a distance-based heuristic if inference exceeds LATENCY_BUDGET_MS.
    """
    t0 = time.perf_counter()

    raw_fare = _run_cached(
        trip_distance=float(raw_features["trip_distance"]),
        passenger_count=int(raw_features["passenger_count"]),
        PULocationID=int(raw_features["PULocationID"]),
        DOLocationID=int(raw_features["DOLocationID"]),
        pickup_hour=int(raw_features["pickup_hour"]),
        pickup_dayofweek=int(raw_features["pickup_dayofweek"]),
        pickup_month=int(raw_features["pickup_month"]),
        RatecodeID=int(raw_features["RatecodeID"]),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if elapsed_ms > LATENCY_BUDGET_MS:
        logger.warning(
            f"SLA breach: {elapsed_ms:.1f}ms > {LATENCY_BUDGET_MS}ms — using fallback heuristic"
        )
        raw_fare = 3.50 + raw_features["trip_distance"] * 2.50

    final_fare = max(2.50, round(raw_fare, 2))
    return final_fare, elapsed_ms


# --- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="models/model_int8.onnx",
        help="Path to ONNX model (prefer INT8 quantised for edge)"
    )
    args = parser.parse_args()

    _init_session(args.model)

    # Start OTA hot-swap watcher in the background
    threading.Thread(
        target=_watch_model, args=(args.model,), daemon=True
    ).start()

    # Simulated trip stream — in production consume from a ROS topic or sensor buffer
    dummy_input = {
        "trip_distance": 3.2,
        "passenger_count": 2,
        "PULocationID": 142,
        "DOLocationID": 239,
        "pickup_hour": 18,
        "pickup_dayofweek": 4,
        "pickup_month": 5,
        "RatecodeID": 1,
    }

    logger.info("Starting inference loop (first call: cache miss; subsequent: cache hit)...")
    for i in range(6):
        fare, ms = predict_fare(dummy_input)
        cache_info = _run_cached.cache_info()
        logger.info(
            f"[{i}] Trip {dummy_input['trip_distance']}mi | "
            f"${fare:.2f} | {ms:.3f}ms | "
            f"cache hits={cache_info.hits} misses={cache_info.misses}"
        )
