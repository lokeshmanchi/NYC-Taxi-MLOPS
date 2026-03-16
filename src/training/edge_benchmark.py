"""
Edge inference benchmark — real-time NYC taxi trip stream simulation.

Patches edge_run.py with a mock ONNX session (no model file required),
then replays trips sampled from the real parquet files in data/ to measure:

  - p50 / p95 / p99 latency
  - Cache hit rate and effective throughput
  - SLA breach rate

Compares two paths:
  1. Baseline  — pandas DataFrame + TemporalFeatureEngineer per call (old approach)
  2. Optimised — direct NumPy + LRU cache (edge_run.py)

Usage:
    python -m src.training.edge_benchmark           # 10 000 trips
    python -m src.training.edge_benchmark 50000     # custom N
"""

import glob
import os
import random
import statistics
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Mock ONNX session — mimics INT8 XGBoost inference at ~0.15 ms CPU latency
# ---------------------------------------------------------------------------

_WEIGHTS = np.array(
    [2.1, 0.3, 0.01, -0.01, 0.05, 0.1, 0.2, 0.05, 0.15, 0.08, 0.04],
    dtype=np.float32,
)
_BIAS = 3.5


class _MockInput:
    name = "features"


class _MockOutput:
    name = "fare"


class MockONNXSession:
    """
    Simulates a quantised ONNX session. Uses a fixed linear model so results
    are deterministic. Adds ~0.15 ms sleep to approximate kernel dispatch
    overhead on an Intel NUC / ARM Cortex-A78 (typical edge CPU).
    """

    def get_inputs(self):
        return [_MockInput()]

    def get_outputs(self):
        return [_MockOutput()]

    def run(self, output_names, input_feed):
        features = input_feed["features"]           # (1, 11) float32
        fare = float(features[0] @ _WEIGHTS) + _BIAS
        time.sleep(0.00015)                         # ~0.15 ms kernel overhead
        return [np.array([fare], dtype=np.float32)]


# ---------------------------------------------------------------------------
# Real NYC taxi trip data loader
# ---------------------------------------------------------------------------

_DATA_COLS = [
    "lpep_pickup_datetime",
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]

_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data"
)


def _load_real_trips() -> list[dict]:
    """
    Read all parquet files from data/ and return a list of feature dicts.
    Applies the same outlier filters as the training pipeline so the
    benchmark reflects the distribution the model was trained on.
    """
    files = sorted(glob.glob(os.path.join(_DATA_DIR, "green_tripdata_*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No green_tripdata_*.parquet files found in {_DATA_DIR}. "
            "Add trip data files to data/ before running the benchmark."
        )

    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=_DATA_COLS)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Derive temporal features from pickup datetime
    dt = pd.to_datetime(df["lpep_pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    df.drop(columns=["lpep_pickup_datetime"], inplace=True)

    # Apply the same outlier + null filters used in training
    df = df[df["trip_distance"].between(0.1, 60)]
    df["passenger_count"] = df["passenger_count"].fillna(1).clip(1, 6).astype(int)
    df["RatecodeID"] = df["RatecodeID"].fillna(1).astype(int)
    df["trip_distance"] = df["trip_distance"].round(1)

    print(f"  Loaded {len(df):,} real trips from {len(files)} file(s)")
    return df.to_dict("records")


# Load once at import time; generate_trip_stream samples from this pool.
random.seed(42)
_REAL_TRIPS = _load_real_trips()


def generate_trip_stream(n: int) -> list[dict]:
    """
    Sample n trips uniformly from the full real-data pool.
    Because the pool contains all raw trips (not deduplicated), the repeat
    rate naturally reflects actual NYC taxi route repetition patterns.
    """
    return random.choices(_REAL_TRIPS, k=n)


# ---------------------------------------------------------------------------
# Baseline path — original approach (pandas DataFrame per call)
# ---------------------------------------------------------------------------

from src.features.core import FEATURE_COLS, TemporalFeatureEngineer  # noqa: E402

_baseline_engineer = TemporalFeatureEngineer()
_baseline_session = MockONNXSession()
_baseline_input = _baseline_session.get_inputs()[0].name
_baseline_output = _baseline_session.get_outputs()[0].name


def baseline_predict(raw_features: dict) -> float:
    """Original approach: pandas DataFrame construction + TemporalFeatureEngineer."""
    df = pd.DataFrame([raw_features])
    df = _baseline_engineer.transform(df)
    inputs = df[FEATURE_COLS].astype("float32").to_numpy()
    result = _baseline_session.run([_baseline_output], {_baseline_input: inputs})
    return max(2.50, round(float(result[0][0]), 2))


# ---------------------------------------------------------------------------
# Optimised path — edge_run.py (NumPy + LRU cache)
# ---------------------------------------------------------------------------

import src.training.edge_run as edge  # noqa: E402

# Inject mock session — bypasses file loading
_mock_session = MockONNXSession()
edge._session = _mock_session
edge._input_name = _mock_session.get_inputs()[0].name
edge._label_name = _mock_session.get_outputs()[0].name
edge._run_cached.cache_clear()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(n_trips: int = 10_000):
    trips = generate_trip_stream(n_trips)
    unique = len({
        (t["PULocationID"], t["DOLocationID"], t["pickup_hour"],
         t["trip_distance"], t["RatecodeID"])
        for t in trips
    })

    print(f"\n{'='*60}")
    print(f"  NYC Taxi Edge Inference Benchmark  ({n_trips:,} trips)")
    print(f"  Source: real parquet data  ({len(_REAL_TRIPS):,} rows in pool)")
    print(f"  Unique route signatures: {unique:,} / {n_trips:,}")
    print(f"{'='*60}\n")

    # ── Baseline ────────────────────────────────────────────────────────────
    print("[ 1/2 ] Baseline (pandas + TemporalFeatureEngineer)...")
    baseline_latencies = []
    for trip in trips:
        t0 = time.perf_counter()
        baseline_predict(trip)
        baseline_latencies.append((time.perf_counter() - t0) * 1000)

    b_p50 = statistics.median(baseline_latencies)
    b_p95 = sorted(baseline_latencies)[int(0.95 * n_trips)]
    b_p99 = sorted(baseline_latencies)[int(0.99 * n_trips)]
    b_total = sum(baseline_latencies)
    b_tp = n_trips / (b_total / 1000)

    print(f"  p50={b_p50:.3f}ms  p95={b_p95:.3f}ms  p99={b_p99:.3f}ms")
    print(f"  throughput={b_tp:,.0f} trips/s\n")

    # ── Optimised ───────────────────────────────────────────────────────────
    print("[ 2/2 ] Optimised (NumPy + LRU cache + SLA guard)...")
    edge._run_cached.cache_clear()
    opt_latencies = []
    sla_breaches = 0
    for trip in trips:
        fare, ms = edge.predict_fare(trip)
        opt_latencies.append(ms)
        if ms > edge.LATENCY_BUDGET_MS:
            sla_breaches += 1

    cache_info = edge._run_cached.cache_info()
    hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) * 100

    o_p50 = statistics.median(opt_latencies)
    o_p95 = sorted(opt_latencies)[int(0.95 * n_trips)]
    o_p99 = sorted(opt_latencies)[int(0.99 * n_trips)]
    o_total = sum(opt_latencies)
    o_tp = n_trips / (o_total / 1000)

    print(f"  p50={o_p50:.3f}ms  p95={o_p95:.3f}ms  p99={o_p99:.3f}ms")
    print(f"  throughput={o_tp:,.0f} trips/s")
    print(f"  cache hits={cache_info.hits:,}  misses={cache_info.misses:,}  "
          f"hit_rate={hit_rate:.1f}%")
    print(f"  SLA breaches (>{edge.LATENCY_BUDGET_MS}ms): {sla_breaches} "
          f"({sla_breaches/n_trips*100:.2f}%)\n")

    # ── Comparison ──────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("  Improvement Summary")
    print(f"{'='*60}")
    _row("p50 latency", b_p50, o_p50)
    _row("p95 latency", b_p95, o_p95)
    _row("p99 latency", b_p99, o_p99)
    _row("Throughput", b_tp, o_tp, higher_is_better=True, unit="trips/s")
    print(f"\n  Cache hit rate  : {hit_rate:.1f}%  "
          f"({cache_info.hits:,} of {n_trips:,} calls skipped ONNX entirely)")
    print(f"  SLA compliance  : {(1 - sla_breaches / n_trips) * 100:.2f}%")
    print(f"{'='*60}\n")


def _row(label: str, baseline: float, optimised: float,
         higher_is_better: bool = False, unit: str = "ms"):
    if higher_is_better:
        speedup = optimised / baseline
        arrow = "▲" if speedup > 1 else "▼"
    else:
        speedup = baseline / optimised
        arrow = "▲" if speedup > 1 else "▼"
    print(f"  {label:<18}: {baseline:>9.2f} {unit}  →  {optimised:>9.2f} {unit}  "
          f"  {arrow} {speedup:.1f}x")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    run_benchmark(n)
