# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLOps pipeline for NYC Green Taxi fare prediction (Jan–May 2025, `data/`). Trains an XGBoost regression model wrapped in a scikit-learn Pipeline that handles feature engineering. Data loading uses Dask for scalability beyond RAM.

## Stack

| Service | Technology | Local URL |
|---------|-----------|-----------|
| Experiment tracking | MLflow | http://localhost:5000 |
| Pipeline orchestration | Prefect 3 | http://localhost:4200 |
| Model serving | FastAPI + Uvicorn | http://localhost:8000 |
| Visualization dashboard | Streamlit | http://localhost:8501 |

## Common Commands

```bash
make build          # Build Docker images
make up             # Start MLflow, Prefect, API, Streamlit
make train          # Run training pipeline (builds trainer image, runs it)
make logs           # Tail all service logs
make down           # Stop services
make clean          # Stop + remove all volumes (full reset)
```

After `make train`, the API must be restarted to load the new model artifact:
```bash
docker compose restart api
```

### Local development (outside Docker)
```bash
pip install -r requirements.txt

python -m src.training.train          # Train locally
python -m src.features.transform      # Materialize processed data to data/processed/
uvicorn src.serving.app:app --reload  # Serve API on :8000
streamlit run src/ui/Home.py          # Run dashboard on :8501
```

### Testing & linting
```bash
make test                                      # Run full pytest suite
pytest src/features/test_features.py -v        # Run a single test file
make format                                    # Run pre-commit hooks (flake8)
pre-commit run --all-files                     # Lint all files manually
```

Flake8 is configured in `.flake8`: max-line-length=120, ignores W292/W293.

## Architecture

### Configuration — `src/config.py`

Central `pydantic.BaseSettings` class (`Settings`) with env var overrides. Key groups:
- **Paths**: `model_path` (default `models/model.pkl`), `model_output_path`, `data_path`, `processed_data_path`
- **MLflow**: `mlflow_tracking_uri`, `experiment_name`
- **DDP**: `ddp_backend`, `rank`/`local_rank`/`world_size`, `batch_size`, `epochs`, `log_step_interval`
- **Serving**: `max_batch_size` (default 256), `api_url`

### Feature Engineering — two-stage design

`src/features/transform.py` defines two feature sets:

- **`BASE_FEATURE_COLS`** (8): raw/cleaned columns read from parquet — `trip_distance`, `passenger_count`, `PULocationID`, `DOLocationID`, `pickup_hour`, `pickup_dayofweek`, `pickup_month`, `RatecodeID`
- **`FEATURE_COLS`** (11): `BASE_FEATURE_COLS` + derived `is_weekend`, `hour_sin`, `hour_cos`

The derived features are **never stored** — they are always generated on-the-fly by `TemporalFeatureEngineer`, a scikit-learn `TransformerMixin`. This prevents training-serving skew.

`load_data()` returns a **lazy Dask DataFrame** (supports `s3://`, `gs://` paths in addition to local). Training calls `.compute()` to materialize to Pandas before fitting. `save_processed_data()` materializes the Dask graph to `data/processed/` Parquet.

### Training pipeline — `src/training/train.py`

Builds a `sklearn.pipeline.Pipeline`:
```
[("features", TemporalFeatureEngineer()), ("model", XGBRegressor(...))]
```
The whole pipeline is logged to MLflow via `mlflow.sklearn.log_model` (artifact path `model`, registered as `nyc-taxi-regressor`). **No `joblib.dump` — the model artifact lives only in MLflow.**

### Model Serving — `src/serving/app.py`

Hybrid ONNX/sklearn serving via a lifespan context manager:
1. On startup, attempts to load an ONNX model first (if available at `model_output_path`)
2. Falls back to loading the sklearn pipeline via `fsspec.open(MODEL_PATH)` (supports cloud URIs)
3. Endpoints: `GET /health`, `POST /predict` (single), `POST /predict_batch` (up to `max_batch_size`)
- Accepts `BASE_FEATURE_COLS` as input; the pipeline/ONNX session handles all transforms.

**Important:** `MODEL_PATH` defaults to `models/model.pkl`. After retraining, either download the MLflow artifact to that path or update `MODEL_PATH` to `runs:/<run_id>/model`.

### Edge Inference — `src/training/edge_run.py`

Standalone ONNX inference for edge/offline devices:
- Auto-selects hardware accelerator (CUDA → CoreML → CPU)
- Applies `TemporalFeatureEngineer` in Python, then runs ONNX session
- Enforces business logic (min fare $2.50)

### Prefect Orchestration — two flows

| File | Flow | Purpose |
|------|------|---------|
| `src/pipelines/training_flow.py` | `training_pipeline` | Wraps `train.py` with retry logic (1 retry, 30s delay) |
| `src/serving/flow.py` | `etl_and_train_flow` | Runs Dask ETL then launches `torchrun` for DDP training |

### Distributed Training — `src/training/train_pytorch_ddp.py`

- `TabularNet`: MLP (128→64→1 with dropout)
- `ParquetStreamingDataset(IterableDataset)`: Streams rows from Parquet via `_manifest.json` — O(1) RAM
- DDP init: reads `torchrun` env vars, falls back NCCL→Gloo
- `save_processed_data()` writes a `_manifest.json` alongside partitioned Parquet so DDP workers can enumerate files in O(1) without cloud list operations

### EDA Dashboard — `src/ui/pages/1_EDA.py`

Expects pre-aggregated summary files (not raw parquet) at `data/summary/hourly_summary.parquet`. This file must be generated by a separate aggregation step before the EDA page will render.

### Model Performance — `src/ui/pages/2_Model_Performance.py`

Loads the sklearn pipeline from MLflow (`runs:/{run_id}/model`), extracts `pipeline.named_steps['model']` (the XGBRegressor) to read feature importances.

### Data flow
```
data/*.parquet
  → load_data() [Dask, lazy]
  → filter_outliers() + engineer_features() [Dask task graph]
  → .compute() [materializes to Pandas]
  → train_test_split
  → Pipeline.fit() [TemporalFeatureEngineer → XGBRegressor]
  → mlflow.sklearn.log_model → MLflow artifact store
  → API loads pipeline → pipeline.predict(BASE_FEATURE_COLS input)
```

## Key data schema notes

- Source: NYC TLC green taxi parquet, 21 columns, ~48k rows/month
- Pickup datetime: `lpep_pickup_datetime`
- `ehail_fee` is always null — not used
- `cbd_congestion_fee` present in 2025 data — not used as feature
- ~1836 rows/file have null `passenger_count`/`RatecodeID` — filled to 1 in `engineer_features()`
- Outlier filters: fare $2.50–$250, distance 0.1–60 mi, passengers 1–6
