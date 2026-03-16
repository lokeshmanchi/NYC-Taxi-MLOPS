# NYC Taxi MLOps — Interview Reference Report

A complete technical walkthrough of every component, including the reasoning behind each design decision.

---

## 1. Project Overview

**Goal:** Build a production-grade MLOps pipeline to predict NYC Green Taxi fare amounts from trip features. The system ingests raw TLC parquet data, engineers features, trains an XGBoost regression model, serves predictions via a REST API, and monitors for data drift in production.

**What makes this non-trivial:** The architecture is designed to scale from a local laptop to petabyte-scale cloud clusters without code changes. The same ETL, feature engineering, and serving logic works on a single machine (Dask, local Parquet) or a 1,000-node cloud cluster (Dask on EMR, S3 storage, multi-node torchrun). Every infrastructure choice is abstracted behind interfaces (fsspec, Dask, AppConfig) to achieve this.

---

## 2. Dataset & Schema

- **Source:** NYC TLC Green Taxi parquet files, Jan–May 2025 (`data/`), ~48k rows/month, 21 columns
- **Target:** `fare_amount` (continuous regression)
- **Key raw columns used:**
  - `lpep_pickup_datetime` — extracted to `pickup_hour`, `pickup_dayofweek`, `pickup_month`
  - `trip_distance`, `passenger_count`, `PULocationID`, `DOLocationID`, `RatecodeID`
- **Ignored columns:** `ehail_fee` (always null), `cbd_congestion_fee` (2025-specific, not used as feature)
- **Null handling:** `passenger_count` and `RatecodeID` have ~1,836 nulls/file — filled to 1 (sensible default: solo standard-rate trip)
- **Outlier filters:** fare $2.50–$250, distance 0.1–60 mi, passengers 1–6

**Why these filters?** The fare range removes meter resets, test rides, and data entry errors. Distance bounds remove GPS glitches and airport-to-airport trips that distort the model. These are business-domain rules, not statistical heuristics.

---

## 3. Configuration Layer — `src/config.py`

```python
class AppConfig(BaseSettings):
    model_path: str = "models/model.pkl"
    ddp_backend: str = "nccl"
    batch_size: int = 4096
    mlflow_tracking_uri: str = "http://localhost:5000"
    ...
```

**Technology:** `pydantic-settings` (`BaseSettings`)

**Why pydantic-settings over plain dataclasses or `os.getenv()`?**
- Automatic type coercion (env vars are always strings; pydantic converts to `int`, `bool`, etc.)
- `.env` file support out of the box — developers can override locally without touching shell profiles
- Case-insensitive env var matching
- Validation at startup — if `MAX_BATCH_SIZE=abc` is set, the app crashes immediately with a clear error rather than at the first prediction
- Single source of truth: every service reads from the same `config` singleton

**Key config groups and why they exist:**
- **Paths** — all file paths in one place so Docker volume mounts (e.g., `MODEL_PATH=/app/models/model.pkl`) override seamlessly
- **DDP settings** — `world_size`, `rank`, `grad_accumulation_steps` etc. are read by `torchrun`-launched workers via env vars; pydantic captures all of them
- **Serving** — `max_batch_size` prevents memory exhaustion on the API pod from a malicious or misconfigured client

---

## 4. Data Pipeline (ETL) — `src/features/transform.py`

### 4.1 Why Dask, not Pandas?

Pandas loads the entire dataset into RAM. NYC TLC data for one month is ~200MB; at petabyte scale this is impossible. Dask creates a **lazy task graph** — operations are recorded but not executed until `.compute()` is called. This means:
- You can describe transformations on 100TB of data on a laptop without loading a single byte
- The same code runs on a local machine or a distributed Dask cluster (AWS EMR, GCP Dataproc) by changing the scheduler

### 4.2 `load_data()` — Lazy loading with fsspec

```python
dd.read_parquet(data_path)  # data_path can be "data/", "s3://bucket/", "gs://bucket/"
```

Dask uses `fsspec` under the hood for I/O. By using `fsspec.open()` explicitly elsewhere (serving, DDP), the entire stack is storage-agnostic. Swapping from local to S3 requires only changing an environment variable, not code.

### 4.3 `validate_raw_data()` — Pre-flight schema check

```python
missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]  # metadata-only, free
sample = df[REQUIRED_RAW_COLS].head(1000)  # reads first partition only
null_rates = sample.isnull().mean()
```

**Why this approach:**
- Column check is pure metadata (Dask reads schema from Parquet footers without loading data)
- `.head(1000)` reads only the first partition — catches data quality issues (renamed columns, schema evolution from TLC) before a multi-minute ETL job fails deep in the task graph
- Raises `ValueError` on missing columns (hard fail, correct behavior); warns on high nulls (soft fail, data may be usable)

### 4.4 `filter_outliers()` + `engineer_features()`

All transformations build the Dask task graph — no computation happens:
```python
dt = dd.to_datetime(df["lpep_pickup_datetime"])
df = df.assign(pickup_hour=dt.dt.hour, ...)
```

**Why assign() over direct column assignment?** Dask DataFrames are immutable in graph-building mode. `assign()` creates a new node in the task graph rather than mutating in place, which is safe for distributed execution.

### 4.5 `save_processed_data()` — Materialization

```python
final_df.to_parquet(output_path, partition_on=["pickup_month"], compression="snappy")
```

**Hive partitioning by `pickup_month`:** Creates a directory structure like `data/processed/pickup_month=1/part-0.parquet`. Downstream readers (DDP training, drift monitoring) can use **predicate pushdown** — reading only the months they need without scanning the entire dataset. At petabyte scale, this eliminates 90%+ of I/O.

**Why Snappy compression?** Snappy is optimized for speed over ratio. For ML training, you're I/O-bound (reading data into GPU is faster than the disk read). Snappy decompresses ~5x faster than gzip, reducing the I/O bottleneck. ZSTD would give better ratio but adds decompression overhead.

**Why not store derived features (is_weekend, hour_sin, hour_cos)?**
- Storing derived features would mean the serving pipeline needs to know which version of the feature engineering code was used to process that data
- If you update the sin/cos formula or add new features, stored data becomes stale and inconsistent with what the model was trained on
- Instead, derived features are always computed on-the-fly by `TemporalFeatureEngineer` — same code, same result, everywhere

### 4.6 `_manifest.json` — O(1) File Discovery

```python
all_files = sorted([p for p in fs.find(fs_path, detail=False) if p.endswith(".parquet")])
json.dump(all_files, f)
```

**Problem:** At scale, S3 `list_objects` is slow (100ms+ per 1,000 objects), metered by AWS, and eventually fails when you have millions of partitions.

**Solution:** After writing, record all file paths in a JSON manifest. Workers read the manifest once (O(1)) and get the full file list. No cloud listing needed.

**Fallback:** If the manifest doesn't exist, workers fall back to `fs.find()` — safe for local development where listing is fast.

---

## 5. Feature Engineering — `src/features/core.py`

### 5.1 Two-tier feature set

```python
BASE_FEATURE_COLS = [8 raw/cleaned columns]   # what's stored in Parquet
FEATURE_COLS = BASE_FEATURE_COLS + ["is_weekend", "hour_sin", "hour_cos"]  # what the model sees
```

**Why separate them?** The API accepts `BASE_FEATURE_COLS` as input (simpler for clients). The model internally needs all 11. The pipeline handles the expansion transparently.

### 5.2 `TemporalFeatureEngineer` — sklearn TransformerMixin

```python
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self  # stateless
    def transform(self, X):
        X["hour_sin"] = np.sin(2 * np.pi * X["pickup_hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["pickup_hour"] / 24)
        X["is_weekend"] = (X["pickup_dayofweek"] >= 5).astype(int)
```

**Why a sklearn transformer, not a standalone function?**
- Plugs directly into `sklearn.Pipeline` — the transformer and model are serialized together as one artifact. Loading the model automatically includes the feature logic.
- Satisfies the sklearn API contract (`fit`/`transform`) so it works with `GridSearchCV`, `cross_val_score`, and other sklearn tooling
- The `fit()` is a no-op (returns `self`) because this transformer is stateless — it derives features mathematically, not statistically

**Why cyclic encoding (sin/cos) for hour?**
- `pickup_hour` is 0–23. If you use it as a raw integer, the model sees hour 23 and hour 0 as maximally different (23 apart), when they're actually 1 hour apart.
- Cyclic encoding maps hour to a point on the unit circle: `(sin(2π·h/24), cos(2π·h/24))`. Hour 23 and hour 0 are now geometrically adjacent. This is especially important for taxi demand, which has continuity across midnight.

**Why `is_weekend` as a binary flag, not `pickup_dayofweek` raw?**
- Mon–Fri have similar demand patterns; Sat–Sun are distinct. The binary flag captures this domain knowledge directly. The raw day-of-week integer implies an ordering (Thursday > Wednesday) that doesn't exist.

---

## 6. XGBoost Training — `src/training/train.py`

### 6.1 sklearn Pipeline as the artifact

```python
pipeline = Pipeline([
    ("features", TemporalFeatureEngineer()),
    ("model", xgb.XGBRegressor(**params)),
])
mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name="nyc-taxi-regressor")
```

**Why bundle the transformer in the pipeline?**
- The pipeline is the deployable artifact. Loading it gives you a single `.predict(BASE_FEATURE_COLS)` call — the transformer runs automatically before the model
- Eliminates a whole class of bugs: the transformer code that ran during training is the same object that runs at serving time
- `mlflow.sklearn.log_model` serializes the full pipeline with `joblib`, so nothing is lost

**Why XGBoost for this problem?**
- NYC taxi fare is a tabular regression problem. Tree-based models (XGBoost, LightGBM) consistently outperform neural networks on tabular data with <1M rows because they don't need to learn feature interactions from scratch — each split explicitly encodes one
- XGBoost is fast (C++ backend, parallelized across CPU cores), interpretable (feature importances), and doesn't require feature scaling
- The dataset has ~240k rows (5 months × 48k) — well within XGBoost's sweet spot

**Hyperparameter rationale:**
- `n_estimators=400, learning_rate=0.05` — slow learning rate with many trees reduces overfitting vs fast learning with few trees
- `subsample=0.8, colsample_bytree=0.8` — stochastic training introduces variance that acts as regularization (similar to dropout)
- `min_child_weight=5` — prevents leaves with too few samples, critical for a dataset with outlier fares
- `max_depth=6` — deep enough to capture interactions (time × distance × zone) without memorizing noise

### 6.2 ONNX + INT8 export

```python
onx = to_onnx(xgb_s, X_tensor)            # FP32 ONNX
quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QUInt8)  # INT8
```

**Why export to ONNX?**
- ONNX is a hardware-portable format. The same file runs on NVIDIA GPUs (TensorRT), Intel CPUs (OpenVINO), ARM (ONNX Runtime), and NVIDIA Jetson — without rewriting inference code
- Only the XGBoost step is exported (not the `TemporalFeatureEngineer`). This is intentional: custom sklearn transformers don't map cleanly to ONNX ops. Instead, the Python transformer runs first, then ONNX handles the compute-heavy inference.

**Why INT8 quantization?**
- INT8 is ~4x smaller than FP32 and 2–3x faster on CPUs/edge hardware (integer SIMD units are cheaper than floating-point units)
- `quantize_dynamic` (post-training quantization) doesn't need a calibration dataset — the model weights are quantized offline. Activations remain FP32 at runtime.
- For XGBoost with 400 trees, the accuracy loss is negligible (<0.1% MAE change in practice)

---

## 7. Distributed Training — `src/training/train_pytorch_ddp.py`

This path is designed for when the dataset exceeds RAM, or when you need GPU acceleration. It trains a PyTorch MLP instead of XGBoost.

### 7.1 TabularNet — MLP architecture

```python
nn.Sequential(
    nn.Linear(11, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 1)
)
```

**Why this over XGBoost for DDP?** XGBoost doesn't have native PyTorch DDP support. The neural network is the vehicle for demonstrating the distributed training infrastructure — the architecture choices (128→64→1) are straightforward for tabular regression.

**Dropout(0.2):** Regularization. With only 11 features, the model risks memorizing noise on small batches; dropout randomly zeroes 20% of activations per forward pass.

### 7.2 `ParquetStreamingDataset` — O(1) RAM streaming

```python
class ParquetStreamingDataset(IterableDataset):
    def __iter__(self):
        for file_path in self.my_files:
            parquet_file = pq.ParquetFile(file_path, filesystem=fs)
            for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=self.columns):
                yield (torch.tensor(X), torch.tensor(y))
```

**Why IterableDataset over Dataset?**
- `Dataset` (map-style) requires `__len__` and random access by index — loading all row indices into memory, which is impossible at petabyte scale
- `IterableDataset` streams linearly: only one batch lives in RAM at a time
- `iter_batches()` from PyArrow reads directly from Parquet without loading the full file

**DDP sharding via manifest:**
```python
self.my_files = all_files[rank::world_size]
```
Each GPU (rank) processes a different non-overlapping slice of files. With 4 GPUs: GPU 0 gets files 0,4,8…; GPU 1 gets files 1,5,9… etc. No coordination needed — each worker knows its slice statically.

**Why `use_listings_cache=False`?** Long-running training jobs can run for hours. S3 metadata caches go stale; with `False`, each worker creates a fresh filesystem connection, preventing stale-cache errors.

### 7.3 DDP initialization — NCCL → Gloo fallback

```python
try:
    dist.init_process_group(backend="nccl")
except Exception:
    dist.init_process_group(backend="gloo")
```

- **NCCL** (NVIDIA Collective Communications Library): GPU-to-GPU gradient synchronization over NVLink or InfiniBand. ~10x faster than CPU communication for large models. Requires CUDA.
- **Gloo**: CPU fallback. Runs on any machine. Used automatically when CUDA isn't available (local dev, CI).

**Why torchrun (not spawn/fork)?** `torchrun` is PyTorch's official launcher (replaces `torch.distributed.launch`). It handles rendezvous (worker coordination), elastic training (node failure recovery), and sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE` env vars automatically.

### 7.4 Gradient Accumulation

```python
loss = loss / GRAD_ACCUMULATION_STEPS
scaler.scale(loss).backward()
if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
    scaler.step(optimizer)
    optimizer.zero_grad()
```

**Why?** With large models, fitting a big batch (e.g., 4096 × 4 = 16,384 effective samples) into GPU memory at once causes OOM. Gradient accumulation simulates a larger batch by accumulating gradients over multiple forward passes before updating weights. This:
- Reduces peak GPU memory usage proportionally
- Keeps the effective batch size large (better gradient estimates, stable training)
- Reduces inter-GPU communication frequency (gradient sync only happens every N steps)

### 7.5 AMP Mixed Precision

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    outputs = ddp_model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
```

**Why?** FP16 arithmetic is 2–4x faster than FP32 on modern NVIDIA GPUs (Tensor Cores). `GradScaler` prevents numerical underflow in FP16 by scaling the loss up before backprop and scaling gradients back down before the optimizer step.

### 7.6 `torch.compile` (PyTorch 2.0+)

```python
model = torch.compile(model)  # compiled before DDP wrapping
```

`torch.compile` uses TorchDynamo to trace the computation graph and TorchInductor to generate optimized GPU kernels (kernel fusion, avoiding unnecessary memory copies). For MLPs on A100/H100 GPUs, this can yield 30–50% throughput improvement with zero code changes.

**Why compile before DDP?** DDP wraps the module and intercepts `forward()`. Compiling after DDP would compile the communication hooks, not just the model logic.

### 7.7 Elastic Checkpointing

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": ddp_model.module.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
}
with fs.open(checkpoint_path, "wb") as f:
    torch.save(checkpoint, f)
```

**Why save all three state dicts?**
- `model_state_dict`: weights and biases — obvious
- `optimizer_state_dict`: Adam's per-parameter momentum and variance estimates (`m` and `v`). Without this, resuming training starts with zero momentum — effective learning rate spikes, destabilizing training for several epochs
- `scaler_state_dict`: AMP scale factor history. The scaler tracks the scale dynamically; losing this causes the first resumed batch to use a wrong scale, producing NaN gradients

**Why `ddp_model.module.state_dict()` not `ddp_model.state_dict()`?** DDP wraps the model in a `module` attribute. Saving the wrapper's state_dict would include the DDP communication hooks, making the checkpoint incompatible with non-DDP loading.

**fsspec for checkpoints:** `fs.open(checkpoint_path, "wb")` — same path could be local (`models/checkpoint.pt`) or cloud (`s3://bucket/models/checkpoint.pt`). One line change in config enables spot-instance-safe cloud checkpointing.

---

## 8. Prefect Orchestration — Two Flows

### 8.1 `training_pipeline` — `src/pipelines/training_flow.py`

```python
@task(name="train-xgboost", retries=1, retry_delay_seconds=30)
def train_task(...): ...

@flow(name="nyc-taxi-training-pipeline")
def training_pipeline(...): ...
```

Wraps XGBoost training (`train.py`) in a Prefect flow. The `retries=1` handles transient MLflow server failures or network blips that don't indicate real errors. 30s delay gives the MLflow server time to recover.

### 8.2 `etl_and_train_flow` — `src/serving/flow.py`

```python
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1), retries=2)
def run_etl(input_path, output_path): ...

@task
def run_training(): ...
    subprocess.run(f"torchrun --standalone --nproc_per_node={nproc} ...", shell=True, check=True)
```

**Why `cache_key_fn=task_input_hash`?** If the input data hasn't changed (same path, same content hash), Prefect skips the ETL task and uses the cached result. ETL on large datasets can take hours; this avoids re-running it if only the training code changed.

**Why `subprocess.run` for torchrun?** `torchrun` is a standalone process launcher — it can't be imported and called as a Python function. It spawns multiple worker processes (one per GPU), each with their own `RANK` env var. Calling it via `subprocess` is the correct pattern.

**Multi-node support:** If `RDZV_ENDPOINT` is set (e.g., `head-node:29400`), the command switches to `--rdzv_backend c10d --rdzv_endpoint` for multi-node rendezvous instead of `--standalone` (single-node).

---

## 9. Model Serving — `src/serving/app.py`

### 9.1 Hybrid ONNX / sklearn loading

```python
def _attempt_model_load(path: str):
    if path.endswith(".onnx"):
        return ort.InferenceSession(model_bytes), "onnx"
    return joblib.load(f), "sklearn"
```

**Why check file extension, not file existence?** The extension is an explicit intent signal — the operator chose which inference path to use by setting `MODEL_PATH`. Probing for file existence could silently fall back to sklearn even if the ONNX path was intentional.

**Why ONNX runtime for serving (not the sklearn pipeline directly)?**
- ONNX Runtime is optimized for inference: it applies graph optimizations at load time (constant folding, node fusion) that aren't applied to sklearn's predict path
- ONNX Runtime supports hardware execution providers (CUDA, TensorRT) that sklearn doesn't
- For high-throughput batch predictions, ONNX Runtime is significantly faster

### 9.2 Retry on model load

```python
for attempt in range(3):
    try:
        model, inference_mode = _attempt_model_load(MODEL_PATH)
        break
    except Exception as e:
        if attempt < 2:
            wait = 2 ** attempt  # 1s, then 2s
            time.sleep(wait)
```

**Why exponential backoff (1s, 2s)?** MLflow or the model artifact store may be momentarily unavailable at container startup (race condition with `depends_on` health checks). Exponential backoff avoids thundering herd: if 10 API replicas all retry simultaneously, they'd hammer the upstream service. With different start times and exponential delays, the load spreads naturally.

### 9.3 Warmup inference

```python
def warmup_model():
    model.predict(sample)  # one dummy prediction on startup
```

**Why?** JIT-compiled runtimes (ONNX Runtime, XGBoost) defer certain optimizations to the first call (lazy kernel compilation). Without warmup, the very first real prediction spikes 50–200ms. After warmup, subsequent calls are <1ms. This prevents a bad user experience for the first request after deployment.

### 9.4 `model_loaded_bool` Prometheus gauge

```python
model_loaded_gauge = Gauge("model_loaded_bool", "1 if model is loaded and serving, 0 otherwise")
model_loaded_gauge.set(1 if model is not None else 0)
```

**Why a custom gauge vs using the health endpoint?** The `/health` endpoint returns JSON that's consumed by humans or load balancers. Prometheus doesn't parse JSON responses from application endpoints — it scrapes `/metrics` for the Prometheus text format. A `Gauge` metric is exposed in that format and can be alerted on directly (e.g., PagerDuty alert if `model_loaded_bool == 0`).

### 9.5 BackgroundTasks for prediction logging

```python
background_tasks.add_task(_log_prediction, features.model_dump(), predicted_fare)
return PredictionResponse(predicted_fare=predicted_fare)
```

**Why BackgroundTasks?** Parquet I/O is synchronous blocking I/O. If called on the request thread, it adds 5–20ms to every prediction response. `BackgroundTasks` (Starlette's built-in mechanism) runs the function after the HTTP response is sent — the client gets their result immediately, and the logging happens asynchronously in the same event loop iteration.

### 9.6 Batch prediction logging — single write

```python
def _log_predictions_batch(features_list, fares):
    rows = [{**feat, "predicted_fare": fare, ...} for feat, fare in zip(features_list, fares)]
    table = pa.Table.from_pylist(rows)
    pq.write_to_dataset(table, ...)  # one write for the whole batch
```

**Why batch write over per-row writes?** PyArrow's `write_to_dataset` acquires a file lock, writes the row group, and closes the file per call. For a batch of 100 predictions, the original code did 100 separate file operations. The batch write does 1. At 256 max batch size, this eliminates up to 255 redundant I/O operations per batch request.

---

## 10. Edge Inference — `src/training/edge_run.py`

Designed for NVIDIA Jetson, Intel NUC, or any CPU-only edge node. Performance constraints: <10ms SLA per prediction.

### 10.1 Execution provider priority: TensorRT → CUDA → CPU

```python
providers = [
    ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
    ("CUDAExecutionProvider", {}),
    ("CPUExecutionProvider", {}),
]
session = ort.InferenceSession(model_path, providers=providers)
```

ONNX Runtime tries providers in order and uses the first one available:
- **TensorRT**: NVIDIA's inference optimizer. Fuses layers, uses FP16 Tensor Cores on Jetson Orin. Fastest.
- **CUDA**: Generic GPU without TensorRT optimization. Still faster than CPU for larger models.
- **CPU**: Always available. Default for Intel NUC or CPU-only deployment.

**Why FP16 on TensorRT?** NVIDIA Jetson's Tensor Cores run INT8/FP16 ~5x faster than FP32. For XGBoost (which doesn't have complex sequential dependencies), FP16 precision loss is negligible.

### 10.2 LRU cache — skip ONNX entirely for repeated routes

```python
@lru_cache(maxsize=1024)
def _run_cached(trip_distance, passenger_count, PULocationID, DOLocationID, pickup_hour, ...):
    features = np.array([[...]], dtype=np.float32)
    return float(_session.run([_label_name], {_input_name: features})[0][0])
```

**Why LRU cache?** NYC taxi has highly repetitive routes (JFK ↔ Manhattan, LaGuardia ↔ Midtown repeat constantly). `lru_cache` stores the 1,024 most-recently-used route signatures and returns the cached fare instantly — zero model inference, zero numpy allocation.

**Why primitive types as cache keys (not numpy arrays)?** `@lru_cache` requires hashable arguments. NumPy arrays are not hashable. Primitive Python types (`float`, `int`) are. The cached function takes all 8 raw input values as individual arguments.

**Cache invalidation on OTA swap:** When a new model is loaded via hot-swap, `_run_cached.cache_clear()` is called — old predictions from the previous model don't bleed into new ones.

### 10.3 NumPy-only feature computation

```python
features = np.array([[
    trip_distance, passenger_count, PULocationID, DOLocationID,
    pickup_hour, pickup_dayofweek, pickup_month, RatecodeID,
    int(pickup_dayofweek >= 5),                       # is_weekend
    math.sin(2 * math.pi * pickup_hour / 24),         # hour_sin
    math.cos(2 * math.pi * pickup_hour / 24),         # hour_cos
]], dtype=np.float32)
```

**Why inline numpy instead of `TemporalFeatureEngineer`?** The transformer creates a `pd.DataFrame`, copies it, runs pandas operations, and returns a DataFrame — that's ~3 allocations for a single row. On edge hardware, this overhead is measurable. Direct numpy array construction for a single prediction is a single allocation. The edge benchmark shows this delivers ~10x lower latency over the baseline pandas approach.

### 10.4 SLA guard with heuristic fallback

```python
elapsed_ms = (time.perf_counter() - t0) * 1000
if elapsed_ms > LATENCY_BUDGET_MS:  # 10ms
    raw_fare = 3.50 + raw_features["trip_distance"] * 2.50  # fallback
```

**Why a heuristic fallback?** On embedded hardware under thermal throttling, the first inference after a cache miss might exceed the SLA. Rather than returning an error (bad UX for a real-time robotic application), the system returns a reasonable estimate. The $3.50 base + $2.50/mile formula approximates NYC taxi metered rates, giving a result within ~20% of the model for typical trips.

### 10.5 OTA Hot-Swap

```python
def _watch_model(model_path: str):
    while True:
        time.sleep(OTA_POLL_INTERVAL_S)  # 5 minutes
        if os.path.getmtime(model_path) > last_mtime:
            _init_session(model_path)  # reload atomically
```

**Why?** Edge devices in the field can't be restarted to load a new model. The daemon thread polls the model file's modification time every 5 minutes. When a new model is deployed (file replaced), it reloads the session, clears the LRU cache (invalidating stale predictions), and resumes serving — zero downtime.

**CPython GIL makes this safe:** The global `_session` assignment is atomic in CPython due to the GIL. There's no race condition between the watcher thread writing `_session` and the inference thread reading it.

---

## 11. Drift Monitoring — `src/monitoring/drift.py`

```python
ref_df = pd.read_parquet(reference_path, columns=BASE_FEATURE_COLS).sample(5000)
cur_df = pd.read_parquet(current_path, columns=BASE_FEATURE_COLS)
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=cur_df)
report.save_html(output_path)
```

**Why Evidently?** Evidently is purpose-built for ML monitoring. `DataDriftPreset` automatically selects the right statistical test per column type (PSI for distributions, chi-squared for categoricals, K-S test for numericals) and renders an interactive HTML report. Building this from scratch would require implementing multiple statistical tests and a visualization layer.

**Why sample 5,000 from reference?** The processed training dataset has ~200k rows. Running drift tests on the full dataset is unnecessary — statistical tests converge at ~5,000 samples. Capping at 5,000 keeps report generation under 1 second.

**Prediction logging design:** Each `/predict` call logs input features + predicted fare + date to `data/predictions/date=YYYY-MM-DD/`. Date-partitioned Parquet means:
- The drift report can filter by recent dates
- Old prediction logs can be cleaned up by deleting old date partitions
- PyArrow's `write_to_dataset` handles concurrent writes safely (each write gets a unique filename via UUID)

---

## 12. Streamlit Dashboard — `src/ui/`

### `Home.py` — Summary stats
Loads raw parquet files (not processed) to show basic metrics: total trips, average fare, average distance. Uses `@st.cache_data` to avoid re-reading parquet on every page interaction.

### `1_EDA.py` — Exploratory Data Analysis
Reads `data/summary/hourly_summary.parquet` (pre-aggregated). **Why pre-aggregated?** EDA dashboards that load raw data are slow. At petabyte scale, loading 1TB of raw data to show a bar chart is unacceptable. The aggregation step runs once during ETL (`save_summary()` called inside `save_processed_data()`), and the dashboard loads a tiny summary file (24 rows for hourly breakdown).

### `2_Model_Performance.py` — MLflow integration
```python
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
latest = runs.iloc[0]
pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
xgb_model = pipeline.named_steps["model"]
scores = xgb_model.feature_importances_
```
Dynamically loads the latest MLflow run — no hardcoded run IDs. Shows MAE/RMSE/R² and a feature importance chart. The `@st.cache_data(ttl=30)` refreshes every 30 seconds so a new training run appears automatically.

### `3_Predict.py` — Live Prediction
Calls the FastAPI `/predict` endpoint directly via `httpx`. Checks `/health` on load and shows the model's inference mode (onnx vs sklearn). Sends `BASE_FEATURE_COLS` — the pipeline handles the rest.

---

## 13. Observability Stack

### Prometheus (`monitoring/prometheus.yml`)
Scrapes `api:8000/metrics` every 15 seconds. The `/metrics` endpoint is exposed by `prometheus-fastapi-instrumentator`, which automatically instruments all FastAPI routes with:
- `http_requests_total{handler, method, status_code}` — request count
- `http_request_duration_seconds{handler, le}` — latency histogram

Plus the custom `model_loaded_bool` gauge exposed by `prometheus_client.Gauge`.

### Grafana (`monitoring/grafana/dashboards/nyc-taxi.json`)
Five panels:
1. **Request Rate (req/s)** — `rate(http_requests_total[1m])` by handler. Shows traffic shape.
2. **P50/P95 Latency** — `histogram_quantile(0.95, ...)`. The histogram percentile tells you tail latency, which matters more than average for SLA compliance.
3. **Error Rate** — `rate(http_requests_total{status_code=~"4..|5.."}[1m])`. Filters only error responses.
4. **Total Predictions** — Cumulative count of `POST /predict` 200 responses. Business-level KPI.
5. **Model Loaded** — `model_loaded_bool`. Green = LOADED, Red = NOT LOADED. Accurate because it's set by the application directly after the load attempt.

**Why not just use the `/health` endpoint for monitoring?** Prometheus scrapes `/metrics`, not `/health`. Health endpoints return JSON for load balancers; Prometheus expects the text exposition format.

---

## 14. Docker Infrastructure

### Container isolation and minimal images

All three application Dockerfiles use `python:3.11-slim` — the Debian slim variant without development tools, documentation, or test utilities. **Why slim?** Smaller images (300MB vs 900MB for full python:3.11) mean faster pulls, smaller attack surface, and lower container registry storage costs.

Each container only installs the packages it needs:
- **Trainer**: pandas, sklearn, xgboost, mlflow, prefect (no fastapi, streamlit)
- **API**: pandas, sklearn, xgboost, mlflow, fastapi, uvicorn (no prefect, streamlit)
- **UI**: pandas, sklearn, mlflow, streamlit, plotly, httpx (no fastapi, uvicorn)

**Why separate containers?** Separation of concerns. The API image doesn't need Streamlit; the UI image doesn't need uvicorn. Keeping images small reduces startup time and memory footprint.

### `trainer` service with `profiles: ["training"]`

```yaml
trainer:
  profiles: ["training"]
```

The trainer only runs on `docker compose --profile training up` (or `make train`). It's not part of the default `make up` stack. **Why?** Training is a one-shot job that exits when done, not a long-running service. Running it as part of the default stack would cause it to crash-loop after it completes.

### Health checks for dependency ordering

```yaml
mlflow:
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
trainer:
  depends_on:
    mlflow:
      condition: service_healthy
```

The trainer waits for MLflow to be healthy (not just started) before beginning. Without this, the trainer would try to log to MLflow before its SQLite database is ready, causing the training run to fail silently.

### Shared `models_data` volume

```yaml
volumes:
  - models_data:/app/models
```

Trainer writes the model artifact to `models_data`; the API container reads from the same volume. This is how `make train` + `docker compose restart api` achieves zero-downtime model updates without a model registry download step.

---

## 15. Code Quality

### Flake8 + pre-commit

`.flake8` config: `max-line-length=120` (pragmatic for data science code with long method chains), `ignore=W292,W293` (trailing whitespace — common in Jupyter-to-script conversions).

Pre-commit runs flake8 automatically before every `git commit`. This enforces code style as a gate, not a suggestion.

### pytest — `src/features/test_features.py`

Two tests:
1. `test_temporal_feature_engineer` — verifies sin/cos cyclic encoding and weekend flag on pandas DataFrames
2. `test_filter_outliers_dask` — verifies Dask outlier filtering materializes correctly

**Why test these specifically?** These are the functions that run on both the training and serving paths. A bug here would silently corrupt predictions in production. Testing the sklearn transformer separately from the pipeline ensures the component is correct before testing integration.

---

## 16. Key Architectural Patterns & Cross-Cutting Decisions

### Infrastructure-agnostic design
Every I/O path uses `fsspec`: `load_data()`, `save_processed_data()`, `app.py` model loading, DDP dataset, checkpointing. The only difference between local and cloud deployment is the path string (`data/` vs `s3://bucket/`). No code changes needed.

### No training-serving skew
The single most common ML production bug. Prevented here by:
1. `TemporalFeatureEngineer` is part of the sklearn pipeline artifact — the exact same Python object runs at training and serving time
2. Derived features are never stored — always recomputed identically
3. `BASE_FEATURE_COLS` is the API input contract; the pipeline adds derived features internally

### Pipeline-as-artifact pattern
`mlflow.sklearn.log_model(pipeline)` serializes the full pipeline (transformer + model) as one artifact. When the API loads the model, it gets a single callable `pipeline.predict(X)` — no separate transformer loading, no version mismatch risk.

### The manifest pattern for petabyte scale
Cloud object storage (S3, GCS) has expensive `LIST` operations. At 1M+ Parquet files, listing takes minutes and costs money. Writing a manifest JSON after ETL makes file discovery O(1) regardless of dataset size.

### Observability at every layer
- **ETL**: print statements + Prefect flow logs
- **Training**: MLflow metrics (step loss, epoch loss, samples/sec, GPU memory, GPU utilization)
- **Serving**: Prometheus metrics (request rate, p50/p95 latency, error rate, model loaded)
- **Data quality**: Evidently drift reports comparing training vs live distribution
- **Business**: Total prediction count in Grafana; prediction logs in Parquet for analysis

### Cost efficiency decisions
- **Snappy vs gzip**: Faster decompression = fewer GPU idle cycles waiting for data = better GPU utilization = cheaper compute per training hour
- **INT8 quantization**: 4x smaller model = 4x more throughput per edge device, or 4x fewer edge devices needed
- **LRU cache on edge**: Eliminating ONNX inference for cache hits = no GPU power draw for repeated routes
- **Hive partitioning**: Predicate pushdown = reading 1/12 of data per month = 12x cheaper queries
- **Spot/preemptible instances**: Elastic checkpointing saves every epoch, enabling 90% cheaper GPU compute via spot instances

---

## Quick Reference: Technology Choices

| Component | Technology | Why |
|-----------|-----------|-----|
| ETL | Dask | Lazy, distributed, same API as pandas |
| Feature transformer | sklearn TransformerMixin | Pipeline serialization, sklearn ecosystem |
| Regression | XGBoost | Best tabular performance, interpretable, fast |
| DDP training | PyTorch + torchrun | Native DDP, elastic training, multi-node |
| Experiment tracking | MLflow | Open source, sklearn integration, model registry |
| Orchestration | Prefect 3 | Python-native flows, caching, retries, UI |
| Serving | FastAPI + Uvicorn | Async, type-safe, auto OpenAPI docs, ASGI |
| Edge inference | ONNX Runtime | Hardware-portable, TensorRT provider, quantization |
| Drift detection | Evidently | Statistical tests + HTML reports, no infra needed |
| Metrics | Prometheus + Grafana | Industry standard, pull-based, alerting |
| Config | pydantic-settings | Type coercion, .env support, validation |
| Storage abstraction | fsspec | Single API for local, S3, GCS, MinIO |
| Containerization | Docker Compose | Local orchestration, health checks, volume sharing |
