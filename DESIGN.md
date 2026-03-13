### **Distributed Training with PyTorch DDP**
To run distributed training using PyTorch's DistributedDataParallel (DDP):

1. **Materialize processed, partitioned data:**
    ```bash
    python -m src.features.transform
    ```
2. **Launch DDP training (example: 4 GPUs on a single machine):**
    ```bash
    torchrun --standalone --nproc_per_node=4 src/training/train_pytorch_ddp.py
    ```
    - For multi-node clusters, set the appropriate environment variables and use the correct `torchrun` arguments for your environment.
3. **Checkpoints and model outputs** will be saved to the path specified by the `MODEL_OUTPUT_PATH` environment variable (default: `models/pytorch_model.pt`).
4. **Resume training** from a checkpoint by ensuring the checkpoint file exists at the expected location; the script will automatically load it if present.

**Note:**
- The DDP script streams partitioned Parquet data using the manifest file for scalable, O(1) memory usage.
- For large clusters or cloud environments, adapt the launch command and environment variables as needed for your orchestration system (e.g., Kubernetes, Slurm, AWS Batch).
# Technical Design Document: Petabyte-Scale NYC Taxi Fare Prediction

## 1. Architecture Overview

This document outlines the architectural decisions required to scale the NYC Taxi Fare Prediction pipeline from a local prototype to a production-grade system capable of processing **petabytes of data**.

While the current implementation runs locally for demonstration purposes, the underlying code structure is designed to be **infrastructure-agnostic**, allowing for seamless horizontal scaling on cloud clusters (AWS EMR, Google Dataproc, or Kubernetes).

### High-Level Data Flow

```mermaid
graph LR
    A[Raw Data Lake<br/>(S3 / GCS)] -->|Lazy Loading| B[Distributed ETL<br/>(Dask)]
    B -.->|Prefect Flow| B
    B -->|Sharded Parquet| C[Feature Store /<br/>Processed Data Lake]
    C -->|Streaming| D[Distributed Training<br/>(PyTorch DDP)]
    D -->|Model Artifacts| E[Model Registry<br/>(MLflow)]
    D -.->|Local .pt File| G[Local Checkpoint]
    E -->|REST API| F[Model Serving<br/>(Kubernetes / TorchServe)]
    
    subgraph "Observability"
        M[Metrics & Logs]
    end
    
    B -.-> M
    D -.-> M
    F -.-> M
```

## 2. Scaling Strategy

### 2.1 Data Ingestion & ETL
*   **Current State:** The `src/features/transform.py` module uses `dask.dataframe` to process data lazily on a single machine.
*   **Petabyte Scale:**
    *   **Distributed Compute:** The exact same Dask logic runs on a cluster of hundreds of nodes. Dask's scheduler manages task distribution, spilling to disk only when necessary.
    *   **Data Partitioning:** The ETL pipeline now writes **Hive-style partitioned Parquet** (e.g., `/pickup_month=01/`) using Dask's `partition_on` argument. This enables downstream consumers to efficiently read only relevant partitions (predicate pushdown), drastically reducing I/O.
    *   **Manifest File:** After writing Parquet files, the ETL generates a `_manifest.json` file listing all output files, enabling fast and scalable file discovery for distributed training and avoiding expensive directory listings.
    *   **Storage Abstraction:** By using libraries like `fsspec` and `pyarrow`, the code seamlessly supports both local paths (`data/`) and object storage (`s3://bucket/`).

### 2.2 Distributed Training
*   **Current State:** `src/training/train_pytorch_ddp.py` implements a custom `IterableDataset` that streams data from disk.
*   **Petabyte Scale:**
    *   **Data Parallelism (DDP):** We use PyTorch `DistributedDataParallel`. The dataset is sharded across $N$ GPUs. If we have 100 nodes with 8 GPUs each (800 total GPUs), the effective batch size becomes $B_{local} \times 800$.
    *   **Streaming I/O:** Traditional random-access DataLoaders crash RAM at this scale. Our `ParquetStreamingDataset` uses manifest-driven file discovery and `pyarrow` to stream partitioned Parquet data linearly from object storage, ensuring **O(1) memory usage** regardless of dataset size.
    *   **Network Optimization:** Gradients are synchronized using NCCL (NVIDIA Collective Communications Library) to minimize inter-node communication overhead.
    *   **Throughput Optimization:** The model is compiled using `torch.compile` (PyTorch 2.0+), which fuses CUDA kernels and optimizes the execution graph for the specific GPU architecture (e.g., A100/H100), significantly increasing training throughput.
    *   **Model Registry Integration:** Both the XGBoost (`train.py`) and PyTorch DDP (`train_pytorch_ddp.py`) pipelines log their final models directly to the MLflow Model Registry, providing a unified location for versioned production artifacts.

## 3. Reliability & Failure Modes

In a distributed training run involving 500+ nodes, hardware failure is not a possibility—it is a certainty.

### 3.1 Training Resilience
*   **Elastic Checkpointing:** The training loop saves a checkpoint (model weights + optimizer state) to a cloud-native path (e.g., S3) at the end of every epoch.
*   **Fault Tolerance:** If a node fails:
    1.  The orchestration layer (e.g., Kubernetes/Slurm) detects the failure.
    2.  The job is automatically restarted.
    3.  Training resumes from the last valid S3 checkpoint, losing at most $N$ steps of progress.
*   **Spot Instance Recovery:** This architecture allows us to use Spot Instances (AWS) or Preemptible VMs (GCP) to reduce training costs by up to 90%, as interruptions are handled gracefully.

### 3.2 Serving Reliability
*   **Training-Serving Skew:** Skew is eliminated by wrapping feature engineering logic (e.g., `TemporalFeatureEngineer`) into the model pipeline itself. The exact same code transforms data during batch training and real-time inference.
*   **Batch Inference:** The API now provides a `/predict_batch` endpoint for high-throughput scoring of multiple records in a single request, reducing HTTP overhead and supporting production-scale inference.
*   **Circuit Breakers:** (Planned) API clients should implement retries and circuit breakers to gracefully handle overloads; this is not yet present in the current codebase.

### 3.3 Edge & Hybrid Deployment
*   **ONNX Standardization:** To support deployment on robotic nodes (which may run C++ stacks) or hardware accelerators (NVIDIA Jetson), the training pipeline exports models to **ONNX**.
*   **Hybrid Inference Pattern:** The Serving API (`app.py`) implements a hybrid pattern:
    1.  Lightweight feature engineering (sin/cos time features) runs in Python.
    2.  Heavy compute (XGBoost inference) runs in `onnxruntime`, allowing for hardware acceleration and reduced dependency footprint.

### 3.4 Orchestration & Quality
*   **Workflow Management:** Prefect defines the end-to-end DAG, with tasks using retries and caching policies (`cache_key_fn`) to handle transient failures and avoid redundant computation.
*   **Data Quality Monitoring:** A separate aggregation pipeline generates summary statistics (e.g., `hourly_summary.parquet`). The EDA dashboard loads only these summaries, ensuring instant dashboard performance even at petabyte scale.

## 4. Bottleneck Analysis

| Bottleneck | Solution | Implementation |
| :--- | :--- | :--- |
| **I/O Bound** (Reading data is slower than GPU compute) | Prefetching & Caching | `DataLoader(..., num_workers=4, prefetch_factor=2)` hides I/O latency. |
| **Network Bound** (Gradient sync takes too long) | Gradient Accumulation | Implemented in the DDP training loop. Gradients are accumulated locally for $N$ steps (configurable via `GRAD_ACCUMULATION_STEPS`) before a single network synchronization, reducing traffic. |
| **Metadata Overhead** (Listing millions of files) | Manifest Files | Use a catalog (e.g., Hive Metastore) or manifest files instead of `glob` listing S3 buckets directly. |
| **Compute Bound** (GPU underutilization) | JIT Compilation | `torch.compile` optimizes kernel fusion to maximize FLOPs utilization on modern GPUs. |

## Opportunities for Improvement

While the current architecture is robust and scalable, several enhancements can further future-proof the pipeline for true petabyte-scale and production reliability:

- **Table Catalog/Schema Evolution:**  
    Integrate a table catalog (e.g., Apache Iceberg, Delta Lake, or Hive Metastore) to support schema evolution and further accelerate metadata operations beyond manifest files.

- **Advanced Data Loaders:**  
    For distributed training at scale, consider integrating Petastorm or WebDataset for efficient, sharded data streaming using the manifest file as input.

- **Observability:**  
    Add centralized logging and metrics (e.g., Prometheus, Grafana, ELK) for end-to-end monitoring and alerting.

- **Circuit Breakers and Retries:**  
    Ensure API clients implement robust retry and circuit breaker logic to gracefully handle service overloads.

- **Automated Data Quality Monitoring:**  
    Automate the generation and validation of summary statistics to catch data drift and anomalies early.