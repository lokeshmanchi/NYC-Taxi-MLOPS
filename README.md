# NYC-Taxi-MLOPS

## Project Architecture Diagram

```mermaid
flowchart TD
    A["Raw Data (Parquet)"] -->|Dask Lazy Load| B["Base Feature Engineering (Transform.py)"]
    B -->|Filtered & Engineered| C["Processed Data Parquet"]
    C -->|Stream w/ Manifest| E2["Distributed Training (PyTorch DDP)"]
    E2 -->|Log Model| F["MLflow Model Registry"]
    A -->|Load & Transform| D["Train/Test Split"]
    D -->|Pandas DataFrame| E["Model Training (XGBoost)"]
    E -->|Log Pipeline| F
    F -->|XGBoost/ONNX Artifact| G["Model Serving API (FastAPI, Hybrid)"]
    G -->|Prediction| H["User/API Client"]
    C -.->|Separate Aggregation Job| I["EDA Dashboard (Streamlit)"]
    F -.->|Load Pipeline| J["Model Performance Dashboard"]
    subgraph Orchestration
      K["Prefect Flow"]
    end
    K -->|Run| B
    K -->|Launch| E2
    subgraph Experiment_Tracking
      L["MLflow"]
    end
    E -.-> L
    E2 -.-> L
    J -.->|Fetch Run Data| L
```

### Diagram Explanation

This diagram illustrates the end-to-end MLOps pipeline for NYC Taxi fare prediction:

- **Raw Data (Parquet):** Source data files, loaded lazily using Dask for scalability.
- **Base Feature Engineering (Transform.py):** Cleans and engineers base features using Dask (see `src/features/transform.py`).
- **Processed Data Parquet:** Output of feature engineering, partitioned and ready for training.
- **Train/Test Split → Model Training (XGBoost):** Local training path. Data is materialized to Pandas, split, and used to train an XGBoost pipeline (see `src/training/train.py`).
- **Stream w/ Manifest → Distributed Training (PyTorch DDP):** Distributed training path. Data is streamed directly from Parquet using a manifest file for efficient sharding and processed by PyTorch DDP with `torch.compile` optimization (see `src/training/train_pytorch_ddp.py`).
- **MLflow Model Registry:** Both training paths log their models to MLflow for experiment tracking and model registry.
- **Model Serving API (FastAPI, Hybrid):** Serves predictions using either the XGBoost pipeline (Cloud) or ONNX artifact (Edge) (see `src/serving/app.py`).
- **User/API Client:** Consumes predictions from the API.
- **EDA Dashboard (Streamlit):** Loads pre-aggregated summaries for exploratory data analysis (see `src/ui/pages/1_EDA.py`).
- **Model Performance Dashboard:** Loads the pipeline from MLflow to display metrics and feature importances (see `src/ui/pages/2_Model_Performance.py`).
- **Prefect Flow:** Orchestrates ETL and distributed training (see `src/serving/flow.py`).
- **MLflow:** Used for experiment tracking, metrics, and model registry.

Each component in the diagram maps directly to a module or script in the codebase, ensuring reproducibility, scalability, and clear separation of concerns.

### Edge & On-Premise Deployment

While the documentation highlights cloud scalability, the architecture is designed to be **infrastructure-agnostic**, making it equally suitable for on-premise and edge environments:

- **Edge Robotic Systems:** This pipeline can be containerized and deployed to edge robotic systems using the same Docker/Kubernetes logic used in the cloud.
- **Storage Independence:** The codebase utilizes `fsspec` (in `transform.py` and `app.py`), allowing seamless switching between cloud storage (S3/GCS) and local on-premise storage (MinIO, NFS) without code changes.
- **Low-Latency Inference:** For resource-constrained edge devices, the architecture supports exporting models to ONNX (Open Neural Network Exchange) to run on specialized hardware accelerators.
