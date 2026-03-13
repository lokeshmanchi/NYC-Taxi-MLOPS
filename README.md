# NYC-Taxi-MLOPS

## Project Architecture Diagram

```mermaid
flowchart TD
    A[Raw Data (Parquet)] -->|Dask Lazy Load| B[Base Feature Engineering (Transform.py)]
    B -->|Filtered & Engineered| C[Processed Data Parquet]
    C -.->|Materialize| D[Train/Test Split]
    D -->|Pandas DataFrame| E[Model Training (XGBoost Pipeline)]
    C -.->|Stream w/ Manifest| E2[Distributed Training (PyTorch DDP)]
    E -->|Log Model| F[MLflow Model Registry]
    E2 -->|Log Model| F
    F -->|Model Artifact| G[Model Serving API (FastAPI)]
    G -->|Prediction| H[User/API Client]
    B -->|Aggregated Summary| I[EDA Dashboard (Streamlit)]
    F -.->|Load Pipeline| J[Model Performance Dashboard]
    subgraph Orchestration
      K[Prefect Flow]
    end
    K -->|Run| B
    K -->|Launch| E2
    subgraph Experiment_Tracking
      L[MLflow]
    end
    E -.-> L
    E2 -.-> L
    J -.->|Fetch Run Data| L
```mermaid
flowchart TD
    %% Data Preparation
    A[Raw Data (Parquet)] -->|Dask Lazy Load| B[Base Feature Engineering]
    B -->|Filtered & Engineered| C[Processed Data Parquet]
    B -->|Aggregated Summary| I[EDA Dashboard (Streamlit)]

    %% Local/XGBoost Training Path
    C -->|Materialize| D1[Train/Test Split (Pandas)]
    D1 -->|Pandas DataFrame| E1[Pipeline Fit (TemporalFeatureEngineer + XGBRegressor)]
    E1 -->|Log Model| F1[MLflow Model Registry]
    F1 -->|Model Artifact| G1[Model Serving API (FastAPI)]
    G1 -->|Prediction| H1[User/API Client]
    F1 -.->|Load Pipeline| J1[Model Performance Dashboard]
    J1 -.->|Fetch Run Data| F1

    %% Distributed/PyTorch DDP Training Path
    C -->|Manifest File| M[Manifest JSON]
    M -->|Stream Parquet| D2[Distributed Training (PyTorch DDP)]
    D2 -->|Log Model| F2[MLflow Model Registry (DDP)]
    F2 -->|Model Artifact| G2[For future: PyTorch Serving]

    %% Orchestration
    subgraph Orchestration
      K[Prefect Flow]
    end
    K -.-> B
    K -.-> D2

    %% Experiment Tracking
    subgraph Experiment_Tracking
      L[MLflow]
    end
    E1 -.-> L
    D2 -.-> L
    F1 -.-> L
    F2 -.-> L
```
