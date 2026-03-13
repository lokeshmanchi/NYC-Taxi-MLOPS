# NYC-Taxi-MLOPS

## Project Architecture Diagram

```mermaid
flowchart TD
		A[Raw Data Parquet] -->|Dask Lazy Load| B[Feature Engineering TemporalFeatureEngineer]
		B -->|Processed Data| C[Train Test Split]
		C -->|Pandas DataFrame| D[Pipeline Fit XGBRegressor]
		D -->|MLflow Log| E[Model Registry]
		E -->|Model Artifact| F[Model Serving API FastAPI Uvicorn]
		F -->|Prediction| G[User API Client]
		E -->|Model Load| F
		B -->|Aggregated Summary| H[EDA Dashboard Streamlit]
		D -->|Feature Importances| I[Model Performance Dashboard]
		subgraph Orchestration
			J[Prefect Flow]
		end
		J -.-> B
		J -.-> D
		J -.-> F
		subgraph Experiment_Tracking
			K[MLflow]
		end
		D -.-> K
		F -.-> K
		J -.-> K
```
