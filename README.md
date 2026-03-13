# NYC-Taxi-MLOPS

## Project Architecture Diagram

```mermaid
flowchart TD
		A[Raw Data (Parquet)] -->|Dask Lazy Load| B[Feature Engineering<br>(TemporalFeatureEngineer)]
		B -->|Processed Data| C[Train/Test Split]
		C -->|Pandas DataFrame| D[Pipeline Fit<br>(XGBRegressor)]
		D -->|MLflow Log| E[Model Registry]
		E -->|Model Artifact| F[Model Serving API<br>(FastAPI/Uvicorn)]
		F -->|Prediction| G[User/API Client]
		E -->|Model Load| F
		B -->|Aggregated Summary| H[EDA Dashboard<br>(Streamlit)]
		D -->|Feature Importances| I[Model Performance Dashboard]
		subgraph Orchestration
			J[Prefect Flow]
		end
		J -.-> B
		J -.-> D
		J -.-> F
		subgraph Experiment Tracking
			K[MLflow]
		end
		D -.-> K
		F -.-> K
		J -.-> K
```
