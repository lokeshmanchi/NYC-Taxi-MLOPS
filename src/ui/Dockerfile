FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir pandas pyarrow numpy mlflow streamlit plotly httpx python-dotenv scikit-learn xgboost joblib

COPY src/ src/

CMD ["streamlit", "run", "src/ui/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
