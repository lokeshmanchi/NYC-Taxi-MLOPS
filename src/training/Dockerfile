FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir pandas pyarrow numpy scikit-learn xgboost joblib mlflow prefect python-dotenv

COPY src/ src/

CMD ["python", "-m", "src.training.train"]
