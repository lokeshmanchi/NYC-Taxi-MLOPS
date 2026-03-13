.PHONY: build up down train logs clean ps

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

train:
	docker compose --profile training run --rm trainer

logs:
	docker compose logs -f

ps:
	docker compose ps

clean:
	docker compose down -v --remove-orphans

# Run training locally (outside Docker) — requires venv with requirements.txt installed
train-local:
	python -m src.pipelines.training_flow

# Run API locally
serve-local:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit locally
ui-local:
	streamlit run src/ui/Home.py
