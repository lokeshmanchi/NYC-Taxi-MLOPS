.PHONY: build up down train logs clean ps
# Makefile for NYC Taxi MLOps
# Defines hooks and standard commands for AI agents and developers.

.PHONY: build up train logs down clean lint format pre-train post-train test

# ─── HOOKS ────────────────────────────────────────────────────────────────────

# Pre-hook: Runs before the main training logic
pre-train:
	@echo "🪝 [PRE-HOOK] Running safety checks and formatting..."
	@# Example: Ensure code is formatted before we run it
	@$(MAKE) format
	@# Example: Check if data exists
	@[ -d "data" ] || (echo "❌ Data directory missing!" && exit 1)

# Post-hook: Runs after training completes successfully
post-train:
	@echo "🪝 [POST-HOOK] performing cleanup..."
	@# Example: Notify or clean up temp files
	@find . -name "*.tmp" -delete
	@echo "✅ Pipeline sequence complete."

# ─── COMMANDS ─────────────────────────────────────────────────────────────────

# Matches CLAUDE.md: "Build Docker images"
build:
	docker compose build

# Matches CLAUDE.md: "Start MLflow, Prefect, API, Streamlit"
up:
	docker compose up -d

down:
	docker compose down
# Matches CLAUDE.md: "Run training pipeline"
# We wrap the actual command with our pre/post hooks here.
train: pre-train
	@echo "🚀 Starting Training..."
	python -m src.training.train
	@$(MAKE) post-train

# Matches CLAUDE.md: "Tail all service logs"
logs:
	docker compose logs -f

ps:
	docker compose ps
down:
	docker compose down

clean:
	docker compose down -v --remove-orphans
	docker compose down -v
	rm -rf data/processed
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Run training locally (outside Docker) — requires venv with requirements.txt installed
train-local:
	python -m src.pipelines.training_flow

# Run API locally
serve-local:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit locally
ui-local:
	streamlit run src/ui/Home.py
# Utility for the pre-hook
format:
	@echo "Running code formatters..."
	# pip install black isort
	# black src && isort src

# Run the full regression test suite
test:
	python -m pytest tests/ -v
