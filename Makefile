# ─────────────────────────────────────────────────────────────────────────────
# Makefile  —  Convenience targets for common tasks
#
# Usage:
#   make setup       # full first-time setup
#   make train       # train with default config
#   make predict     # predict on a test image (set IMG=path/to/img.jpg)
#   make evaluate    # run evaluation report
#   make api         # start REST API server
#   make clean       # remove generated files
#
# Requires:
#   GNU Make (Linux/macOS built-in; Windows: choco install make)
# ─────────────────────────────────────────────────────────────────────────────

VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
IMG     ?= data/test_image.jpg
PORT    ?= 8080

.PHONY: all setup data train train-quick train-lrfind predict evaluate api \
        docker-build docker-run clean help

all: help

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@bash scripts/setup_env.sh

setup-no-data:
	@bash scripts/setup_env.sh --no-data

setup-cpu:
	@bash scripts/setup_env.sh --cpu-only

# ── Data ─────────────────────────────────────────────────────────────────────
data:
	$(PYTHON) scripts/download_sample_data.py

data-custom:
	$(PYTHON) scripts/download_sample_data.py --classes cat dog bird --limit 200

# ── Training ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) train.py

train-quick:
	$(PYTHON) train.py --quick

train-lrfind:
	$(PYTHON) train.py --lr-finder

# ── Prediction ────────────────────────────────────────────────────────────────
predict:
	$(PYTHON) predict.py --image $(IMG)

predict-json:
	$(PYTHON) predict.py --image $(IMG) --output-format json

predict-folder:
	$(PYTHON) predict.py --folder data/test/

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) evaluation.py

# ── API server ────────────────────────────────────────────────────────────────
api:
	$(VENV)/bin/uvicorn api:app --host 0.0.0.0 --port $(PORT) --reload

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t image-classifier:latest .

docker-run:
	docker run -p 8080:8080 -v $(PWD)/models:/app/models image-classifier:latest

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f training.log training_history.csv lr_finder_plot.png eval_report.json
	rm -f models/saved/best_model.pth

clean-all: clean
	rm -rf $(VENV) models/exported/

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  IMAGE CLASSIFIER — Make targets"
	@echo "  ────────────────────────────────────────────────────"
	@echo "  make setup          First-time setup (venv + deps + data)"
	@echo "  make setup-cpu      Setup with CPU-only PyTorch"
	@echo "  make setup-no-data  Setup without downloading data"
	@echo ""
	@echo "  make data           Download sample dataset"
	@echo "  make train          Train the model"
	@echo "  make train-quick    1-epoch smoke test"
	@echo "  make train-lrfind   Train with auto LR detection"
	@echo ""
	@echo "  make predict IMG=path/to/image.jpg"
	@echo "  make predict-json   IMG=path/to/image.jpg"
	@echo "  make predict-folder"
	@echo ""
	@echo "  make evaluate       Full evaluation report"
	@echo "  make api            Start REST API on port 8080"
	@echo ""
	@echo "  make clean          Remove logs and temp files"
	@echo "  make clean-all      Remove venv and exported models too"
	@echo ""
