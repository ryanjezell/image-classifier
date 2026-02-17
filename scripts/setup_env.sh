#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/setup_env.sh
#
# One-shot environment setup for Linux / macOS.
# Creates a Python virtual environment, installs all dependencies,
# and downloads sample data so you can train immediately.
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh
#
# Options:
#   --no-data    Skip sample data download
#   --cpu-only   Install CPU-only PyTorch (smaller, no CUDA required)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_DIR=".venv"
SKIP_DATA=false
CPU_ONLY=false

# ── Parse flags ───────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --no-data)   SKIP_DATA=true  ;;
    --cpu-only)  CPU_ONLY=true   ;;
  esac
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  IMAGE CLASSIFIER — Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Python version check ──────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "❌  Python not found. Install Python 3.9+ and try again."
  exit 1
fi
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python  : $PY_VERSION  ($PYTHON)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  echo "  venv    : $VENV_DIR already exists — skipping creation."
else
  echo "  venv    : creating $VENV_DIR …"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "  venv    : activated ($VENV_DIR)"

# Upgrade pip silently
pip install --upgrade pip --quiet

# ── PyTorch ───────────────────────────────────────────────────────────────────
if $CPU_ONLY; then
  echo "  torch   : installing CPU-only build …"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
else
  echo "  torch   : installing (GPU if available, CPU fallback) …"
  pip install torch torchvision --quiet
fi

# ── Remaining dependencies ────────────────────────────────────────────────────
echo "  deps    : installing requirements.txt …"
pip install -r requirements.txt --quiet

echo "  ✓ All packages installed."

# ── Create required directories ───────────────────────────────────────────────
mkdir -p data/dataset models/exported

# ── Sample data ──────────────────────────────────────────────────────────────
if $SKIP_DATA; then
  echo "  data    : skipped (--no-data flag)."
  echo "  → Populate data/dataset/<classname>/ with your images."
else
  echo "  data    : downloading sample dataset …"
  python scripts/download_sample_data.py
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Setup complete!"
echo ""
echo "  Next steps:"
echo ""
echo "    1. Activate environment (if not already active):"
echo "         source $VENV_DIR/bin/activate"
echo ""
echo "    2. Train:"
echo "         python train.py"
echo "         python train.py --lr-finder   # auto-detect best LR"
echo ""
echo "    3. Predict:"
echo "         python predict.py --image path/to/image.jpg"
echo ""
echo "    4. Evaluate:"
echo "         python evaluation.py"
echo ""
echo "    5. Serve API (optional):"
echo "         uvicorn api:app --host 0.0.0.0 --port 8080"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
