# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile
#
# Builds a production container for the classifier API.
#
# Build:
#   docker build -t image-classifier:latest .
#
# Run:
#   docker run -p 8080:8080 -v $(pwd)/models:/app/models image-classifier:latest
#
# Health check:
#   curl http://localhost:8080/health
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies required by Pillow / OpenCV / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libgomp1 wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (smaller image, no CUDA drivers needed)
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/        src/
COPY predict.py  .
COPY api.py      .
COPY config/     config/

# Model is mounted at runtime (not baked into the image)
RUN mkdir -p models/exported

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
