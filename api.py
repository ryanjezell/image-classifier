#!/usr/bin/env python3
"""
api.py — FastAPI REST server for production inference.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8080

Endpoints:
    POST /classify   multipart image upload → JSON prediction
    GET  /health     liveness / readiness check
    GET  /classes    list all trained classes
"""

import shutil
import tempfile
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    raise SystemExit("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

from predict import ImageClassifier

# ── Startup ──────────────────────────────────────────────────────────────────
# Model is loaded ONCE at server startup — not per request.
# Reloading on every request adds 1-5 s latency and is unnecessary.
MODEL_PATH = "models/exported/model_state.pt"
try:
    classifier = ImageClassifier(MODEL_PATH, cpu_only=True)
except FileNotFoundError as e:
    raise SystemExit(str(e))

app = FastAPI(
    title="Image Classifier API",
    description="Multi-class image classification via transfer learning",
    version="1.0.0",
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@app.post("/classify", summary="Classify an uploaded image")
async def classify(file: UploadFile = File(...)):
    """
    Upload an image (JPEG / PNG / WebP) and receive a JSON classification result.

    Returns:
        {
          "image_path": "<filename>",
          "label":      "cat",
          "confidence": 0.9821,
          "all_probs":  {"cat": 0.9821, "dog": 0.0134, "bird": 0.0045},
          "top_3":      [["cat", 0.9821], ...]
        }
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = classifier.predict(tmp_path)
        result["image_path"] = file.filename   # Return original filename, not temp path
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health", summary="Health / readiness check")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "classes": classifier.classes,
        "num_classes": len(classifier.classes),
    }


@app.get("/classes", summary="List trained classes")
async def classes():
    return {"classes": classifier.classes}
