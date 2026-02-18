import json
from pathlib import Path

import torch

from src.torch_pipeline.transforms import IMAGENET_MEAN, IMAGENET_STD


ARTIFACT_MODEL = "model_state.pt"
ARTIFACT_CLASSES = "classes.json"
ARTIFACT_CONFIG = "inference_config.json"


def _resolve_export_dir(export_path: str) -> Path:
    p = Path(export_path)
    if p.suffix:
        return p.parent
    return p


def save_training_artifacts(model, classes, cfg):
    out_dir = _resolve_export_dir(cfg.model.export_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / ARTIFACT_MODEL)
    (out_dir / ARTIFACT_CLASSES).write_text(json.dumps(list(classes), indent=2))

    inference_cfg = {
        "architecture": cfg.model.architecture,
        "image_size": cfg.data.image_size,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "num_classes": len(classes),
    }
    (out_dir / ARTIFACT_CONFIG).write_text(json.dumps(inference_cfg, indent=2))

    return out_dir


def load_inference_artifacts(model_path: str):
    p = Path(model_path)
    artifact_dir = p if p.is_dir() else p.parent

    model_state_path = artifact_dir / ARTIFACT_MODEL
    classes_path = artifact_dir / ARTIFACT_CLASSES
    inference_path = artifact_dir / ARTIFACT_CONFIG

    if not model_state_path.exists():
        raise FileNotFoundError(f"Missing artifact: {model_state_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing artifact: {classes_path}")
    if not inference_path.exists():
        raise FileNotFoundError(f"Missing artifact: {inference_path}")

    classes = json.loads(classes_path.read_text())
    inference_cfg = json.loads(inference_path.read_text())

    return artifact_dir, model_state_path, classes, inference_cfg
