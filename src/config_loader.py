# ─────────────────────────────────────────────────────────────────────────────
# src/config_loader.py  —  Typed YAML config loader
# ─────────────────────────────────────────────────────────────────────────────

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    return default


@dataclass
class TrainingConfig:
    head_epochs: int = 4
    head_lr: float = 1e-3
    finetune_epochs: int = 10
    finetune_lr_min: float = 1e-6
    finetune_lr_max: float = 1e-4
    dropout: float = 0.5
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.4


@dataclass
class DataConfig:
    dataset_path: str = "data/dataset"
    image_size: int = 224
    valid_pct: float = 0.2
    batch_size: int = 32


@dataclass
class ModelConfig:
    architecture: str = "resnet50"
    pretrained: bool = True
    export_path: str = "models/exported/classifier.pkl"


@dataclass
class AugmentationConfig:
    flip_horiz: bool = True
    max_rotate: float = 15.0
    max_zoom: float = 1.15
    max_lighting: float = 0.2
    max_warp: float = 0.2
    p_affine: float = 0.75


@dataclass
class AppConfig:
    project_name: str = "image_classifier"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Loads YAML config and returns a typed AppConfig. Falls back to defaults if missing."""
    p = Path(config_path)
    if not p.exists():
        logging.warning(f"[CONFIG] '{config_path}' not found — using defaults.")
        return AppConfig()

    with open(p) as f:
        raw = yaml.safe_load(f)

    project = raw.get('project', {})
    d = raw.get('data', {})
    m = raw.get('model', {})
    t = raw.get('training', {})
    a = raw.get('augmentation', {})

    cfg = AppConfig(
        project_name=project.get('name', 'image_classifier'),
        seed=_as_int(project.get('seed', 42), 42),
        data=DataConfig(
            dataset_path=d.get('dataset_path', 'data/dataset'),
            image_size=_as_int(d.get('image_size', 224), 224),
            valid_pct=_as_float(d.get('valid_pct', 0.2), 0.2),
            batch_size=_as_int(d.get('batch_size', 32), 32),
        ),
        model=ModelConfig(
            architecture=m.get('architecture', 'resnet50'),
            pretrained=_as_bool(m.get('pretrained', True), True),
            export_path=m.get('export_path', 'models/exported/classifier.pkl'),
        ),
        training=TrainingConfig(
            head_epochs=_as_int(t.get('head_epochs', 4), 4),
            head_lr=_as_float(t.get('head_lr', 1e-3), 1e-3),
            finetune_epochs=_as_int(t.get('finetune_epochs', 10), 10),
            finetune_lr_min=_as_float(t.get('finetune_lr_min', 1e-6), 1e-6),
            finetune_lr_max=_as_float(t.get('finetune_lr_max', 1e-4), 1e-4),
            dropout=_as_float(t.get('dropout', 0.5), 0.5),
            weight_decay=_as_float(t.get('weight_decay', 0.01), 0.01),
            label_smoothing=_as_float(t.get('label_smoothing', 0.1), 0.1),
            mixup_alpha=_as_float(t.get('mixup_alpha', 0.4), 0.4),
        ),
        augmentation=AugmentationConfig(
            flip_horiz=_as_bool(a.get('flip_horiz', True), True),
            max_rotate=_as_float(a.get('max_rotate', 15.0), 15.0),
            max_zoom=_as_float(a.get('max_zoom', 1.15), 1.15),
            max_lighting=_as_float(a.get('max_lighting', 0.2), 0.2),
            max_warp=_as_float(a.get('max_warp', 0.2), 0.2),
            p_affine=_as_float(a.get('p_affine', 0.75), 0.75),
        ),
    )

    logging.info(f"[CONFIG] Loaded '{config_path}' — arch={cfg.model.architecture}, "
                 f"size={cfg.data.image_size}px, bs={cfg.data.batch_size}")
    return cfg
