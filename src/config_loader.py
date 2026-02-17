# ─────────────────────────────────────────────────────────────────────────────
# src/config_loader.py  —  Typed YAML config loader
# ─────────────────────────────────────────────────────────────────────────────

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field


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
        seed=project.get('seed', 42),
        data=DataConfig(
            dataset_path=d.get('dataset_path', 'data/dataset'),
            image_size=d.get('image_size', 224),
            valid_pct=d.get('valid_pct', 0.2),
            batch_size=d.get('batch_size', 32),
        ),
        model=ModelConfig(
            architecture=m.get('architecture', 'resnet50'),
            pretrained=m.get('pretrained', True),
            export_path=m.get('export_path', 'models/exported/classifier.pkl'),
        ),
        training=TrainingConfig(
            head_epochs=t.get('head_epochs', 4),
            head_lr=t.get('head_lr', 1e-3),
            finetune_epochs=t.get('finetune_epochs', 10),
            finetune_lr_min=t.get('finetune_lr_min', 1e-6),
            finetune_lr_max=t.get('finetune_lr_max', 1e-4),
            dropout=t.get('dropout', 0.5),
            weight_decay=t.get('weight_decay', 0.01),
            label_smoothing=t.get('label_smoothing', 0.1),
            mixup_alpha=t.get('mixup_alpha', 0.4),
        ),
        augmentation=AugmentationConfig(
            flip_horiz=a.get('flip_horiz', True),
            max_rotate=a.get('max_rotate', 15.0),
            max_zoom=a.get('max_zoom', 1.15),
            max_lighting=a.get('max_lighting', 0.2),
            max_warp=a.get('max_warp', 0.2),
            p_affine=a.get('p_affine', 0.75),
        ),
    )

    logging.info(f"[CONFIG] Loaded '{config_path}' — arch={cfg.model.architecture}, "
                 f"size={cfg.data.image_size}px, bs={cfg.data.batch_size}")
    return cfg
