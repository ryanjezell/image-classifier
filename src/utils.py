# ─────────────────────────────────────────────────────────────────────────────
# src/utils.py  —  Shared utility functions
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import logging
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Seeds Python, NumPy, and PyTorch (CPU + CUDA) for full reproducibility.
    Must be called BEFORE any model creation or data loading.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"[SEED] Global random seed set to {seed}")


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configures structured logging to both console and file."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log", mode='a'),
        ]
    )
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """Returns the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"[DEVICE] CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("[DEVICE] Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logging.warning("[DEVICE] No GPU detected — training on CPU will be slow.")
    return device


def validate_dataset_structure(dataset_path: str, min_images_per_class: int = 20) -> dict:
    """
    Validates dataset directory: must contain >= 2 class subdirs,
    each with >= min_images_per_class valid image files.

    Returns dict of {class_name: image_count}.
    Raises FileNotFoundError or ValueError on failure.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset path '{dataset_path}' not found.\n"
            "  → Run:  python scripts/download_sample_data.py\n"
            "  → Or:   populate data/dataset/<classname>/ with your images."
        )

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    class_info = {}

    entries = [e for e in os.scandir(dataset_path) if e.is_dir()]
    if len(entries) < 2:
        raise ValueError(
            f"Dataset must have ≥ 2 class folders. Found {len(entries)} in '{dataset_path}'."
        )

    for entry in sorted(entries):
        count = sum(
            1 for f in os.scandir(entry.path)
            if f.is_file() and os.path.splitext(f.name)[1].lower() in valid_ext
        )
        class_info[entry.name] = count
        if count < min_images_per_class:
            raise ValueError(
                f"Class '{entry.name}' has only {count} images (min {min_images_per_class}).\n"
                "  → Add more images or lower min_images_per_class."
            )

    logging.info(f"[DATASET] {len(class_info)} classes found:")
    for cls, cnt in class_info.items():
        logging.info(f"  └─ {cls}: {cnt} images")
    return class_info
