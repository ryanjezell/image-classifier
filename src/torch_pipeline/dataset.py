from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.torch_pipeline.transforms import build_train_transform, build_eval_transform


def _split_indices(length: int, valid_pct: float, seed: int):
    n_valid = max(1, int(length * valid_pct))
    n_train = length - n_valid
    if n_train <= 0:
        raise ValueError("Dataset split would leave no training samples.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(length, generator=g).tolist()
    return perm[:n_train], perm[n_train:]


def build_dataloaders(cfg):
    root = Path(cfg.data.dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path '{root}' not found")

    full = ImageFolder(root=root)
    if len(full.samples) == 0:
        raise ValueError(f"No images found under '{root}'.")

    train_idx, valid_idx = _split_indices(len(full.samples), cfg.data.valid_pct, cfg.seed)

    train_ds = Subset(ImageFolder(root=root, transform=build_train_transform(cfg)), train_idx)
    valid_ds = Subset(ImageFolder(root=root, transform=build_eval_transform(cfg)), valid_idx)

    num_workers = 0 if cfg.data.batch_size < 4 else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": train_loader,
        "valid": valid_loader,
        "classes": list(full.classes),
    }
