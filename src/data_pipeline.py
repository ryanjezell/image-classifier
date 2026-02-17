# ─────────────────────────────────────────────────────────────────────────────
# src/data_pipeline.py  —  DataBlock construction, augmentation, DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path

from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock,
    get_image_files, RandomSplitter, parent_label,
    RandomResizedCrop, aug_transforms,
    Normalize, imagenet_stats,
)

from src.config_loader import AppConfig


def build_augmentation_transforms(cfg: AppConfig) -> list:
    """
    Returns GPU-accelerated training augmentations from config.
    Applied to training data ONLY — never validation.
    """
    a = cfg.augmentation
    tfms = aug_transforms(
        do_flip=a.flip_horiz,
        max_rotate=a.max_rotate,
        max_zoom=a.max_zoom,
        max_lighting=a.max_lighting,
        max_warp=a.max_warp,
        p_affine=a.p_affine,
    )
    logging.info(
        f"[AUG] flip={a.flip_horiz}  rotate=±{a.max_rotate}°  "
        f"zoom={a.max_zoom}x  lighting=±{a.max_lighting}"
    )
    return tfms


def build_dataloaders(cfg: AppConfig):
    """
    Builds fast.ai DataLoaders from an ImageFolder-style directory.

    Expected layout:
        data/dataset/
            <class_a>/  image1.jpg  image2.jpg ...
            <class_b>/  ...
            <class_c>/  ...

    Returns:
        fast.ai ImageDataLoaders ready for training.
    """
    path = Path(cfg.data.dataset_path)
    files = get_image_files(path)
    if not files:
        raise ValueError(f"No images found under '{path}'.")

    logging.info(f"[DATA] {len(files)} images found in '{path}'")

    aug_tfms = build_augmentation_transforms(cfg)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=cfg.data.valid_pct, seed=42),
        get_y=parent_label,
        item_tfms=RandomResizedCrop(cfg.data.image_size, min_scale=0.8),
        batch_tfms=[*aug_tfms, Normalize.from_stats(*imagenet_stats)],
    )

    dls = dblock.dataloaders(path, bs=cfg.data.batch_size, num_workers=4)

    logging.info(f"[DATA] Classes ({len(dls.vocab)}): {list(dls.vocab)}")
    logging.info(f"[DATA] Train batches: {len(dls.train)}  |  Val batches: {len(dls.valid)}")
    return dls
