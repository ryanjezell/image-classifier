#!/usr/bin/env python3
"""
train.py — Main training entry point.

Usage:
    python train.py                          # default config
    python train.py --config my.yaml         # custom config
    python train.py --lr-finder              # auto-detect best LR
    python train.py --show-batch             # preview augmented batch
    python train.py --quick                  # 1-epoch smoke-test
"""

import argparse
import sys
import logging
from pathlib import Path

from src.config_loader import load_config
from src.utils import set_global_seed, setup_logging, get_device, validate_dataset_structure
from src.data_pipeline import build_dataloaders
from src.model_builder import build_learner, find_learning_rate, export_model
from src.trainer import run_training


def parse_args():
    p = argparse.ArgumentParser(
        description="Train an image classifier using transfer learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--config', default='config/config.yaml',
                   help='Path to YAML config (default: config/config.yaml)')
    p.add_argument('--lr-finder', action='store_true',
                   help='Run LR finder and use suggested LR')
    p.add_argument('--show-batch', action='store_true',
                   help='Display an augmented sample batch before training')
    p.add_argument('--skip-validation', action='store_true',
                   help='Skip dataset structure pre-check')
    p.add_argument('--quick', action='store_true',
                   help='1-epoch smoke-test (useful for CI / debugging)')
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("INFO")

    logger.info("━" * 60)
    logger.info("  IMAGE CLASSIFIER — TRAINING PIPELINE")
    logger.info("━" * 60)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    if args.quick:
        cfg.training.head_epochs = 1
        cfg.training.finetune_epochs = 1
        logger.info("[QUICK] Smoke-test mode: 1 epoch per phase.")

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_global_seed(cfg.seed)
    get_device()

    # ── Dataset validation ────────────────────────────────────────────────────
    if not args.skip_validation:
        try:
            validate_dataset_structure(cfg.data.dataset_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"\n[ERROR] Dataset problem:\n  {e}\n")
            logger.error("  → Run:  python scripts/download_sample_data.py")
            sys.exit(1)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info("[MAIN] Building data pipeline …")
    try:
        dls = build_dataloaders(cfg)
    except Exception as e:
        logger.error(f"[ERROR] DataLoader build failed: {e}")
        sys.exit(1)

    if args.show_batch:
        try:
            dls.show_batch(max_n=9)
        except Exception:
            logger.warning("[MAIN] Could not display batch (no display).")

    # ── Learner ───────────────────────────────────────────────────────────────
    logger.info("[MAIN] Building model …")
    learn = build_learner(dls, cfg)

    # ── LR finder ────────────────────────────────────────────────────────────
    if args.lr_finder:
        suggested = find_learning_rate(learn)
        cfg.training.head_lr = suggested
        cfg.training.finetune_lr_max = suggested / 10
        cfg.training.finetune_lr_min = suggested / 1000
        logger.info(f"[MAIN] LRs updated from finder: head={suggested:.2e}")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("[MAIN] Starting training …")
    try:
        run_training(learn, cfg)
    except KeyboardInterrupt:
        logger.warning("[MAIN] Interrupted — saving partial model …")
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        raise

    # ── Export ────────────────────────────────────────────────────────────────
    export_model(learn, cfg.model.export_path)

    logger.info("")
    logger.info("  ✓ Model  →  " + cfg.model.export_path)
    logger.info("  ✓ Log    →  training.log")
    logger.info("  ✓ CSV    →  training_history.csv")
    logger.info("")
    logger.info("  Run predictions:")
    logger.info("    python predict.py --image path/to/image.jpg")
    logger.info("")


if __name__ == "__main__":
    main()
