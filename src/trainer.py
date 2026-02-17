# ─────────────────────────────────────────────────────────────────────────────
# src/trainer.py  —  Two-phase training orchestration
#
# Phase 1  Backbone frozen   → train classification head only
# Phase 2  Backbone unfrozen → fine-tune all layers with discriminative LRs
# ─────────────────────────────────────────────────────────────────────────────

import logging
from fastai.vision.all import slice    # discriminative LR slice

from src.config_loader import AppConfig


def run_training(learn, cfg: AppConfig) -> None:
    """
    Executes Phase 1 (head-only) then Phase 2 (full fine-tune).
    Modifies `learn` in place. Call export_model() afterwards.
    """
    tc = cfg.training

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    logging.info("━" * 60)
    logging.info(f"PHASE 1 — head training  (epochs={tc.head_epochs}, lr={tc.head_lr:.1e})")
    logging.info("━" * 60)

    learn.fit_one_cycle(tc.head_epochs, lr_max=tc.head_lr)

    m = learn.validate()
    logging.info(f"[P1] val_loss={m[0]:.4f}  accuracy={m[1]:.4f}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    logging.info("━" * 60)
    logging.info(f"PHASE 2 — full fine-tune  (epochs={tc.finetune_epochs}, "
                 f"lr={tc.finetune_lr_min:.1e}→{tc.finetune_lr_max:.1e})")
    logging.info("━" * 60)

    learn.unfreeze()

    learn.fit_one_cycle(
        tc.finetune_epochs,
        lr_max=slice(tc.finetune_lr_min, tc.finetune_lr_max),
    )

    m = learn.validate()
    logging.info("━" * 60)
    logging.info("TRAINING COMPLETE")
    logging.info(f"  val_loss  : {m[0]:.4f}")
    logging.info(f"  top-1 acc : {m[1]:.4f}  ({m[1]*100:.1f}%)")
    logging.info(f"  top-2 acc : {m[2]:.4f}  ({m[2]*100:.1f}%)")
    logging.info("━" * 60)

    # Reload best checkpoint saved by SaveModelCallback
    try:
        learn.load("best_model")
        logging.info("[TRAINER] Best model checkpoint loaded.")
    except Exception as e:
        logging.warning(f"[TRAINER] Could not load best checkpoint ({e}). Using final weights.")
