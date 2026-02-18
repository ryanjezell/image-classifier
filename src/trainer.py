# ─────────────────────────────────────────────────────────────────────────────
# src/trainer.py  —  Two-phase training orchestration
#
# Phase 1  Backbone frozen   → train classification head only
# Phase 2  Backbone unfrozen → fine-tune all layers with discriminative LRs
# ─────────────────────────────────────────────────────────────────────────────

import logging


def run_training(learn, cfg):
    """
    Runs two-phase transfer learning based on config.
    """
    tc = cfg.training

    logging.info(
        "[TRAIN] Phase 1: freeze + fit_one_cycle(epochs=%s, lr=%s)",
        tc.head_epochs,
        tc.head_lr,
    )
    if tc.head_epochs > 0:
        learn.freeze()
        learn.fit_one_cycle(tc.head_epochs, lr_max=tc.head_lr)

    logging.info(
        "[TRAIN] Phase 2: unfreeze + fit_one_cycle(epochs=%s, lr=slice(%s, %s))",
        tc.finetune_epochs,
        tc.finetune_lr_min,
        tc.finetune_lr_max,
    )
    if tc.finetune_epochs > 0:
        learn.unfreeze()
        learn.fit_one_cycle(
            tc.finetune_epochs,
            lr_max=slice(tc.finetune_lr_min, tc.finetune_lr_max),
        )

    logging.info("[TRAIN] Training complete")
    return learn
