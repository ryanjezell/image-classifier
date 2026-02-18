# ─────────────────────────────────────────────────────────────────────────────
# src/trainer.py  —  Two-phase training orchestration
#
# Phase 1  Backbone frozen   → train classification head only
# Phase 2  Backbone unfrozen → fine-tune all layers with discriminative LRs
# ─────────────────────────────────────────────────────────────────────────────

from fastai.vision.all import *
import logging

# NOTE: We do NOT import 'slice' because it is a Python built-in.
# FastAI uses the built-in slice() for discriminative learning rates.

def run_training(learn, cfg, quick: bool = False):
    """
    Runs the training cycle based on the provided configuration.
    Optimized for Colab to prevent flickering and progress bar crashes.
    """
    tc = cfg.training
    head_epochs = tc.head_epochs
    finetune_epochs = tc.finetune_epochs

    if quick:
        head_epochs = 1
        finetune_epochs = min(1, finetune_epochs)

    print(
        "🚀 Training schedule: "
        f"head={head_epochs} epochs @ {tc.head_lr} | "
        f"finetune={finetune_epochs} epochs @ slice({tc.finetune_lr_min}, {tc.finetune_lr_max})"
    )
    
    # We use learn.no_bar() to prevent the flickering purple box in Colab
    # We use learn.fit_one_cycle which internally handles discriminative LRs
    try:
        with learn.no_bar():
            if head_epochs > 0:
                print(f"🧠 Phase 1/2: training classifier head for {head_epochs} epoch(s)...")
                learn.fit_one_cycle(head_epochs, tc.head_lr)
            else:
                print("⏭️ Phase 1/2 skipped (head_epochs=0).")

            if finetune_epochs > 0:
                print(f"🔓 Phase 2/2: fine-tuning full model for {finetune_epochs} epoch(s)...")
                learn.unfreeze()
                learn.fit_one_cycle(
                    finetune_epochs,
                    slice(tc.finetune_lr_min, tc.finetune_lr_max)
                )
            else:
                print("⏭️ Phase 2/2 skipped (finetune_epochs=0).")
            
        print("✅ Training complete.")
        # Print a clean summary of the final metrics
        learn.recorder.print_log()
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise e

    return learn
