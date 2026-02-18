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

def run_training(learn, cfg):
    """
    Runs the training cycle based on the provided configuration.
    Optimized for Colab to prevent flickering and progress bar crashes.
    """
    epochs = cfg.training.epochs
    lr_max = cfg.training.lr_max
    
    print(f"🚀 Training for {epochs} epochs with max learning rate: {lr_max}")
    
    # We use learn.no_bar() to prevent the flickering purple box in Colab
    # We use learn.fit_one_cycle which internally handles discriminative LRs
    try:
        with learn.no_bar():
            # If lr_max is a slice (e.g. slice(1e-6, 1e-4)), 
            # FastAI handles it automatically.
            learn.fit_one_cycle(epochs, lr_max)
            
        print("✅ Training complete.")
        # Print a clean summary of the final metrics
        learn.recorder.print_log()
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise e

    return learn