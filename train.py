import argparse
import logging
import sys
from pathlib import Path

# Custom module imports
from src.utils import load_config, setup_logging, prepare_data
from src.model_builder import get_model, find_learning_rate, export_model
from src.trainer import run_training

def parse_args():
    # We remove __doc__ to prevent the NameError and provide a clean string instead
    p = argparse.ArgumentParser(
        description="Train an image classifier using transfer learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python train.py --config config/config.yaml --lr-finder"
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
    
    # 1. Setup
    setup_logging()
    cfg = load_config(args.config)
    
    # 2. Data
    print("📦 Preparing data...")
    dls = prepare_data(cfg)
    
    if args.show_batch:
        print("🖼️ Showing sample batch...")
        dls.show_batch(max_n=9)
        # Note: In Colab, you might need plt.show() if not using %matplotlib inline
    
    # 3. Model
    learn = get_model(dls)
    
    # 4. Learning Rate Logic
    if args.lr_finder:
        print("🔍 Finding optimal learning rate...")
        # This uses the stable suggested_funcs we put in model_builder
        suggested_lr = find_learning_rate(learn)
        print(f"✅ Suggested LR (Valley): {suggested_lr.valley:.2e}")
        return # Stop here as requested by --lr-finder flag

    # 5. Training
    if args.quick:
        print("⚡ Running quick smoke-test (1 epoch)...")
        cfg.training.epochs = 1
        
    learn = run_training(learn, cfg)
    
    # 6. Export
    export_model(learn, cfg.model.export_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)