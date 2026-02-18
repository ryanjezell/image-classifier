import argparse
import sys

from src.config_loader import load_config
from src.data_pipeline import build_dataloaders
from src.model_builder import build_learner, find_learning_rate, export_model
from src.trainer import run_training
from src.utils import setup_logging, set_global_seed, validate_dataset_structure


def parse_args():
    p = argparse.ArgumentParser(
        description="Train an image classifier using transfer learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python train.py --config config/config.yaml --lr-finder",
    )
    p.add_argument('--config', default='config/config.yaml',
                   help='Path to YAML config (default: config/config.yaml)')
    p.add_argument('--lr-finder', action='store_true',
                   help='Run LR finder and print suggested LR')
    p.add_argument('--show-batch', action='store_true',
                   help='Display an augmented sample batch before training')
    p.add_argument('--skip-validation', action='store_true',
                   help='Skip dataset structure pre-check')
    p.add_argument('--quick', action='store_true',
                   help='1-epoch smoke-test (useful for CI / debugging)')
    return p.parse_args()


def main():
    args = parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    if args.quick:
        print("⚡ Running quick smoke-test (1 epoch)...")
        cfg.training.head_epochs = 1
        cfg.training.finetune_epochs = 0
        cfg.model.pretrained = False

    if not args.skip_validation:
        validate_dataset_structure(cfg.data.dataset_path)

    print("📦 Preparing data...")
    dls = build_dataloaders(cfg)

    if args.show_batch:
        print("🖼️ Showing sample batch...")
        dls.show_batch(max_n=9)

    learn = build_learner(dls, cfg)

    if args.lr_finder:
        print("🔍 Finding optimal learning rate...")
        suggested_lr = find_learning_rate(learn)
        print(f"✅ Suggested LR (Valley): {suggested_lr:.2e}")
        return

    learn = run_training(learn, cfg)
    export_model(learn, cfg.model.export_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)
