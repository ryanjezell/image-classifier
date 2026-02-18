import argparse
import json
import sys

from src.config_loader import load_config
from src.torch_pipeline import build_dataloaders, create_model, save_training_artifacts, train_model
from src.utils import get_device, set_global_seed, setup_logging, validate_dataset_structure


def parse_args():
    p = argparse.ArgumentParser(
        description="Train an image classifier using torchvision + PyTorch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python train.py --config config/config.yaml --quick",
    )
    p.add_argument('--config', default='config/config.yaml', help='Path to YAML config')
    p.add_argument('--lr-finder', action='store_true', help='Show configured LR and exit')
    p.add_argument('--show-batch', action='store_true', help='Show one training batch tensor shape')
    p.add_argument('--skip-validation', action='store_true', help='Skip dataset structure pre-check')
    p.add_argument('--quick', action='store_true', help='1-epoch smoke-test')
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    if not args.skip_validation:
        validate_dataset_structure(cfg.data.dataset_path, min_images_per_class=1)

    dls = build_dataloaders(cfg)
    if args.show_batch:
        xb, yb = next(iter(dls["train"]))
        print(f"Batch tensors: images={tuple(xb.shape)}, labels={tuple(yb.shape)}")

    if args.lr_finder:
        print(f"Configured training LR: {cfg.training.head_lr}")
        return

    if args.quick:
        cfg.training.head_epochs = 1
        cfg.training.finetune_epochs = 0

    model = create_model(
        architecture=cfg.model.architecture,
        num_classes=len(dls["classes"]),
        pretrained=cfg.model.pretrained,
        dropout=cfg.training.dropout,
    )

    device = get_device()
    model = model.to(device)
    model, history = train_model(model, dls, cfg, device)

    out_dir = save_training_artifacts(model, dls["classes"], cfg)
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"✅ Saved artifacts in: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)
