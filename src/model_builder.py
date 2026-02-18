# ─────────────────────────────────────────────────────────────────────────────
# src/model_builder.py  —  Learner construction, LR finder, model export
# ─────────────────────────────────────────────────────────────────────────────

import logging
from functools import partial
from pathlib import Path

import torch
from fastai.vision.all import (
    vision_learner, resnet34, resnet50,
    accuracy, top_k_accuracy,
    CrossEntropyLossFlat, MixUp,
    SaveModelCallback, EarlyStoppingCallback, CSVLogger,
)

try:
    FP16_AVAILABLE = True
except ImportError:
    FP16_AVAILABLE = False

from src.config_loader import AppConfig

# ── Architecture registry ────────────────────────────────────────────────────
ARCH_REGISTRY = {
    "resnet34": resnet34,
    "resnet50": resnet50,
}

try:                                    # EfficientNet requires timm
    import timm                         # noqa: F401
    from fastai.vision.all import timm_learner
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.info("[MODEL] timm not installed — EfficientNet unavailable. "
                 "Install with:  pip install timm")


def _get_arch(name: str):
    if name in ARCH_REGISTRY:
        return ARCH_REGISTRY[name], False          # (arch_fn, is_timm)
    if TIMM_AVAILABLE and name.startswith("efficientnet"):
        return name, True
    available = list(ARCH_REGISTRY.keys()) + (
        ["efficientnet_b0", "efficientnet_b3"] if TIMM_AVAILABLE else []
    )
    raise ValueError(f"Unknown architecture '{name}'. Available: {available}")


def build_learner(dls, cfg: AppConfig):
    """
    Constructs and returns a fast.ai Learner:
      • Pretrained ResNet/EfficientNet backbone (frozen by default)
      • Custom AdaptiveConcatPool head
      • CrossEntropyLoss with label smoothing
      • AdamW optimizer
      • MixUp, SaveModel, EarlyStopping, CSVLogger callbacks
      • FP16 mixed precision when CUDA is present
    """
    arch, is_timm = _get_arch(cfg.model.architecture)
    tc = cfg.training

    cbs = [
        MixUp(alpha=tc.mixup_alpha) if tc.mixup_alpha > 0 else None,
        SaveModelCallback(monitor='valid_loss', fname='best_model', min_delta=0.001),
        EarlyStoppingCallback(monitor='valid_loss', patience=3, min_delta=0.001),
        CSVLogger(fname="training_history.csv", append=True),
    ]
    cbs = [c for c in cbs if c is not None]

    kwargs = dict(
        dls=dls,
        metrics=[accuracy, partial(top_k_accuracy, k=3)],
        loss_func=CrossEntropyLossFlat(label_smoothing=tc.label_smoothing),
        wd=tc.weight_decay,
        pretrained=cfg.model.pretrained,
        cbs=cbs,
    )

    if is_timm:
        learn = timm_learner(arch=arch, **kwargs)
    else:
        learn = vision_learner(arch=arch, **kwargs)

    if torch.cuda.is_available() and FP16_AVAILABLE:
        learn = learn.to_fp16()
        logging.info("[MODEL] FP16 mixed precision enabled")

    total = sum(p.numel() for p in learn.model.parameters())
    trainable = sum(p.numel() for p in learn.model.parameters() if p.requires_grad)
    logging.info(f"[MODEL] {cfg.model.architecture}  |  "
                 f"params: {total:,} total / {trainable:,} trainable (head only)")
    return learn


def find_learning_rate(learn, num_iter: int = 100) -> float:
    """
    Runs the LR Range Test and returns the suggested valley LR.
    Saves lr_finder_plot.png for inspection.
    Model weights are restored after the test — safe to call at any time.
    """
    logging.info("[LR_FINDER] Running LR range test …")
    result = learn.lr_find(num_it=num_iter, suggest_funcs=('valley', 'steep'))
    logging.info(f"[LR_FINDER] valley={result.valley:.2e}  steep={result.steep:.2e}")
    fig = learn.recorder.plot_lr_find(return_fig=True)
    if fig:
        fig.savefig("lr_finder_plot.png", dpi=150, bbox_inches='tight')
        logging.info("[LR_FINDER] Plot → lr_finder_plot.png")
    return result.valley


def export_model(learn, export_path: str) -> None:
    """
    Exports the full Learner (weights + vocab + transforms) to a .pkl file.
    This single file is the only artifact needed for inference.
    """
    p = Path(export_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # CSVLogger can hold open file handles that are not pickleable.
    for cb in list(getattr(learn, "cbs", [])):
        if isinstance(cb, CSVLogger):
            learn.remove_cb(cb)

    learn.export(p)
    size_mb = p.stat().st_size / 1_048_576
    logging.info(f"[EXPORT] Saved to '{export_path}'  ({size_mb:.1f} MB)")
