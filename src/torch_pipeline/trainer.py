import copy

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


def _run_epoch(model, loader, criterion, optimizer, device, train: bool, scheduler=None, step_per_batch: bool = False):
    model.train(train)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None and step_per_batch:
                    scheduler.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def train_model(model, dataloaders, cfg, device):
    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]

    epochs = max(1, int(cfg.training.head_epochs) + int(cfg.training.finetune_epochs))
    lr = float(cfg.training.head_lr)
    min_lr = float(cfg.training.finetune_lr_min)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.training.label_smoothing))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.training.weight_decay))

    scheduler_name = getattr(cfg.training, "scheduler", "onecycle").lower()
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        step_per_batch = False
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.2,
            div_factor=max(lr / max(min_lr, 1e-8), 1.0),
        )
        step_per_batch = True

    best = {"acc": 0.0, "state_dict": copy.deepcopy(model.state_dict())}
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, True, scheduler, step_per_batch
        )
        valid_loss, valid_acc = _run_epoch(model, valid_loader, criterion, optimizer, device, False)

        if scheduler is not None and not step_per_batch:
            scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "valid_loss": round(valid_loss, 6),
            "valid_acc": round(valid_acc, 6),
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )

        if valid_acc >= best["acc"]:
            best["acc"] = valid_acc
            best["state_dict"] = copy.deepcopy(model.state_dict())

    model.load_state_dict(best["state_dict"])
    return model, history
