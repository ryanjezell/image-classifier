#!/usr/bin/env python3
"""evaluation.py — Comprehensive model evaluation via torch inference artifacts."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

from src.torch_pipeline.inference import TorchImageClassifier


VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def evaluate(model_path: str, dataset_path: str, cpu_only: bool = False) -> dict:
    clf = TorchImageClassifier(model_path, cpu_only=cpu_only)
    classes = clf.classes
    class_to_idx = {c: i for i, c in enumerate(classes)}

    root = Path(dataset_path)
    all_images = [p for p in root.rglob('*') if p.suffix.lower() in VALID_EXTS]
    logging.info("[EVAL] Evaluating %d images …", len(all_images))

    labels, preds, probs = [], [], []
    for img_path in all_images:
        true = img_path.parent.name
        if true not in class_to_idx:
            continue
        result = clf.predict(img_path)
        labels.append(class_to_idx[true])
        preds.append(class_to_idx[result["label"]])
        probs.append([result["all_probs"][c] for c in classes])

    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    top1 = float((labels == preds).mean()) if len(labels) else 0.0
    top3 = float(top_k_accuracy_score(labels, probs, k=min(3, len(classes)))) if len(labels) else 0.0
    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))

    results = {
        "summary": {
            "top1_accuracy": round(top1, 4),
            "top3_accuracy": round(top3, 4),
            "macro_f1": round(report['macro avg']['f1-score'], 4),
            "weighted_f1": round(report['weighted avg']['f1-score'], 4),
            "total_images": int(len(labels)),
            "num_classes": len(classes),
        },
        "per_class": {
            c: {k: round(report[c][k], 4) for k in ('precision', 'recall', 'f1-score', 'support')}
            for c in classes
        },
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }
    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    p = argparse.ArgumentParser(description="Evaluate a trained image classifier.")
    p.add_argument('--model', default='models/exported/model_state.pt')
    p.add_argument('--data', default='data/dataset')
    p.add_argument('--output', default='eval_report.json')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    results = evaluate(args.model, args.data, cpu_only=args.cpu)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Full report → '{args.output}'")


if __name__ == "__main__":
    main()
