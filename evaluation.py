#!/usr/bin/env python3
"""
evaluation.py — Comprehensive model evaluation.

Computes: top-1/top-3 accuracy, per-class P/R/F1, confusion matrix.

Usage:
    python evaluation.py
    python evaluation.py --model models/exported/classifier.pkl --data data/dataset
    python evaluation.py --output eval_report.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from fastai.vision.all import load_learner, get_image_files, parent_label, PILImage
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score


def evaluate(model_path: str, dataset_path: str) -> dict:
    learn = load_learner(model_path)
    classes = list(learn.dls.vocab)
    all_images = get_image_files(Path(dataset_path))

    logging.info(f"[EVAL] Evaluating {len(all_images)} images …")

    labels, preds, probs = [], [], []
    for img_path in all_images:
        true = parent_label(img_path)
        if true not in classes:
            continue
        _, idx, p = learn.predict(PILImage.create(img_path))
        labels.append(classes.index(true))
        preds.append(int(idx))
        probs.append([float(x) for x in p])

    labels = np.array(labels)
    preds  = np.array(preds)
    probs  = np.array(probs)

    top1 = (labels == preds).mean()
    top3 = top_k_accuracy_score(labels, probs, k=min(3, len(classes)))
    report = classification_report(labels, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(labels, preds)

    results = {
        "summary": {
            "top1_accuracy":  round(float(top1), 4),
            "top3_accuracy":  round(float(top3), 4),
            "macro_f1":       round(report['macro avg']['f1-score'], 4),
            "weighted_f1":    round(report['weighted avg']['f1-score'], 4),
            "total_images":   int(len(labels)),
            "num_classes":    len(classes),
        },
        "per_class": {
            c: {k: round(report[c][k], 4) for k in ('precision', 'recall', 'f1-score', 'support')}
            for c in classes
        },
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EVALUATION REPORT")
    print("═"*60)
    print(f"  Top-1 accuracy : {top1*100:.2f}%")
    print(f"  Top-3 accuracy : {top3*100:.2f}%")
    print(f"  Macro  F1      : {report['macro avg']['f1-score']*100:.2f}%")
    print(f"  Weighted F1    : {report['weighted avg']['f1-score']*100:.2f}%")
    print()
    print("  Per-class breakdown:")
    for c in classes:
        r = results["per_class"][c]
        print(f"    {c:>16}:  P={r['precision']:.3f}  R={r['recall']:.3f}  "
              f"F1={r['f1-score']:.3f}  n={int(r['support'])}")
    print()
    print("  Confusion matrix  (rows=true, cols=predicted):")
    hdr = "".join(f"{c:>12}" for c in classes)
    print(f"    {'':>16} {hdr}")
    for i, row in enumerate(cm):
        print(f"    {classes[i]:>16} {''.join(f'{v:>12}' for v in row)}")
    print("═"*60)

    return results


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    p = argparse.ArgumentParser(description="Evaluate a trained image classifier.")
    p.add_argument('--model', default='models/exported/classifier.pkl')
    p.add_argument('--data',  default='data/dataset')
    p.add_argument('--output', default='eval_report.json')
    args = p.parse_args()

    results = evaluate(args.model, args.data)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Full report → '{args.output}'")


if __name__ == "__main__":
    main()
