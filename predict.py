#!/usr/bin/env python3
"""
predict.py — Inference entry point.

Usage:
    python predict.py --image  cat.jpg
    python predict.py --images cat.jpg dog.jpg bird.jpg
    python predict.py --folder data/test/
    python predict.py --image  cat.jpg --output-format json
    python predict.py --image  cat.jpg --cpu
    python predict.py --image  cat.jpg --threshold 0.7
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from fastai.vision.all import load_learner, PILImage


# ─────────────────────────────────────────────────────────────────────────────
class ImageClassifier:
    """
    Production wrapper around a fast.ai exported .pkl model.
    Load once, call predict() many times — efficient for batch use.
    """

    def __init__(self, model_path: str, cpu_only: bool = False):
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model '{model_path}' not found.\n"
                "  → Train first:  python train.py"
            )
        logging.info(f"[CLASSIFIER] Loading '{model_path}' …")
        self.learn = load_learner(p, cpu=cpu_only or not torch.cuda.is_available())
        self.classes = list(self.learn.dls.vocab)
        logging.info(f"[CLASSIFIER] Ready — {len(self.classes)} classes: {self.classes}")

    # ── single image ─────────────────────────────────────────────────────────
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Classifies one image.

        Returns:
            {
              "image_path": str,
              "label":      str,          # top-1 predicted class
              "confidence": float,        # probability of top-1 (0–1)
              "all_probs":  {cls: prob},  # all classes, sorted desc
              "top_3":      [(cls, prob), ...]
            }
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: '{path}'")

        label, idx, probs = self.learn.predict(PILImage.create(path))

        probs_dict = dict(
            sorted(
                {cls: round(float(p), 6) for cls, p in zip(self.classes, probs)}.items(),
                key=lambda x: x[1], reverse=True,
            )
        )
        return {
            "image_path": str(path),
            "label":      str(label),
            "confidence": round(float(probs[idx]), 6),
            "all_probs":  probs_dict,
            "top_3":      list(probs_dict.items())[:3],
        }

    # ── batch ─────────────────────────────────────────────────────────────────
    def predict_batch(self, paths: List[Union[str, Path]], verbose: bool = True) -> List[Dict]:
        results = []
        for i, p in enumerate(paths):
            if verbose and i % 10 == 0:
                logging.info(f"[BATCH] {i+1}/{len(paths)}")
            try:
                results.append(self.predict(p))
            except Exception as e:
                logging.error(f"[BATCH] Failed on '{p}': {e}")
                results.append({"image_path": str(p), "error": str(e),
                                 "label": None, "confidence": None, "all_probs": None})
        return results

    # ── folder ────────────────────────────────────────────────────────────────
    def predict_folder(self, folder: Union[str, Path]) -> List[Dict]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: '{folder}'")
        files = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        )
        if not files:
            raise ValueError(f"No images found in '{folder}'")
        logging.info(f"[FOLDER] {len(files)} images in '{folder}'")
        return self.predict_batch(files)

    # ── pretty print ─────────────────────────────────────────────────────────
    def print_result(self, r: Dict) -> None:
        if "error" in r:
            print(f"\n❌  {r['image_path']}  →  ERROR: {r['error']}")
            return
        print(f"\n{'─'*52}")
        print(f"  Image  : {r['image_path']}")
        print(f"  Result : {r['label'].upper()}  ({r['confidence']*100:.2f}%)")
        print()
        for cls, prob in r['all_probs'].items():
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            tag = "  ◄" if cls == r['label'] else ""
            print(f"  {cls:>14}: {bar} {prob*100:5.1f}%{tag}")
        print(f"{'─'*52}")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a trained image classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--image',  type=str,           help='Single image path')
    grp.add_argument('--images', type=str, nargs='+', help='Multiple image paths')
    grp.add_argument('--folder', type=str,           help='Classify all images in folder')

    p.add_argument('--model', default='models/exported/classifier.pkl',
                   help='Path to exported .pkl model')
    p.add_argument('--output-format', choices=['pretty', 'json'], default='pretty')
    p.add_argument('--output-file', type=str, default=None,
                   help='Save JSON results to this file')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU inference')
    p.add_argument('--threshold', type=float, default=None,
                   help='Confidence threshold — labels below this marked UNCERTAIN')
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')
    args = parse_args()

    try:
        clf = ImageClassifier(args.model, cpu_only=args.cpu)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    if args.image:
        results = [clf.predict(args.image)]
    elif args.images:
        results = clf.predict_batch(args.images)
    else:
        results = clf.predict_folder(args.folder)

    # Apply confidence threshold
    if args.threshold:
        for r in results:
            if r.get('confidence') and r['confidence'] < args.threshold:
                r['label'] = f"UNCERTAIN (best: {r['label']})"

    if args.output_format == 'json':
        out = json.dumps(results, indent=2)
        if args.output_file:
            Path(args.output_file).write_text(out)
            logging.info(f"Results saved to '{args.output_file}'")
        else:
            print(out)
    else:
        for r in results:
            clf.print_result(r)
        if len(results) > 1:
            ok = sum(1 for r in results if 'error' not in r)
            print(f"\n  Total: {len(results)}  ✓ {ok}  ✗ {len(results)-ok}")

    sys.exit(1 if any('error' in r for r in results) else 0)


if __name__ == "__main__":
    main()
