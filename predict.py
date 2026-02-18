#!/usr/bin/env python3
"""predict.py — Inference entry point."""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.torch_pipeline.inference import TorchImageClassifier as ImageClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a trained image classifier.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--image', type=str, help='Single image path')
    grp.add_argument('--images', type=str, nargs='+', help='Multiple image paths')
    grp.add_argument('--folder', type=str, help='Classify all images in folder')

    p.add_argument('--model', default='models/exported/model_state.pt', help='Path to model_state.pt or artifact directory')
    p.add_argument('--output-format', choices=['pretty', 'json'], default='pretty')
    p.add_argument('--output-file', type=str, default=None, help='Save JSON results to this file')
    p.add_argument('--cpu', action='store_true', help='Force CPU inference')
    p.add_argument('--threshold', type=float, default=None, help='Confidence threshold')
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
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

    if args.threshold is not None:
        for r in results:
            if r.get('confidence') is not None and r['confidence'] < args.threshold:
                r['label'] = f"UNCERTAIN (best: {r['label']})"

    if args.output_format == 'json':
        out = json.dumps(results, indent=2)
        if args.output_file:
            Path(args.output_file).write_text(out)
            logging.info("Results saved to '%s'", args.output_file)
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
