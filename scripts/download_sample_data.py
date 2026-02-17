#!/usr/bin/env python3
"""
scripts/download_sample_data.py
────────────────────────────────
Downloads a 3-class sample dataset (cats / dogs / birds) so you can run
the full pipeline out of the box with zero manual data prep.

Uses the Oxford-IIIT Pet Dataset subset via fast.ai's built-in downloader
(no API keys required, ~800 MB total).

Usage:
    python scripts/download_sample_data.py
    python scripts/download_sample_data.py --classes cat dog bird --limit 200
"""

import argparse
import shutil
import sys
from pathlib import Path


def download_fastai_pets(dest: Path) -> None:
    """
    Downloads the fast.ai 'pets' dataset and reorganises it into a flat
    data/dataset/<classname>/ structure expected by train.py.

    The pets dataset contains 37 breeds of cats and dogs.
    We collapse them into two super-classes (cat, dog).
    """
    try:
        from fastai.vision.all import untar_data, URLs
    except ImportError:
        sys.exit("fastai is not installed. Run:  pip install fastai")

    print("[DOWNLOAD] Fetching Oxford-IIIT Pet Dataset via fast.ai …")
    path = untar_data(URLs.PETS)          # downloads & caches automatically
    images_path = path / "images"

    # fast.ai pet images follow the naming convention:
    #   Abyssinian_001.jpg  (capital first letter → cat breed)
    #   american_bulldog_001.jpg  (lowercase first letter → dog breed)
    DEST_CATS  = dest / "cat"
    DEST_DOGS  = dest / "dog"
    DEST_CATS.mkdir(parents=True, exist_ok=True)
    DEST_DOGS.mkdir(parents=True, exist_ok=True)

    cat_count = dog_count = 0
    for img in sorted(images_path.glob("*.jpg")):
        stem = img.stem          # e.g.  "Abyssinian_001"
        first_char = stem[0]
        if first_char.isupper():          # Cat breeds start with capital letter
            shutil.copy2(img, DEST_CATS / img.name)
            cat_count += 1
        else:                             # Dog breeds start with lowercase
            shutil.copy2(img, DEST_DOGS / img.name)
            dog_count += 1

    print(f"  ✓ cats : {cat_count} images → {DEST_CATS}")
    print(f"  ✓ dogs : {dog_count} images → {DEST_DOGS}")


def download_bing(classes: list, limit: int, dest: Path) -> None:
    """
    Downloads images for arbitrary class names using bing-image-downloader.
    Requires:  pip install bing-image-downloader
    """
    try:
        from bing_image_downloader import downloader
    except ImportError:
        sys.exit(
            "bing-image-downloader not installed.\n"
            "  Run:  pip install bing-image-downloader"
        )

    print(f"[DOWNLOAD] Downloading {limit} images each for: {classes}")
    for cls in classes:
        downloader.download(
            query=cls,
            limit=limit,
            output_dir=str(dest),
            adult_filter_off=True,
            force_replace=False,
            timeout=10,
        )
        # bing-image-downloader creates a subfolder named after the query
        src = dest / cls
        if src.exists():
            print(f"  ✓ {cls}: {len(list(src.glob('*.jpg')))} images → {src}")
        else:
            print(f"  ✗ {cls}: no images downloaded")


def main():
    parser = argparse.ArgumentParser(
        description="Download sample image data for the classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--classes', nargs='+', default=None,
        help='Class names to download via Bing (e.g. --classes cat dog bird). '
             'Omit to use the built-in fast.ai pets dataset.',
    )
    parser.add_argument(
        '--limit', type=int, default=150,
        help='Max images per class when using --classes (default: 150)',
    )
    parser.add_argument(
        '--dest', type=str, default='data/dataset',
        help='Destination folder (default: data/dataset)',
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.classes:
        download_bing(args.classes, args.limit, dest)
    else:
        download_fastai_pets(dest)

    # Summary
    print("\n[DONE] Dataset summary:")
    total = 0
    for cls_dir in sorted(dest.iterdir()):
        if cls_dir.is_dir():
            n = len(list(cls_dir.glob("*.*")))
            total += n
            print(f"  {cls_dir.name:>16}: {n:>5} images")
    print(f"  {'TOTAL':>16}: {total:>5} images")
    print(f"\n  Ready to train:  python train.py\n")


if __name__ == "__main__":
    main()
