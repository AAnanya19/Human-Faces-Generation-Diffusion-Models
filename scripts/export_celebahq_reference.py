#!/usr/bin/env python3
"""
Export the CelebA-HQ test split written by training as PNG files.

The training script (``src/training/train.py``) writes ``test_files.txt`` into the
run directory when ``folder_test_size`` is positive. Each line is an absolute
path to an image that was held out from training.

This script reads that file, resizes each image to a fixed square size, and
writes the result into ``--out-dir`` so it can be used as the **real** folder
for FID evaluation against the model-generated **fake** folder.

Example:
    python scripts/export_celebahq_reference.py \\
        --test-files /content/drive/MyDrive/aml/ddpm_runs/celebahq_run_001/test_files.txt \\
        --out-dir /content/drive/MyDrive/aml/ddpm_runs/celebahq_run_001/real_test_set \\
        --image-size 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize and export CelebA-HQ test images for FID evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-files",
        type=Path,
        required=True,
        help="Path to test_files.txt produced by the training run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where resized real images will be written (created if missing).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize side; should match the training/sampling --image_size for fair FID.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="real",
        help="Filename prefix; outputs are <prefix>_00000.png, <prefix>_00001.png, ...",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to export (default: all).",
    )
    return parser.parse_args()


def read_test_paths(test_files: Path) -> list[Path]:
    """Return absolute Paths from test_files.txt, skipping blank lines."""
    if not test_files.is_file():
        raise SystemExit(f"test_files.txt not found: {test_files}")
    paths: list[Path] = []
    for raw in test_files.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        paths.append(Path(line))
    if not paths:
        raise SystemExit(f"No paths listed in {test_files}")
    return paths


def main() -> None:
    args = parse_args()

    paths = read_test_paths(args.test_files)
    if args.limit is not None:
        paths = paths[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(paths)} reference images to {args.out_dir.resolve()}")

    missing: list[Path] = []
    written = 0
    for idx, src in enumerate(paths):
        if not src.is_file():
            missing.append(src)
            continue
        with Image.open(src) as im:
            rgb = im.convert("RGB").resize(
                (args.image_size, args.image_size), Image.BICUBIC
            )
        rgb.save(args.out_dir / f"{args.prefix}_{idx:05d}.png")
        written += 1

    print(f"Wrote {written}/{len(paths)} images.")
    if missing:
        print(
            f"WARNING: {len(missing)} files were missing on disk and skipped. "
            "First few:",
            file=sys.stderr,
        )
        for p in missing[:5]:
            print(f"  - {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
