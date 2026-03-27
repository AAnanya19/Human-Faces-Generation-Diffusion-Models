#!/usr/bin/env python3
"""
Export butterfly images from the Hugging Face dataset to PNG files on disk.

Uses the same train/val/test splits as ``data/dataloader.py`` (same ``seed``),
so the exported set matches what your DataLoader calls the test (or val) split.

Typical use for FID:
  - Export the **test** split to your "Real" folder (reference distribution).
  - After sampling, put generated images in your "Fake" folder.
  - Run: python -m src.evaluation.run_eval --real-dir ... --fake-dir ...

Example:
  python scripts/export_butterfly_reference.py \\
    --out-dir "/path/to/Real" \\
    --split test \\
    --image-size 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from torchvision import transforms

from data.dataloader import splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export butterfly split images to PNG (for FID 'real' folder)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write PNG files (created if missing).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Which split to export (default: test, for FID reference).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Must match data/dataloader.py if you want identical splits.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Resize to (S, S) before save; match training/FID --image-size.",
    )
    args = parser.parse_args()

    train_ds, val_ds, test_ds = splits(seed=args.seed)
    name_to_ds = {"train": train_ds, "val": val_ds, "test": test_ds}
    hf_ds = name_to_ds[args.split]

    resize = transforms.Compose(
        [transforms.Resize((args.image_size, args.image_size))]
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(hf_ds)
    for i in range(n):
        img = hf_ds[i]["image"].convert("RGB")
        img = resize(img)
        img.save(out_dir / f"butterfly_{args.split}_{i:04d}.png")

    print(f"Wrote {n} images to {out_dir}")


if __name__ == "__main__":
    main()
