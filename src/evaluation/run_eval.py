"""
CLI entrypoint for FID between two folders of images.

Run from repository root::

    python -m src.evaluation.run_eval --real-dir path/to/real --fake-dir path/to/fake

Optional JSON output for reports or experiment tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.fid import compute_fid_from_directories, list_image_paths
from src.evaluation.visualize import grid_from_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FID between reference and generated image folders."
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="Directory containing reference images (e.g. CelebA-HQ test crop exports).",
    )
    parser.add_argument(
        "--fake-dir",
        type=Path,
        required=True,
        help="Directory containing generated images to score.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize side for both sets (default: 256 for CelebA-HQ).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 0 on Colab/Windows if needed).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write {\"fid\": float, ...}.",
    )
    parser.add_argument(
        "--grid-from",
        type=Path,
        default=None,
        help="Optional folder of images; saves a montage PNG next to out-json or cwd.",
    )
    parser.add_argument(
        "--grid-out",
        type=Path,
        default=None,
        help="Output path for grid image when --grid-from is set.",
    )
    args = parser.parse_args()

    score = compute_fid_from_directories(
        args.real_dir,
        args.fake_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    fid_val = float(score.item() if score.numel() == 1 else score)
    print(f"FID: {fid_val:.4f}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fid": fid_val}
        args.out_json.write_text(json.dumps(payload, indent=2))

    if args.grid_from is not None:
        paths = list_image_paths(args.grid_from)
        out = args.grid_out
        if out is None:
            out = Path("results/butterflies/eval/sample_grid.png")
        grid_from_paths(paths, out)
        print(f"Saved grid to {out}")


if __name__ == "__main__":
    main()
