#!/usr/bin/env python3
"""
Compute FID between a real image folder and a fake image folder, then write a
JSON summary suitable for the report and the project-wide aggregator.

This is a thin wrapper around ``src.evaluation.fid.compute_fid_from_directories``
that adds:

- a richer JSON output (image counts, image size, run name, paths)
- optional grids saved next to the JSON
- compatibility with the path passed in ``notebooks/colab_faces_generate.ipynb``
  (its closing markdown cell tells us to run ``scripts/evaluate_fid.py``)

Example:
    python scripts/evaluate_fid.py \\
        --real-dir /content/drive/MyDrive/aml/ddpm_runs/celebahq_run_001/real_test_set \\
        --fake-dir /content/drive/MyDrive/aml/ddpm_runs/celebahq_run_001/generated_faces \\
        --image-size 256 \\
        --run-name celebahq_run_001 \\
        --out-json /content/drive/MyDrive/aml/ddpm_runs/celebahq_run_001/fid.json \\
        --grid-out  /content/drive/MyDrive/aml/ddpm_runs/celebahq_run_001/eval/generated_grid.png \\
        --real-grid-out /content/drive/MyDrive/aml/ddpm_runs/celebahq_run_001/eval/real_grid.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.fid import compute_fid_from_directories, list_image_paths
from src.evaluation.visualize import grid_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FID between a real and fake image folder and dump JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-dir", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize both sets to (S, S). Use the training/sampling image size.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cuda/mps/cpu); auto-detect if omitted.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label written into the JSON (e.g. celebahq_run_001).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Where to write the FID JSON summary.",
    )
    parser.add_argument(
        "--grid-out",
        type=Path,
        default=None,
        help="Optional output PNG: 8x8 montage of fake images.",
    )
    parser.add_argument(
        "--real-grid-out",
        type=Path,
        default=None,
        help="Optional output PNG: 8x8 montage of real images.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=8,
        help="Images per row in any grid output.",
    )
    parser.add_argument(
        "--grid-count",
        type=int,
        default=64,
        help="Number of images included in each grid (truncated from each folder).",
    )
    return parser.parse_args()


def maybe_save_grid(folder: Path, out: Path | None, *, count: int, nrow: int) -> None:
    if out is None:
        return
    paths = list_image_paths(folder)[:count]
    grid_from_paths(paths, out, nrow=nrow)
    print(f"Saved grid to {out}")


def main() -> None:
    args = parse_args()

    real_paths = list_image_paths(args.real_dir)
    fake_paths = list_image_paths(args.fake_dir)
    print(
        f"FID inputs: real={len(real_paths)} (from {args.real_dir}) "
        f"fake={len(fake_paths)} (from {args.fake_dir}) "
        f"image_size={args.image_size}"
    )

    score = compute_fid_from_directories(
        args.real_dir,
        args.fake_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    fid_val = float(score.item() if score.numel() == 1 else score)
    print(f"FID: {fid_val:.4f}")

    payload = {
        "fid": fid_val,
        "run_name": args.run_name,
        "image_size": args.image_size,
        "n_real": len(real_paths),
        "n_fake": len(fake_paths),
        "real_dir": str(args.real_dir.resolve()),
        "fake_dir": str(args.fake_dir.resolve()),
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote FID summary to {args.out_json}")

    maybe_save_grid(args.fake_dir, args.grid_out, count=args.grid_count, nrow=args.grid_rows)
    maybe_save_grid(args.real_dir, args.real_grid_out, count=args.grid_count, nrow=args.grid_rows)


if __name__ == "__main__":
    main()
