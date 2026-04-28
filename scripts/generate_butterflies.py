"""
Load a trained U-Net checkpoint and write generated butterfly images.

By default produces a single preview grid. Pass ``--num_images > batch_size`` to
also save individual PNGs (needed for FID evaluation against a reference set).

Examples (from project root):
    # Quick preview grid only:
    python scripts/generate_butterflies.py --checkpoint checkpoints/ddpm_final.pth

    # Generate 100 individual PNGs for FID:
    python scripts/generate_butterflies.py \\
        --checkpoint checkpoints/ddpm_final.pth \\
        --num_images 100 \\
        --batch_size 16
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.diffusion.sample import sample  # noqa: E402
from src.diffusion.scheduler import DDPMScheduler  # noqa: E402
from src.models.unet import UNet  # noqa: E402


def resolve_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ddpm_final.pth",
        help="Path to .pth file saved by train.py (state_dict only).",
    )
    parser.add_argument("--out_dir", type=str, default="results/generated")
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Total number of images to generate. If unset, generates one grid of "
             "batch_size images. Use the butterfly test split size for FID parity.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--base_channels",
        type=int,
        default=64,
        help="U-Net base channel width; must match training.",
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=256,
        help="U-Net time-embedding dim; must match training.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no_grid",
        action="store_true",
        help="Skip writing the preview grid (only useful with --num_images).",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    if not ckpt.is_file():
        raise SystemExit(
            f"Checkpoint not found: {ckpt}\n"
            f"(Paths relative to project root: {_ROOT})\n"
            f"Example: --checkpoint checkpoints_smoke/ddpm_final.pth"
        )

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    scheduler = DDPMScheduler(timesteps=args.timesteps, device=device)

    print(
        f"Generating on device={device} with batch_size={args.batch_size}, "
        f"image_size={args.image_size}, timesteps={args.timesteps}"
    )
    if device == "cpu":
        print(
            "Warning: CPU sampling with 1000 DDPM timesteps can be very slow. "
            "Use --device mps or --device cuda when available, or run generation in Colab on a GPU."
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single preview grid (legacy behaviour) when --num_images is omitted.
    if args.num_images is None:
        images = sample(
            model,
            scheduler,
            image_size=args.image_size,
            batch_size=args.batch_size,
            channels=3,
            device=device,
        )
        images = (images.clamp(-1, 1) + 1) * 0.5
        out_path = out_dir / "butterfly_grid.png"
        vutils.save_image(images, out_path, nrow=min(4, args.batch_size))
        print(f"Saved {args.batch_size} samples to {out_path.resolve()}")
        return

    # Generate ``--num_images`` total in batches and save individual PNGs.
    num_images = int(args.num_images)
    batch_size = max(1, int(args.batch_size))
    num_batches = math.ceil(num_images / batch_size)
    remaining = num_images
    saved = 0
    all_for_grid = []
    with torch.inference_mode():
        for batch_idx in range(num_batches):
            current = min(batch_size, remaining)
            imgs = sample(
                model,
                scheduler,
                image_size=args.image_size,
                batch_size=current,
                channels=3,
                device=device,
            )
            imgs = (imgs.clamp(-1, 1) + 1) * 0.5
            for j in range(current):
                vutils.save_image(imgs[j], out_dir / f"butterfly_{saved:05d}.png")
                saved += 1
            all_for_grid.append(imgs.cpu())
            remaining -= current
            print(f"  Generated {num_images - remaining}/{num_images} images …")

    print(f"Saved {saved} individual images to {out_dir.resolve()}")

    if not args.no_grid and all_for_grid:
        preview = torch.cat(all_for_grid, dim=0)[:64]
        grid_path = out_dir / "butterfly_grid.png"
        vutils.save_image(preview, grid_path, nrow=8)
        print(f"Saved preview grid ({preview.shape[0]} images) to {grid_path.resolve()}")


if __name__ == "__main__":
    main()
