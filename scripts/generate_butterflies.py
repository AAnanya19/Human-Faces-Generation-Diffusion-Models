"""
Load a trained U-Net checkpoint and write a grid of generated butterfly images.

Example (from project root):
    python scripts/generate_butterflies.py --checkpoint checkpoints/ddpm_final.pth
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ddpm_final.pth",
        help="Path to .pth file saved by train.py (state_dict only).",
    )
    parser.add_argument("--out_dir", type=str, default="results/generated")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
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

    model = UNet(in_channels=3, out_channels=3, base_channels=64, time_dim=256).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

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

    images = sample(
        model,
        scheduler,
        image_size=args.image_size,
        batch_size=args.batch_size,
        channels=3,
        device=device,
    )
    # [-1, 1] -> [0, 1] for saving
    images = (images.clamp(-1, 1) + 1) * 0.5

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "butterfly_grid.png"
    vutils.save_image(images, out_path, nrow=min(4, args.batch_size))

    print(f"Saved {args.batch_size} samples to {out_path.resolve()}")


if __name__ == "__main__":
    main()
