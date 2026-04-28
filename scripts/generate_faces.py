"""
Load a trained U-Net checkpoint and write individual generated face images.

Example (from project root):
    python3.10 scripts/generate_faces.py --checkpoint checkpoints/ddpm_faces_final.pth

    # Generate 300 images (required for FID evaluation):
    python scripts/generate_faces.py \
        --checkpoint checkpoints/ddpm_faces_final.pth \
        --num_images 300 \
        --batch_size 16 \
        --image_size 256

    # Quick smoke-test on CPU (small config):
    python scripts/generate_faces.py \
        --checkpoint checkpoints/ddpm_faces_final.pth \
        --num_images 4 \
        --image_size 64 \
        --timesteps 200 \
        --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so src.* imports work whether
# this script is called from the root or from scripts/.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.diffusion.sample import sample        # noqa: E402
from src.diffusion.scheduler import DDPMScheduler  # noqa: E402
from src.models.unet import UNet              # noqa: E402


# ---------------------------------------------------------------------------
# Suggested U-Net config for CelebA-HQ 256x256
# ---------------------------------------------------------------------------
# base_channels=128  → doubles feature maps vs butterfly (64×64) model;
#                      handles the 16× larger spatial resolution well.
# time_dim=512       → richer sinusoidal time embedding; helps the model
#                      distinguish fine-grained noise levels across T=1000.
# At least 4 down-sampling levels are expected inside UNet so that the
# bottleneck sees an 8×8 or 16×16 feature map — large enough receptive field
# to model global face structure (symmetry, pose).
# Attention at 16×16 and 8×8 resolutions is strongly recommended inside UNet
# for facial coherence (eyes, alignment).  These are controlled inside
# src/models/unet.py; default args here match that expectation.
# ---------------------------------------------------------------------------
DEFAULT_BASE_CHANNELS = 128
DEFAULT_TIME_DIM = 512
DEFAULT_IMAGE_SIZE = 256
DEFAULT_TIMESTEPS = 1000
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_IMAGES = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: str | None) -> str:
    """Return a concrete device string, preferring CUDA then MPS then CPU."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(base_channels: int, time_dim: int, device: str) -> UNet:
    """Instantiate U-Net with the recommended face-generation config."""
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        time_dim=time_dim,
    ).to(device)
    return model


def load_checkpoint(model: UNet, ckpt_path: Path, device: str) -> None:
    """Load a state-dict-only checkpoint saved by train.py."""
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()


def generate_in_batches(
    model: UNet,
    scheduler: DDPMScheduler,
    num_images: int,
    batch_size: int,
    image_size: int,
    device: str,
) -> list[torch.Tensor]:
    """
    Run the reverse diffusion process in batches and collect results.

    Returns a list of image tensors (values in [0, 1]) each of shape
    (B, 3, image_size, image_size).
    """
    batches: list[torch.Tensor] = []
    remaining = num_images

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        imgs = sample(
            model,
            scheduler,
            image_size=image_size,
            batch_size=current_batch,
            channels=3,
            device=device,
        )
        # Rescale from [-1, 1] → [0, 1]
        imgs = (imgs.clamp(-1, 1) + 1) * 0.5
        batches.append(imgs.cpu())
        remaining -= current_batch
        generated_so_far = num_images - remaining
        print(f"  Generated {generated_so_far}/{num_images} images …")

    return batches


def save_individual_images(
    batches: list[torch.Tensor],
    out_dir: Path,
    prefix: str = "face",
) -> None:
    """Save each image as an individual PNG file for FID evaluation."""
    idx = 0
    for batch in batches:
        for img in batch:
            fname = out_dir / f"{prefix}_{idx:05d}.png"
            vutils.save_image(img, fname)
            idx += 1
    print(f"Saved {idx} individual images to {out_dir.resolve()}")


def save_preview_grid(
    batches: list[torch.Tensor],
    out_dir: Path,
    nrow: int = 8,
    max_preview: int = 64,
    filename: str = "faces_grid.png",
) -> None:
    """
    Save a grid of up to `max_preview` images for quick visual inspection.
    Useful for including in your report.
    """
    all_imgs = torch.cat(batches, dim=0)[:max_preview]
    grid_path = out_dir / filename
    vutils.save_image(all_imgs, grid_path, nrow=nrow)
    print(f"Saved preview grid ({len(all_imgs)} images) to {grid_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate human face images from a trained DDPM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ddpm_faces_final.pth",
        help="Path to .pth checkpoint (state_dict only) saved by train.py. "
             "Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/generated_faces",
        help="Directory where individual PNGs and the preview grid are saved.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Total number of images to generate. Use 300 for FID evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Images generated per forward pass. Reduce if you hit OOM.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Spatial resolution (height == width). Must match training config.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Number of DDPM reverse diffusion steps. Must match training config.",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=DEFAULT_BASE_CHANNELS,
        help="U-Net base channel width. Must match training config.",
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=DEFAULT_TIME_DIM,
        help="Sinusoidal time-embedding dimension. Must match training config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a specific device (cuda / mps / cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no_grid",
        action="store_true",
        help="Skip saving the preview grid (saves time when only FID files are needed).",
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=8,
        help="Number of columns in the preview grid image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ device
    device = resolve_device(args.device)
    print(f"torch : {torch.__version__}")
    print(f"device: {device}")
    if device == "cpu":
        print(
            "\nWarning: CPU inference with 1000 DDPM steps at 256×256 is extremely slow.\n"
            "         Use --device cuda (Colab GPU) or --device mps (Apple Silicon).\n"
            "         For a quick CPU smoke-test use --timesteps 200 --image_size 64.\n"
        )

    # --------------------------------------------------------------- paths
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    if not ckpt.is_file():
        raise SystemExit(
            f"\nCheckpoint not found: {ckpt}\n"
            f"Project root resolved to: {_ROOT}\n"
            f"Tip: pass --checkpoint relative/to/project/root/ddpm_faces_final.pth\n"
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- model
    print(
        f"\nLoading U-Net  (base_channels={args.base_channels}, "
        f"time_dim={args.time_dim}) …"
    )
    model = build_model(args.base_channels, args.time_dim, device)
    load_checkpoint(model, ckpt, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded   ({n_params:.1f}M parameters)")

    # ----------------------------------------------------------- scheduler
    scheduler = DDPMScheduler(timesteps=args.timesteps, device=device)

    # ----------------------------------------------------------- generation
    num_batches = math.ceil(args.num_images / args.batch_size)
    print(
        f"\nGenerating {args.num_images} images "
        f"in {num_batches} batches of up to {args.batch_size} …\n"
        f"image_size={args.image_size}, timesteps={args.timesteps}\n"
    )

    with torch.inference_mode():
        batches = generate_in_batches(
            model=model,
            scheduler=scheduler,
            num_images=args.num_images,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=device,
        )

    # --------------------------------------------------------------- saving
    print("\nSaving outputs …")
    save_individual_images(batches, out_dir)

    if not args.no_grid:
        save_preview_grid(batches, out_dir, nrow=args.grid_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()