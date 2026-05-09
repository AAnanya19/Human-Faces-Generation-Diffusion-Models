"""
Generate CelebA-HQ face samples from a trained DDPM checkpoint.

The script can read the model/diffusion parameters saved inside checkpoints
from src/training/train.py. CLI values override checkpoint values when passed.
"""

from __future__ import annotations

import argparse
import json
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


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_int_list(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated integers, got: {raw!r}")
    return values


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    return resolved


def safe_torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint_payload(ckpt_path: Path, device: str) -> tuple[dict, dict]:
    payload = safe_torch_load(ckpt_path, device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload.get("eval_model_state_dict", payload["model_state_dict"])
        return state_dict, payload
    return payload, {}


def config_from_checkpoint(checkpoint: dict) -> tuple[dict, dict]:
    model_config = dict(checkpoint.get("model_config") or {})
    diffusion_config = dict(checkpoint.get("diffusion_config") or {})
    run_config = checkpoint.get("run_config") or {}
    model_config.update(run_config.get("model", {}))
    diffusion_config.update(run_config.get("diffusion", {}))
    return model_config, diffusion_config


def make_seeded_noise(
    shape: tuple[int, ...],
    *,
    device: str,
    seed: int,
) -> torch.Tensor:
    device_type = torch.device(device).type
    generator_device = "cpu" if device_type == "mps" else device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=generator_device)
    return noise.to(device)


def build_model_from_config(
    *,
    model_config: dict,
    image_size: int,
    base_channels: int,
    time_dim: int,
    channel_mults: tuple[int, ...],
    num_res_blocks: int,
    dropout: float,
    attention_resolutions: tuple[int, ...],
    device: str,
) -> UNet:
    return UNet(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 3)),
        base_channels=base_channels,
        time_dim=time_dim,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        image_size=image_size,
    ).to(device)


@torch.no_grad()
def generate_in_batches(
    model: UNet,
    scheduler: DDPMScheduler,
    *,
    num_images: int,
    batch_size: int,
    image_size: int,
    device: str,
    seed: int | None,
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    remaining = num_images
    batch_index = 0
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        initial_noise = None
        if seed is not None:
            initial_noise = make_seeded_noise(
                (current_batch, 3, image_size, image_size),
                device=device,
                seed=seed + batch_index,
            )
        images = sample(
            model,
            scheduler,
            image_size=image_size,
            batch_size=current_batch,
            channels=3,
            device=device,
            initial_noise=initial_noise,
        )
        images = (images.clamp(-1, 1) + 1) * 0.5
        batches.append(images.cpu())
        remaining -= current_batch
        print(f"  Generated {num_images - remaining}/{num_images} images")
        batch_index += 1
    return batches


def save_individual_images(
    batches: list[torch.Tensor],
    out_dir: Path,
    *,
    prefix: str,
) -> None:
    image_index = 0
    for batch in batches:
        for image in batch:
            vutils.save_image(image, out_dir / f"{prefix}_{image_index:05d}.png")
            image_index += 1
    print(f"Saved {image_index} individual images to {out_dir.resolve()}")


def save_preview_grid(
    batches: list[torch.Tensor],
    out_dir: Path,
    *,
    filename: str,
    nrow: int,
    max_preview: int,
) -> None:
    all_images = torch.cat(batches, dim=0)[:max_preview]
    grid_path = out_dir / filename
    vutils.save_image(all_images, grid_path, nrow=nrow)
    print(f"Saved preview grid to {grid_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate face images from a trained DDPM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="runs/ddpm_runs/celebahq_run_001/best_model.pth")
    parser.add_argument("--out_dir", type=str, default="results/generated_faces")
    parser.add_argument("--num_images", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--noise_schedule", choices=["linear", "cosine"], default=None)
    parser.add_argument("--noise_max_beta", type=float, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--time_dim", type=int, default=None)
    parser.add_argument("--channel_mults", type=str, default=None)
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention_resolutions", type=str, default=None)

    parser.add_argument("--prefix", type=str, default="face")
    parser.add_argument("--no_individual", action="store_true")
    parser.add_argument("--no_grid", action="store_true")
    parser.add_argument("--grid_rows", type=int, default=8)
    parser.add_argument("--max_preview", type=int, default=64)
    parser.add_argument("--grid_name", type=str, default="faces_grid.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    state_dict, checkpoint = load_checkpoint_payload(checkpoint_path, device)
    model_config, diffusion_config = config_from_checkpoint(checkpoint)

    image_size = args.image_size or int(model_config.get("image_size", 64))
    timesteps = args.timesteps or int(diffusion_config.get("timesteps", 1000))
    noise_schedule = args.noise_schedule or diffusion_config.get("noise_schedule", "linear")
    noise_max_beta = (
        args.noise_max_beta
        if args.noise_max_beta is not None
        else float(diffusion_config.get("noise_max_beta", 0.999))
    )
    base_channels = args.base_channels or int(model_config.get("base_channels", 64))
    time_dim = args.time_dim or int(model_config.get("time_dim", 256))
    channel_mults = parse_int_list(
        args.channel_mults,
        tuple(model_config.get("channel_mults", [1, 2, 4, 8])),
    )
    num_res_blocks = args.num_res_blocks or int(model_config.get("num_res_blocks", 2))
    dropout = args.dropout if args.dropout is not None else float(model_config.get("dropout", 0.1))
    attention_resolutions = parse_int_list(
        args.attention_resolutions,
        tuple(model_config.get("attention_resolutions", [16, 8])),
    )

    print("Project root:", _ROOT)
    print("Checkpoint:", checkpoint_path)
    print("Output dir:", out_dir)
    print("Device:", device)
    print(
        "Generation params:",
        json.dumps(
            {
                "num_images": args.num_images,
                "batch_size": args.batch_size,
                "image_size": image_size,
                "timesteps": timesteps,
                "noise_schedule": noise_schedule,
                "noise_max_beta": noise_max_beta,
                "base_channels": base_channels,
                "time_dim": time_dim,
                "channel_mults": list(channel_mults),
                "num_res_blocks": num_res_blocks,
                "dropout": dropout,
                "attention_resolutions": list(attention_resolutions),
            },
            indent=2,
        ),
    )

    model = build_model_from_config(
        model_config=model_config,
        image_size=image_size,
        base_channels=base_channels,
        time_dim=time_dim,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        device=device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    scheduler = DDPMScheduler(
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        noise_max_beta=noise_max_beta,
        device=device,
    )

    num_batches = math.ceil(args.num_images / args.batch_size)
    print(f"Generating {args.num_images} images in {num_batches} batches.")
    batches = generate_in_batches(
        model,
        scheduler,
        num_images=args.num_images,
        batch_size=args.batch_size,
        image_size=image_size,
        device=device,
        seed=args.seed,
    )

    if not args.no_individual:
        save_individual_images(batches, out_dir, prefix=args.prefix)
    if not args.no_grid:
        save_preview_grid(
            batches,
            out_dir,
            filename=args.grid_name,
            nrow=args.grid_rows,
            max_preview=args.max_preview,
        )
    print("Done.")


if __name__ == "__main__":
    main()
