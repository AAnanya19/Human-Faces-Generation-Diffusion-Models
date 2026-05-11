"""Compare DDPM and DDIM sampling with FID on  fixed test split.

Example:
    python3 scripts/compare_ddim_ddpm_fid.py \
      --checkpoint runs/ddpm_runs/celebahq_run_001/best_model.pth \
      --dataset_path data/celeba_hq_256 \
      --num_images 300 \
      --batch_size 16 \
      --fid_batch_size 16 \
      --ddim_steps 50

Outputs:
    results/ddim_ddpm_fid/
      comparison_summary.json
      fid_comparison.csv
      real_test_grid.png
      ddpm_grid.png
      ddim_grid.png
      ddpm/*.png
      ddim/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchvision.utils as vutils

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.butterfly_dataset import create_dataloaders  # noqa: E402
from src.diffusion.ddim import DDIMSampler  # noqa: E402
from src.diffusion.evaluation.metrics import (  # noqa: E402
    InceptionFeatureExtractor,
    calculate_fid_from_features,
    collect_features,
)
from src.diffusion.sample import sample as ddpm_sample  # noqa: E402
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


def sync_device(device: str) -> None:
    device_type = torch.device(device).type
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "mps" and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    return resolved


def parse_int_list(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated integers, got: {raw!r}")
    return values


def safe_torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def load_checkpoint_payload(ckpt_path: Path, device: str) -> tuple[dict, dict]:
    payload = safe_torch_load(ckpt_path, device)
    if isinstance(payload, dict):
        for key in ("eval_model_state_dict", "model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key], payload
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload, {}
    raise ValueError(
        "Checkpoint format not recognized. Expected a plain state_dict or a "
        "checkpoint containing eval_model_state_dict/model_state_dict."
    )


def config_from_checkpoint(checkpoint: dict) -> tuple[dict, dict, dict]:
    run_config = checkpoint.get("run_config") or {}
    model_config = dict(checkpoint.get("model_config") or {})
    diffusion_config = dict(checkpoint.get("diffusion_config") or {})
    model_config.update(run_config.get("model", {}))
    diffusion_config.update(run_config.get("diffusion", {}))
    dataset_config = dict(run_config.get("training", {}).get("dataset", {}))
    return model_config, diffusion_config, dataset_config


def make_seeded_noise(shape: tuple[int, ...], *, device: str, seed: int) -> torch.Tensor:
    device_type = torch.device(device).type
    generator_device = "cpu" if device_type == "mps" else device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=generator_device)
    return noise.to(device)


def build_model(
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
def collect_real_fid_images(test_loader, *, num_images: int) -> torch.Tensor:
    images = []
    collected = 0
    for batch in test_loader:
        needed = num_images - collected
        if needed <= 0:
            break
        batch = batch[:needed]
        images.append(((batch.clamp(-1, 1) + 1) * 0.5).cpu())
        collected += batch.shape[0]

    if collected < num_images:
        raise RuntimeError(
            f"FID needs {num_images} real test images, but only found {collected}. "
            "Increase --folder_test_size or lower --num_images."
        )
    return torch.cat(images, dim=0)


@torch.no_grad()
def generate_with_sampler(
    *,
    sampler_name: str,
    model: UNet,
    scheduler: DDPMScheduler,
    ddim_sampler: DDIMSampler | None,
    image_size: int,
    num_images: int,
    batch_size: int,
    device: str,
    seed: int,
    save_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generated_batches = []
    generated = 0
    batch_index = 0
    save_dir.mkdir(parents=True, exist_ok=True)

    sync_device(device)
    start = time.perf_counter()

    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)
        initial_noise = make_seeded_noise(
            (current_batch, 3, image_size, image_size),
            device=device,
            seed=seed + batch_index,
        )

        batch_start = time.perf_counter()
        if sampler_name == "ddpm":
            images = ddpm_sample(
                model,
                scheduler,
                image_size=image_size,
                batch_size=current_batch,
                channels=3,
                device=device,
                initial_noise=initial_noise,
            )
        elif sampler_name == "ddim":
            if ddim_sampler is None:
                raise ValueError("ddim_sampler is required when sampler_name='ddim'")
            images = ddim_sampler.sample(
                model,
                image_size=image_size,
                batch_size=current_batch,
                channels=3,
                device=device,
                initial_noise=initial_noise,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        sync_device(device)
        batch_seconds = time.perf_counter() - batch_start

        images = ((images.clamp(-1, 1) + 1) * 0.5).cpu()
        for offset, image in enumerate(images):
            image_index = generated + offset
            vutils.save_image(image, save_dir / f"{sampler_name}_{image_index:05d}.png")

        generated_batches.append(images)
        generated += current_batch
        batch_index += 1
        print(
            f"  {sampler_name.upper()} generation: {generated}/{num_images} "
            f"({batch_seconds:.2f}s for batch)"
        )

    sync_device(device)
    total_seconds = time.perf_counter() - start
    timing = {
        "generation_seconds": total_seconds,
        "seconds_per_image": total_seconds / max(num_images, 1),
        "images_per_second": num_images / total_seconds if total_seconds > 0 else None,
    }
    return torch.cat(generated_batches, dim=0), timing


def save_preview_grid(images: torch.Tensor, path: Path, *, nrow: int, max_images: int) -> None:
    preview = images[:max_images]
    vutils.save_image(preview, path, nrow=nrow)


def timed_collect_features(
    images: torch.Tensor,
    feature_extractor: InceptionFeatureExtractor,
    *,
    device: str,
    batch_size: int,
) -> tuple[Any, float]:
    sync_device(device)
    start = time.perf_counter()
    features = collect_features(
        images,
        feature_extractor,
        device=device,
        batch_size=batch_size,
    )
    sync_device(device)
    return features, time.perf_counter() - start


def write_results(out_dir: Path, summary: dict) -> None:
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    csv_path = out_dir / "fid_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sampler",
                "fid",
                "num_images",
                "sampling_steps",
                "eta",
                "generation_seconds",
                "seconds_per_image",
                "images_per_second",
                "feature_seconds",
                "total_eval_seconds",
                "generated_dir",
            ],
        )
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DDPM and DDIM samples and compare FID on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="runs/ddpm_runs/celebahq_run_001/best_model.pth")
    parser.add_argument("--out_dir", type=str, default="results/ddim_ddpm_fid")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fid_device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=999)

    parser.add_argument("--num_images", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fid_batch_size", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--no_clip_denoised", action="store_true")

    parser.add_argument("--dataset_source", choices=["hf", "folder"], default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--folder_subset_size", type=int, default=None)
    parser.add_argument("--folder_test_size", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

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

    parser.add_argument("--grid_rows", type=int, default=8)
    parser.add_argument("--max_preview", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    script_start = time.perf_counter()
    args = parse_args()
    device = resolve_device(args.device)
    fid_device = resolve_device(args.fid_device) if args.fid_device else device
    checkpoint_path = resolve_project_path(args.checkpoint)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    state_dict, checkpoint = load_checkpoint_payload(checkpoint_path, device)
    model_config, diffusion_config, dataset_config = config_from_checkpoint(checkpoint)

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

    dataset_source = args.dataset_source or dataset_config.get("source") or "folder"
    dataset_path_value = args.dataset_path or dataset_config.get("path") or "data/celeba_hq_256"
    dataset_path = resolve_project_path(dataset_path_value) if dataset_path_value else None
    folder_subset_size = (
        args.folder_subset_size
        if args.folder_subset_size is not None
        else dataset_config.get("folder_subset_size")
    )
    folder_test_size = (
        args.folder_test_size
        if args.folder_test_size is not None
        else int(dataset_config.get("folder_test_size") or args.num_images)
    )
    split_seed = args.split_seed if args.split_seed is not None else int(dataset_config.get("split_seed", 42))
    num_workers = args.num_workers if args.num_workers is not None else int(dataset_config.get("num_workers", 0))

    if dataset_source == "folder" and folder_test_size < args.num_images:
        raise ValueError(
            f"folder_test_size={folder_test_size} is smaller than num_images={args.num_images}; "
            "FID needs the requested number of real test images."
        )

    print("Project root:", _ROOT)
    print("Checkpoint:", checkpoint_path)
    print("Output dir:", out_dir)
    print("Device:", device)
    print("FID device:", fid_device)
    print(
        "Experiment params:",
        json.dumps(
            {
                "num_images": args.num_images,
                "batch_size": args.batch_size,
                "fid_batch_size": args.fid_batch_size,
                "ddpm_steps": timesteps,
                "ddim_steps": args.ddim_steps,
                "ddim_eta": args.ddim_eta,
                "image_size": image_size,
                "dataset_source": dataset_source,
                "dataset_path": str(dataset_path) if dataset_path is not None else None,
                "folder_subset_size": folder_subset_size,
                "folder_test_size": folder_test_size,
                "split_seed": split_seed,
            },
            indent=2,
        ),
    )

    loader_start = time.perf_counter()
    _, _, test_loader = create_dataloaders(
        batch_size=args.fid_batch_size,
        image_size=image_size,
        dataset_source=dataset_source,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        seed=split_seed,
        folder_subset_size=folder_subset_size,
        folder_test_size=folder_test_size,
        num_workers=num_workers,
    )
    loader_seconds = time.perf_counter() - loader_start
    if test_loader is None:
        raise ValueError("No test loader exists. Use dataset_source='folder' with --folder_test_size >= --num_images.")

    model = build_model(
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
    ddim_sampler = DDIMSampler(
        scheduler,
        sampling_steps=args.ddim_steps,
        eta=args.ddim_eta,
        clip_denoised=not args.no_clip_denoised,
    )

    print(f"Collecting {args.num_images} real test images for FID.")
    real_start = time.perf_counter()
    real_images = collect_real_fid_images(test_loader, num_images=args.num_images)
    real_collection_seconds = time.perf_counter() - real_start
    save_preview_grid(
        real_images,
        out_dir / "real_test_grid.png",
        nrow=args.grid_rows,
        max_images=args.max_preview,
    )

    feature_extractor = InceptionFeatureExtractor().to(fid_device)
    real_features, real_feature_seconds = timed_collect_features(
        real_images,
        feature_extractor,
        device=fid_device,
        batch_size=args.fid_batch_size,
    )

    results = []
    for sampler_name in ("ddpm", "ddim"):
        sampler_eval_start = time.perf_counter()
        sampler_dir = out_dir / sampler_name
        images, generation_timing = generate_with_sampler(
            sampler_name=sampler_name,
            model=model,
            scheduler=scheduler,
            ddim_sampler=ddim_sampler,
            image_size=image_size,
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            save_dir=sampler_dir,
        )
        save_preview_grid(
            images,
            out_dir / f"{sampler_name}_grid.png",
            nrow=args.grid_rows,
            max_images=args.max_preview,
        )
        features, feature_seconds = timed_collect_features(
            images,
            feature_extractor,
            device=fid_device,
            batch_size=args.fid_batch_size,
        )
        fid = calculate_fid_from_features(real_features, features)
        total_eval_seconds = time.perf_counter() - sampler_eval_start

        sampling_steps = timesteps if sampler_name == "ddpm" else args.ddim_steps
        eta = "" if sampler_name == "ddpm" else args.ddim_eta
        row = {
            "sampler": sampler_name,
            "fid": f"{fid:.6f}",
            "num_images": args.num_images,
            "sampling_steps": sampling_steps,
            "eta": eta,
            "generation_seconds": f"{generation_timing['generation_seconds']:.4f}",
            "seconds_per_image": f"{generation_timing['seconds_per_image']:.6f}",
            "images_per_second": f"{generation_timing['images_per_second']:.6f}"
            if generation_timing["images_per_second"] is not None
            else "",
            "feature_seconds": f"{feature_seconds:.4f}",
            "total_eval_seconds": f"{total_eval_seconds:.4f}",
            "generated_dir": str(sampler_dir),
        }
        results.append(row)
        print(
            f"{sampler_name.upper()} | FID: {fid:.4f} | "
            f"generation: {generation_timing['generation_seconds']:.2f}s | "
            f"{generation_timing['seconds_per_image']:.4f}s/image | "
            f"feature extraction: {feature_seconds:.2f}s | "
            f"total eval: {total_eval_seconds:.2f}s"
        )

    ddpm_fid = float(next(row["fid"] for row in results if row["sampler"] == "ddpm"))
    ddim_fid = float(next(row["fid"] for row in results if row["sampler"] == "ddim"))
    total_script_seconds = time.perf_counter() - script_start

    summary = {
        "checkpoint": str(checkpoint_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "device": device,
        "fid_device": fid_device,
        "model": {
            "image_size": image_size,
            "base_channels": base_channels,
            "time_dim": time_dim,
            "channel_mults": list(channel_mults),
            "num_res_blocks": num_res_blocks,
            "dropout": dropout,
            "attention_resolutions": list(attention_resolutions),
        },
        "diffusion": {
            "timesteps": timesteps,
            "noise_schedule": noise_schedule,
            "noise_max_beta": noise_max_beta,
        },
        "dataset": {
            "source": dataset_source,
            "path": str(dataset_path) if dataset_path is not None else None,
            "folder_subset_size": folder_subset_size,
            "folder_test_size": folder_test_size,
            "split_seed": split_seed,
        },
        "sampling": {
            "ddpm_steps": timesteps,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "clip_denoised": not args.no_clip_denoised,
            "same_initial_noise_per_batch": True,
        },
        "timing": {
            "loader_setup_seconds": loader_seconds,
            "real_image_collection_seconds": real_collection_seconds,
            "real_feature_seconds": real_feature_seconds,
            "total_script_seconds": total_script_seconds,
        },
        "fid_delta_ddpm_minus_ddim": ddpm_fid - ddim_fid,
        "results": results,
    }
    write_results(out_dir, summary)

    better = "DDIM" if ddim_fid < ddpm_fid else "DDPM"
    print("\n=== Final comparison ===")
    for row in results:
        print(
            f"{row['sampler'].upper()}: FID={row['fid']}, "
            f"steps={row['sampling_steps']}, "
            f"generation_seconds={row['generation_seconds']}, "
            f"seconds_per_image={row['seconds_per_image']}, "
            f"images_per_second={row['images_per_second']}"
        )
    print(f"FID delta DDPM - DDIM: {ddpm_fid - ddim_fid:.6f}")
    print(f"Better FID in this run: {better}")
    print(f"Total script time: {total_script_seconds:.2f}s")
    print(f"Wrote summary: {(out_dir / 'comparison_summary.json').resolve()}")
    print(f"Wrote CSV: {(out_dir / 'fid_comparison.csv').resolve()}")


if __name__ == "__main__":
    main()
