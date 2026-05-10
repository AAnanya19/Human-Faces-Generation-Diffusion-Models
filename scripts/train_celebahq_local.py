"""Local CelebA-HQ training launcher."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train import resolve_device, train  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}




def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false, yes/no, 1/0, or on/off."
    )


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    return resolved


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def zip_has_top_level_folder(zip_path: Path, folder_name: str) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        top_levels = {
            Path(info.filename).parts[0]
            for info in zip_file.infolist()
            if info.filename and not info.filename.startswith("__MACOSX/")
        }
    return top_levels == {folder_name}


def extract_dataset_zip(zip_path: Path, dataset_dir: Path) -> None:
    extract_root = (
        dataset_dir.parent
        if zip_has_top_level_folder(zip_path, dataset_dir.name)
        else dataset_dir
    )
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_root)


def prepare_dataset(
    dataset_dir: Path,
    dataset_zip: Path | None,
    expected_image_count: int,
) -> int:
    image_count = count_images(dataset_dir)
    if image_count > 0:
        return image_count
    
    if dataset_zip is not None and dataset_zip.is_file():
        print(f"Extracting dataset zip: {dataset_zip}")
        extract_dataset_zip(dataset_zip, dataset_dir)
        image_count = count_images(dataset_dir)

    if image_count == 0:
        raise FileNotFoundError(
            f"No images found in {dataset_dir}. Put CelebA-HQ there, or pass "
            "--dataset_zip data/celeba_hq_256.zip to extract it locally."
        )
    return image_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CelebA-HQ DDPM training locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Local paths and dataset
    local = parser.add_argument_group("Local paths and dataset")
    local.add_argument("--dataset_dir", type=str, default="data/celeba_hq_256")
    local.add_argument("--dataset_zip", type=str, default="data/celeba_hq_256.zip")
    local.add_argument("--expected_image_count", type=int, default=30000)
    local.add_argument("--run_name", type=str, default="celebahq_run_001_baseline")
    local.add_argument("--runs_root", type=str, default="runs/ddpm_runs")
    local.add_argument("--save_dir", type=str, default=None)
    local.add_argument("--allow_cpu", action="store_true")

    # Diffusion params
    diffusion = parser.add_argument_group("Diffusion params")
    diffusion.add_argument("--timesteps", type=int, default=1000)
    diffusion.add_argument("--noise_schedule", choices=["linear", "cosine"], default="linear")

    # Model params
    model = parser.add_argument_group("Model params")
    model.add_argument("--image_size", type=int, default=128)
    model.add_argument("--base_channels", type=int, default=64)
    model.add_argument("--time_dim", type=int, default=256)
    model.add_argument("--channel_mults", type=str, default="1,2,2,4,4")
    model.add_argument("--num_res_blocks", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--attention_resolutions", type=str, default="16,8")

    # Training params
    training = parser.add_argument_group("Training params")
    training.add_argument("--epochs", type=int, default=1000)
    training.add_argument("--batch_size", type=int, default=8)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--lr_scheduler", choices=["fixed", "cosine"], default="fixed")
    training.add_argument("--cosine_t_max", type=int, default=None)
    training.add_argument("--cosine_eta_min", type=float, default=1e-6)
    training.add_argument("--weight_decay", type=float, default=1e-4)
    training.add_argument("--grad_clip", type=float, default=1.0)
    training.add_argument("--checkpoint_every", type=int, default=50)

    training.add_argument("--use_ema", type=parse_bool, default=True)
    training.add_argument("--ema_decay", type=float, default=0.9999)
    training.add_argument("--resume_checkpoint", type=str, default=None)
    training.add_argument("--device", type=str, default=None)

    # Split and dataloader params
    data = parser.add_argument_group("Split and dataloader params")
    data.add_argument("--folder_subset_size", type=int, default=3000)
    data.add_argument("--folder_test_size", type=int, default=300)
    data.add_argument("--split_seed", type=int, default=67)
    data.add_argument("--num_workers", type=int, default=8)

    # Evaluation params
    evaluation = parser.add_argument_group("Evaluation params")
    evaluation.add_argument("--sample_every", type=int, default=50)
    evaluation.add_argument("--num_sample_images", type=int, default=8)
    evaluation.add_argument("--fixed_sample_seed", type=int, default=123)
    evaluation.add_argument("--fixed_trajectory_seed", type=int, default=321)
    evaluation.add_argument("--trajectory_save_every", type=int, default=100)
    evaluation.add_argument("--disable_fid", action="store_true")
    evaluation.add_argument("--fid_every", type=int, default=100)
    evaluation.add_argument("--fid_num_images", type=int, default=300)
    evaluation.add_argument("--fid_batch_size", type=int, default=8)
    evaluation.add_argument("--fid_seed", type=int, default=999)
    evaluation.add_argument("--fid_patience", type=int, default=4)
    evaluation.add_argument("--fid_device", type=str, default=None)
    evaluation.add_argument("--no_save_fid_images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = resolve_project_path(args.dataset_dir)
    dataset_zip = resolve_project_path(args.dataset_zip) if args.dataset_zip else None
    save_dir = (
        resolve_project_path(args.save_dir)
        if args.save_dir
        else resolve_project_path(args.runs_root) / args.run_name
    )

    image_count = prepare_dataset(
        dataset_dir=dataset_dir,
        dataset_zip=dataset_zip,
        expected_image_count=args.expected_image_count,
    )

    device = resolve_device(args.device)
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA or Apple MPS GPU was detected. Use a GPU-enabled PyTorch "
            "install, pass --device cuda or --device mps, or pass --allow_cpu "
            "if you really want to train on CPU."
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", _ROOT)
    print("Dataset dir:", dataset_dir)
    print("Image files found:", image_count)
    print("Run dir:", save_dir)
    print("torch:", torch.__version__)
    print("device:", device)

    train(
        # Diffusion params
        timesteps=args.timesteps,
        noise_schedule=args.noise_schedule,
        # Model params
        image_size=args.image_size,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        channel_mults=tuple(int(x) for x in args.channel_mults.split(",")),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        attention_resolutions=tuple(
            int(x) for x in args.attention_resolutions.split(",")
        ),
        # Training params
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_dir=str(save_dir),
        lr_scheduler=args.lr_scheduler,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_every=args.checkpoint_every,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        resume_checkpoint=args.resume_checkpoint,
        # Split and dataloader params
        dataset_source="folder",
        dataset_path=str(dataset_dir),
        folder_subset_size=args.folder_subset_size,
        folder_test_size=args.folder_test_size,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        # Evaluation params
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        fixed_sample_seed=args.fixed_sample_seed,
        fixed_trajectory_seed=args.fixed_trajectory_seed,
        trajectory_save_every=args.trajectory_save_every,
        enable_fid=not args.disable_fid,
        fid_every=args.fid_every,
        fid_num_images=args.fid_num_images,
        fid_batch_size=args.fid_batch_size,
        fid_seed=args.fid_seed,
        fid_patience=args.fid_patience,
        fid_device=args.fid_device,
        save_fid_images=not args.no_save_fid_images,
    )

    print("Training complete.")
    print("Checkpoints:", save_dir)
    print("Best model:", save_dir / "best_model.pth")
    print("Latest checkpoint:", save_dir / "latest_checkpoint.pth")
    print("Samples:", save_dir / "generated_samples")
    print("Trajectories:", save_dir / "trajectories")
    print("FID log:", save_dir / "fid_log.csv")
    print("Training log:", save_dir / "training_log.csv")
    print("Loss log:", save_dir / "loss_log.csv")
    print("Metadata:", save_dir / "run_config.json")


if __name__ == "__main__":
    main()
