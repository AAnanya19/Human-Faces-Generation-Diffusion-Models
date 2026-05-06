"""
Local CelebA-HQ training launcher.

This mirrors notebooks/colab_celebahq_train.ipynb, but keeps the dataset,
checkpoints, samples, trajectories, logs, and metadata on your local machine.

Example:
    python3 scripts/train_celebahq_local.py --dataset_dir data/celeba_hq_256
"""

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
    if expected_image_count <= 0 and image_count > 0:
        return image_count
    if image_count >= expected_image_count:
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
    if expected_image_count > 0 and image_count < expected_image_count:
        raise RuntimeError(
            f"Dataset looks incomplete: found {image_count} images in "
            f"{dataset_dir}, expected about {expected_image_count}. If you are "
            "intentionally training on a smaller local copy, pass "
            "--expected_image_count 0."
        )
    return image_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/celeba_hq_256")
    parser.add_argument("--dataset_zip", type=str, default="data/celeba_hq_256.zip")
    parser.add_argument("--expected_image_count", type=int, default=30000)
    parser.add_argument("--run_name", type=str, default="celebahq_run_001")
    parser.add_argument("--runs_root", type=str, default="runs/ddpm_runs")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--allow_cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--num_sample_images", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--folder_subset_size", type=int, default=3000)
    parser.add_argument("--folder_test_size", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--channel_mults", type=str, default="1,2,4,8")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_resolutions", type=str, default="16,8")
    parser.add_argument("--fixed_sample_seed", type=int, default=123)
    parser.add_argument("--fixed_trajectory_seed", type=int, default=321)
    parser.add_argument("--trajectory_save_every", type=int, default=100)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        timesteps=args.timesteps,
        device=device,
        save_dir=str(save_dir),
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        checkpoint_every=args.checkpoint_every,
        dataset_source="folder",
        dataset_path=str(dataset_dir),
        folder_subset_size=args.folder_subset_size,
        folder_test_size=args.folder_test_size,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        channel_mults=tuple(int(x) for x in args.channel_mults.split(",")),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        attention_resolutions=tuple(
            int(x) for x in args.attention_resolutions.split(",")
        ),
        fixed_sample_seed=args.fixed_sample_seed,
        fixed_trajectory_seed=args.fixed_trajectory_seed,
        trajectory_save_every=args.trajectory_save_every,
        resume_checkpoint=args.resume_checkpoint,
    )

    print("Training complete.")
    print("Checkpoints:", save_dir)
    print("Samples:", save_dir / "generated_samples")
    print("Trajectories:", save_dir / "trajectories")
    print("Loss log:", save_dir / "loss_log.csv")
    print("Metadata:", save_dir / "run_config.json")


if __name__ == "__main__":
    main()
