"""
Fréchet Inception Distance (FID) between two sets of images on disk.

Expects RGB images saved as PNG/JPEG. Images are resized to a fixed square size,
loaded as float tensors in [0, 1], then converted to ``uint8`` in [0, 255] for the
Inception backbone (required when ``torch-fidelity`` is installed).

If your sampling code outputs tensors in [-1, 1], denormalize to [0, 1] before
saving files, or add a small helper next to the inference script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_image_paths(root: str | Path) -> list[Path]:
    """Return sorted paths to PNG/JPEG/WebP images under ``root`` (recursive)."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


class _ImageListDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as im:
            rgb = im.convert("RGB")
        return self.transform(rgb)


def _batched(
    loader: DataLoader,
) -> Iterator[torch.Tensor]:
    for batch in loader:
        yield batch


def _float01_batch_to_uint8(batch: torch.Tensor) -> torch.Tensor:
    """Convert NCHW float in [0, 1] to uint8 [0, 255] for torch-fidelity FID."""
    return (batch * 255.0).clamp(0.0, 255.0).to(torch.uint8)


def compute_fid_from_directories(
    real_dir: str | Path,
    fake_dir: str | Path,
    *,
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Compute FID between all images under ``real_dir`` and ``fake_dir``.

    Args:
        real_dir: Directory of reference images (e.g. test set exports).
        fake_dir: Directory of generated images.
        image_size: All images are resized to (S, S) so batches stack cleanly.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers (0 is safest on notebooks).
        device: Compute device; defaults to CUDA if available else CPU.

    Returns:
        Scalar tensor with the FID score.
    """
    real_path = Path(real_dir)
    fake_path = Path(fake_dir)
    real_paths = list_image_paths(real_path)
    fake_paths = list_image_paths(fake_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    real_ds = _ImageListDataset(real_paths, transform)
    fake_ds = _ImageListDataset(fake_paths, transform)

    real_loader = DataLoader(
        real_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    fake_loader = DataLoader(
        fake_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    fid = FrechetInceptionDistance(feature=2048).to(device)

    for batch in _batched(real_loader):
        x = _float01_batch_to_uint8(batch.to(device))
        fid.update(x, real=True)
    for batch in _batched(fake_loader):
        x = _float01_batch_to_uint8(batch.to(device))
        fid.update(x, real=False)

    return fid.compute()
