"""Save tensors or image files as a single grid for qualitative review."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image


def save_image_grid(
    images: torch.Tensor,
    out_path: str | Path,
    *,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: tuple[float, float] | None = (-1.0, 1.0),
) -> None:
    """
    Save a batch of images ``images`` (B, C, H, W) to a single PNG.

    Args:
        images: Float tensor batch.
        out_path: Output file path (parent directories are created).
        nrow: Images per row in the grid.
        padding: Pixel padding between cells.
        normalize: Passed to ``save_image``; if True, use ``value_range``.
        value_range: If tensors are in [-1, 1] from training normalization, use
            (-1, 1). For [0, 1] tensors, set ``value_range=(0, 1)`` or ``normalize=False``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        images,
        out_path,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )


def grid_from_paths(
    image_paths: Sequence[str | Path],
    out_path: str | Path,
    *,
    nrow: int = 8,
    padding: int = 2,
) -> None:
    """Build a grid from image files on disk (RGB, loaded as [0, 1])."""
    tensors: list[torch.Tensor] = []
    for p in image_paths:
        with Image.open(Path(p)) as im:
            t = to_tensor(im.convert("RGB"))
        tensors.append(t)
    batch = torch.stack(tensors, dim=0)
    grid = make_grid(batch, nrow=nrow, padding=padding, normalize=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
