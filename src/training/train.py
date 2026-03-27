"""General training script.

Notes
- Default loader points to the butterfly dataloader in `data/dataloader.py`.
- The model/loss here are placeholders so the loop runs end-to-end.
- When you switch to diffusion, you mainly change `build_model()` + `compute_loss()`.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def _resolve_device(device: str) -> torch.device:
    # `auto` = cuda if available, else cpu
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _import_attr(path: str):
    """Import `module:attr` (or `module.attr`)."""
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


@dataclass
class DataLoaders:
    train: DataLoader
    val: Optional[DataLoader] = None
    test: Optional[DataLoader] = None


def build_dataloaders(
    *,
    loader_fn_path: str,
    batch_size: int,
    image_size: int,
    seed: int,
    num_workers: int,
) -> DataLoaders:
    """Calls your loader factory and wraps outputs.

    Expected return:
    - either a single train `DataLoader`, or
    - `(train_loader, val_loader, test_loader)`.
    """
    loader_fn = _import_attr(loader_fn_path)
    out = loader_fn(
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        num_workers=num_workers,
    )
    if isinstance(out, tuple) and len(out) == 3:
        return DataLoaders(train=out[0], val=out[1], test=out[2])
    if isinstance(out, DataLoader):
        return DataLoaders(train=out)
    raise TypeError(
        "loader_fn must return a DataLoader or (train, val, test) tuple of DataLoaders"
    )


class TinyConvNet(nn.Module):
    # Placeholder model. Swap this out later.
    def __init__(self, in_ch: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model() -> nn.Module:
    # TODO: replace with UNet
    return TinyConvNet()


def compute_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    pred = model(batch)
    target = batch
    return torch.nn.functional.mse_loss(pred, target)


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: Optional[float],
) -> float:
    # Standard training epoch 
    model.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        loss = compute_loss(model, batch)
        (loss / grad_accum_steps).backward()
        if (step + 1) % grad_accum_steps == 0:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_epoch(*, model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    # Uses the same `compute_loss()` by default.
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        loss = compute_loss(model, batch)
        total += float(loss.detach().cpu())
        n += 1
    return total / max(1, n)


def save_checkpoint(out_dir: Path, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer) -> Path:
    # Minimal checkpoint: model/optim + epoch
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loader",
        type=str,
        default="data.dataloader:create_dataloaders",
        help="Dataloader factory import path (module:attr).",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="results/checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    # Seed for basic reproducibility.
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    loaders = build_dataloaders(
        loader_fn_path=args.loader,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            loader=loaders.train,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=max(1, args.grad_accum_steps),
            max_grad_norm=args.max_grad_norm,
        )
        msg = f"epoch={epoch} train_loss={train_loss:.4f}"

        # Optional validation (only if the loader returns one)
        if loaders.val is not None:
            val_loss = eval_epoch(model=model, loader=loaders.val, device=device)
            msg += f" val_loss={val_loss:.4f}"
        print(msg)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt = save_checkpoint(out_dir, epoch, model, optimizer)
            print(f"saved: {ckpt}")


if __name__ == "__main__":
    main()
