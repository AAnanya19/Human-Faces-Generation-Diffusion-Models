# training/train.py
"""
DDPM Training Loop

This file:
- Loads batches of clean images from the dataloader
- Samples random diffusion timesteps for each image
- Adds Gaussian noise using the scheduler
- Trains the U-Net to predict the added noise
- Saves model checkpoints during training

Reference:
- The training follows the standard DDPM formulation:
  predict the noise added to x_0 at a random timestep t
- The loss used here is mean squared error between:
    predicted_noise and true_noise

Note:
- This is a simple training loop for the butterfly toy task
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python src/training/train.py` or running from a parent folder (e.g. aml/):
# project root must be on sys.path so `src` and `data` resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.butterfly_dataset import create_dataloaders  # noqa: E402
from src.diffusion.sample import sample, sample_with_trajectory  # noqa: E402
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


def write_run_metadata(save_dir: str, metadata: dict) -> None:
    metadata_path = Path(save_dir) / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def write_split_files(save_dir: str, split_info: dict | None) -> None:
    if not split_info:
        return
    split_dir = Path(save_dir)
    train_files = split_info.get("train_files")
    test_files = split_info.get("test_files")
    if train_files is not None:
        (split_dir / "train_files.txt").write_text("\n".join(train_files) + "\n")
    if test_files is not None:
        (split_dir / "test_files.txt").write_text("\n".join(test_files) + "\n")


def append_loss_log(save_dir: str, epoch: int, avg_loss: float) -> None:
    log_path = Path(save_dir) / "loss_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,avg_loss\n")
    with log_path.open("a") as f:
        f.write(f"{epoch},{avg_loss:.8f}\n")


@torch.no_grad()
def save_generated_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    save_path: Path,
    batch_size: int,
    seed: int | None = None,
) -> None:
    initial_noise = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        initial_noise = torch.randn(
            batch_size,
            3,
            image_size,
            image_size,
            generator=generator,
            device=device,
        )
    images = sample(
        model,
        scheduler,
        image_size=image_size,
        batch_size=batch_size,
        channels=3,
        device=device,
        initial_noise=initial_noise,
    )
    images = (images.clamp(-1, 1) + 1) * 0.5
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, save_path, nrow=min(4, batch_size))


@torch.no_grad()
def save_trajectory_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    save_path: Path,
    channels: int = 3,
    seed: int = 0,
    save_every: int = 100,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial_noise = torch.randn(1, channels, image_size, image_size, generator=generator, device=device)
    _, trajectory = sample_with_trajectory(
        model,
        scheduler,
        image_size=image_size,
        batch_size=1,
        channels=channels,
        device=device,
        save_every=save_every,
        initial_noise=initial_noise,
    )
    frames = [((frame.clamp(-1, 1) + 1) * 0.5).cpu() for frame in trajectory]
    grid = torch.cat(frames, dim=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, save_path, nrow=len(frames))


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated integers, got: {raw!r}")
    return values


def train(
    epochs: int = 20,
    batch_size: int = 32,
    image_size: int = 64,
    lr: float = 1e-4,
    timesteps: int = 1000,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    sample_every: int = 10,
    num_sample_images: int = 8,
    checkpoint_every: int = 50,
    dataset_source: str = "hf",
    dataset_path: str | None = None,
    folder_subset_size: int | None = None,
    folder_test_size: int = 0,
    base_channels: int = 64,
    time_dim: int = 256,
    channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    dropout: float = 0.1,
    attention_resolutions: tuple[int, ...] = (16, 8),
    fixed_sample_seed: int = 123,
    fixed_trajectory_seed: int = 321,
    trajectory_save_every: int = 100,
):
    """
    Train the DDPM denoising model on the butterfly dataset.

    Args:
        epochs:
            Number of full passes through the training set.

        batch_size:
            Number of images per batch.

        image_size:
            Resolution used for training images.

        lr:
            Learning rate for Adam optimizer.

        timesteps:
            Total number of diffusion steps.

        device:
            Training device, e.g. "cpu" or "cuda".

        save_dir:
            Folder where model checkpoints will be saved.

        weight_decay:
            Weight decay used by AdamW.

        grad_clip:
            Max gradient norm for clipping. Set to 0 to disable.

        sample_every:
            Save generated sample grids every N epochs. Set to 0 to disable.

        num_sample_images:
            Number of generated images to include in each saved grid.

        checkpoint_every:
            Save model checkpoints every N epochs.

        dataset_source:
            Dataset backend. Use "hf" for the butterfly dataset or "folder" for
            a local image directory such as an unzipped Kaggle dataset.

        dataset_path:
            Root folder containing images when dataset_source is "folder".

        folder_subset_size:
            Optional cap on how many images to use from a folder dataset.

        folder_test_size:
            Number of images reserved for the test split when using a folder dataset.

        time_dim:
            Timestep embedding size.

        base_channels:
            Base channel width for the U-Net.

        channel_mults:
            Per-resolution channel multipliers for the U-Net.

        num_res_blocks:
            Number of residual blocks per resolution.

        dropout:
            Dropout used inside residual blocks.

        attention_resolutions:
            Spatial resolutions where attention is applied.

        fixed_sample_seed:
            Seed for deterministic generated sample grids.

        fixed_trajectory_seed:
            Seed for deterministic denoising trajectory visualizations.

        trajectory_save_every:
            Timestep interval for trajectory snapshots.

    Returns:
        model:
            Trained U-Net model

        losses:
            List of average training loss per epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    run_metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "timesteps": timesteps,
        "device": device,
        "save_dir": save_dir,
        "dataset": {
            "source": dataset_source,
            "name": "huggan/smithsonian_butterflies_subset" if dataset_source == "hf" else None,
            "path": dataset_path,
            "folder_subset_size": folder_subset_size,
            "folder_test_size": folder_test_size,
        },
        "model": {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": base_channels,
            "time_dim": time_dim,
            "channel_mults": list(channel_mults),
            "num_res_blocks": num_res_blocks,
            "dropout": dropout,
            "attention_resolutions": list(attention_resolutions),
        },
        "checkpoints": {
            "pattern": "ddpm_epoch_{epoch}.pth",
            "final": "ddpm_final.pth",
            "checkpoint_every": checkpoint_every,
        },
        "samples": {
            "directory": "generated_samples",
            "pattern": "epoch_{epoch}.png",
            "sample_every": sample_every,
            "num_images": num_sample_images,
            "fixed_sample_seed": fixed_sample_seed,
        },
        "trajectories": {
            "directory": "trajectories",
            "pattern": "epoch_{epoch}.png",
            "seed": fixed_trajectory_seed,
            "save_every": trajectory_save_every,
        },
    }
    write_run_metadata(save_dir, run_metadata)

    # --------------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------------
    train_loader, _, _, split_info = create_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        folder_subset_size=folder_subset_size,
        folder_test_size=folder_test_size,
        return_split_info=True,
    )
    write_split_files(save_dir, split_info)
    run_metadata["dataset"]["train_split_file"] = "train_files.txt"
    run_metadata["dataset"]["test_split_file"] = "test_files.txt"
    if split_info is not None:
        train_files = split_info.get("train_files")
        test_files = split_info.get("test_files")
        run_metadata["dataset"]["train_size"] = len(train_files) if train_files is not None else None
        run_metadata["dataset"]["test_size"] = len(test_files) if test_files is not None else None
    write_run_metadata(save_dir, run_metadata)

    # --------------------------------------------------------------
    # 2. Create model, scheduler, loss, optimizer
    # --------------------------------------------------------------
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        time_dim=time_dim,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        image_size=image_size,
    ).to(device)

    scheduler = DDPMScheduler(
        timesteps=timesteps,
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    sample_dir = Path(save_dir) / "generated_samples"
    trajectory_dir = Path(save_dir) / "trajectories"

    # --------------------------------------------------------------
    # 3. Training loop
    # --------------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            # Move clean images to device
            x_start = batch.to(device)

            # Sample random Gaussian noise with same shape as images
            noise = torch.randn_like(x_start)

            # Sample a random timestep for each image in the batch
            batch_size_current = x_start.shape[0]
            timesteps_batch = torch.randint(
                0,
                timesteps,
                (batch_size_current,),
                device=device,
            ).long()

            # Create noisy version x_t from clean image x_0
            x_t = scheduler.add_noise(x_start, noise, timesteps_batch)

            # Predict the noise using the model
            pred_noise = model(x_t, timesteps_batch)

            # Compute DDPM training loss
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        append_loss_log(save_dir, epoch + 1, avg_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.6f}")

        # ----------------------------------------------------------
        # 4. Save checkpoint after each epoch
        # ----------------------------------------------------------
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(save_dir, f"ddpm_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        if sample_every > 0 and (epoch + 1) % sample_every == 0:
            save_generated_samples(
                model,
                scheduler,
                image_size=image_size,
                device=device,
                save_path=sample_dir / f"epoch_{epoch + 1}.png",
                batch_size=num_sample_images,
                seed=fixed_sample_seed,
            )
            save_trajectory_samples(
                model,
                scheduler,
                image_size=image_size,
                device=device,
                save_path=trajectory_dir / f"epoch_{epoch + 1}.png",
                seed=fixed_trajectory_seed,
                save_every=trajectory_save_every,
            )

    # Save final model
    final_path = os.path.join(save_dir, "ddpm_final.pth")
    torch.save(model.state_dict(), final_path)

    run_metadata["loss_history"] = losses
    run_metadata["final_checkpoint"] = final_path
    write_run_metadata(save_dir, run_metadata)

    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--num_sample_images", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--dataset_source", type=str, default="hf")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--folder_subset_size", type=int, default=None)
    parser.add_argument("--folder_test_size", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--channel_mults", type=str, default="1,2,4,8")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_resolutions", type=str, default="16,8")
    parser.add_argument("--fixed_sample_seed", type=int, default=123)
    parser.add_argument("--fixed_trajectory_seed", type=int, default=321)
    parser.add_argument("--trajectory_save_every", type=int, default=100)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print("Device:", device)
    print("Saving checkpoints to:", args.save_dir)

    model, losses = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        timesteps=args.timesteps,
        device=device,
        save_dir=args.save_dir,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        checkpoint_every=args.checkpoint_every,
        dataset_source=args.dataset_source,
        dataset_path=args.dataset_path,
        folder_subset_size=args.folder_subset_size,
        folder_test_size=args.folder_test_size,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        channel_mults=parse_int_list(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        attention_resolutions=parse_int_list(args.attention_resolutions),
        fixed_sample_seed=args.fixed_sample_seed,
        fixed_trajectory_seed=args.fixed_trajectory_seed,
        trajectory_save_every=args.trajectory_save_every,
    )

    print("Training complete.")
    print("Loss history:", losses)
