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
- The training objective follows the standard DDPM formulation:
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
from tqdm import tqdm  # noqa: E402

from src.data.butterfly_dataset import create_dataloaders  # noqa: E402
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


def train(
    epochs: int = 20,
    batch_size: int = 32,
    image_size: int = 64,
    lr: float = 1e-4,
    timesteps: int = 1000,
    device: str = "cpu",
    save_dir: str = "checkpoints",
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
        "timesteps": timesteps,
        "device": device,
        "save_dir": save_dir,
        "model": {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 64,
            "time_dim": 256,
        },
        "checkpoints": {
            "pattern": "ddpm_epoch_{epoch}.pth",
            "final": "ddpm_final.pth",
        },
    }
    write_run_metadata(save_dir, run_metadata)

    # --------------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
    )

    # --------------------------------------------------------------
    # 2. Create model, scheduler, loss, optimizer
    # --------------------------------------------------------------
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_dim=256,
    ).to(device)

    scheduler = DDPMScheduler(
        timesteps=timesteps,
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

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
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.6f}")

        # ----------------------------------------------------------
        # 4. Save checkpoint after each epoch
        # ----------------------------------------------------------
        checkpoint_path = os.path.join(save_dir, f"ddpm_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

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
    )

    print("Training complete.")
    print("Loss history:", losses)
