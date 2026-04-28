"""DDPM Training Loop

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
from src.diffusion.sample import sample  # noqa: E402
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


@torch.no_grad()
def save_generated_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    save_path: Path,
    batch_size: int,
) -> None:
    images = sample(
        model,
        scheduler,
        image_size=image_size,
        batch_size=batch_size,
        channels=3,
        device=device,
    )
    images = (images.clamp(-1, 1) + 1) * 0.5
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, save_path, nrow=min(4, batch_size))


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
    dataset_source: str = "hf",
    dataset_path: str | None = None,
    folder_subset_size: int | None = None,
    folder_test_size: int = 0,
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

        dataset_source:
            Dataset backend. Use "hf" for the butterfly dataset or "folder" for
            a local image directory such as an unzipped Kaggle dataset.

        dataset_path:
            Root folder containing images when dataset_source is "folder".

        folder_subset_size:
            Optional cap on how many images to use from a folder dataset.

        folder_test_size:
            Number of images reserved for the test split when using a folder dataset.

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
            "base_channels": 64,
            "time_dim": 256,
        },
        "checkpoints": {
            "pattern": "ddpm_epoch_{epoch}.pth",
            "final": "ddpm_final.pth",
        },
        "samples": {
            "directory": "generated_samples",
            "pattern": "epoch_{epoch}.png",
            "sample_every": sample_every,
            "num_images": num_sample_images,
        },
    }
    write_run_metadata(save_dir, run_metadata)

    # --------------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------------
    train_loader, _, _ = create_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        folder_subset_size=folder_subset_size,
        folder_test_size=folder_test_size,
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    sample_dir = Path(save_dir) / "generated_samples"

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

        print(f"Epoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.6f}")

        # ----------------------------------------------------------
        # 4. Save checkpoint after each epoch
        # ----------------------------------------------------------
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
    parser.add_argument("--dataset_source", type=str, default="hf")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--folder_subset_size", type=int, default=None)
    parser.add_argument("--folder_test_size", type=int, default=0)
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
        dataset_source=args.dataset_source,
        dataset_path=args.dataset_path,
        folder_subset_size=args.folder_subset_size,
        folder_test_size=args.folder_test_size,
    )

    print("Training complete.")
    print("Loss history:", losses)
