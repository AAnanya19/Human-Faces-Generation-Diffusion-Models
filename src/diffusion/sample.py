"""
DDPM sampling utilities for unconditional image generation.

The sampling loop starts from Gaussian noise and repeatedly applies the
scheduler's reverse step until it reaches x_0.
"""

import torch


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    scheduler,
    image_size: int,
    batch_size: int,
    channels: int = 3,
    device: str = "cpu",
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a batch of images by iterating the reverse diffusion process."""
    model.eval()

    # Start from Gaussian noise (or provided initial noise).
    if initial_noise is None:
        x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)
    else:
        x_t = initial_noise.to(device)

    # Iterate from T timesteps down to 0, progressively denoising.
    for t in reversed(range(len(scheduler))):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict the noise present at this timestep.
        pred_noise = model(x_t, timesteps)

        # Take one reverse diffusion step toward a cleaner image.
        x_t = scheduler.sample_previous_timestep(x_t, pred_noise, timesteps)

    return x_t


@torch.no_grad()
def sample_with_trajectory(
    model: torch.nn.Module,
    scheduler,
    image_size: int,
    batch_size: int,
    channels: int = 3,
    device: str = "cpu",
    save_every: int = 100,
    initial_noise: torch.Tensor | None = None,
):
    """Generate images and store intermediate denoising states for inspection."""
    model.eval()

    # Start from Gaussian noise (or provided initial noise).
    if initial_noise is None:
        x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)
    else:
        x_t = initial_noise.to(device)

    # Track the denoising trajectory at regular intervals.
    trajectory = [x_t.detach().cpu()]

    # Iterate from T timesteps down to 0, progressively denoising.
    for t in reversed(range(len(scheduler))):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict the noise present at this timestep.
        pred_noise = model(x_t, timesteps)

        # Take one reverse diffusion step toward a cleaner image.
        x_t = scheduler.sample_previous_timestep(x_t, pred_noise, timesteps)

        # Save intermediate states for visualization.
        if t % save_every == 0 or t == 0:
            trajectory.append(x_t.detach().cpu())

    return x_t, trajectory
