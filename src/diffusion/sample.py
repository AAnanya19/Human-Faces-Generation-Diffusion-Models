# diffusion/sample.py
"""
DDPM Sampling

This file:
- Runs the full reverse diffusion loop for image generation
- Starts from pure Gaussian noise
- Uses the trained model to predict the noise at each timestep
- Uses the scheduler to move from x_t to x_{t-1}

Reference:
- The reverse denoising procedure follows DDPM:
  Ho et al., "Denoising Diffusion Probabilistic Models", 2020
- The scheduler contains the reverse step 

Note:
- This implementation assumes:
    1. the model predicts noise epsilon_theta(x_t, t)
    2. the scheduler handles one reverse denoising step at a time

Mathematical summary:
Sampling starts from:
    x_T ~ N(0, I)

Then for t = T-1, T-2, ..., 0:
    1. predict noise using the model
    2. compute x_{t-1} using the scheduler

At the end of the loop:
    x_0 is the final generated image
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
    """
    Generate a batch of images using the full reverse diffusion process.

    This function:
    - starts from random Gaussian noise
    - loops backwards through all timesteps
    - predicts the noise in the current sample at each step
    - applies one reverse denoising step through the scheduler

    Args:
        model:
            The trained denoising model.

            Expected input:
                - x_t of shape [B, C, H, W]
                - timesteps of shape [B]

            Expected output:
                - predicted noise of shape [B, C, H, W]

        scheduler:
            It is responsible for one reverse denoising step.

        image_size:
            Height and width of generated images.

        batch_size:
            Number of images to generate.

        channels:
            Number of image channels.


    Returns:
        Final generated images of shape [B, C, H, W]
    """
    model.eval()

    # 1. Start from pure Gaussian noise
    # ------------------------------------------------------------------
    # x_T in the diffusion process.
    # At this point there is no image structure, only random noise.
    if initial_noise is None:
        x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)
    else:
        x_t = initial_noise.to(device)

    # 2. Reverse diffusion loop
    # ------------------------------------------------------------------
    # We move backwards through time:
    #   T-1, T-2, ..., 0
    #
    # At each step:
    # - create a timestep tensor for the full batch
    # - let the model predict the noise in x_t
    # - let the scheduler compute x_{t-1}
    for t in reversed(range(len(scheduler))):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict the noise present in the current sample x_t
        pred_noise = model(x_t, timesteps)

        # Use the scheduler to take one reverse step:
        #   x_t -> x_{t-1}
        x_t = scheduler.sample_previous_timestep(x_t, pred_noise, timesteps)

    # After the loop finishes, x_t should now represent x_0,
    # which is the final generated image.
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
    """
    Generate images while also storing intermediate denoising states.

    Useful for:
    - visualising how noise gradually becomes structure
    - debugging sampling

    Args:
        model:
            The trained denoising model.

        scheduler:
            Instance of DDPMScheduler.

        image_size:
            Height and width of generated images.

        batch_size:
            Number of images to generate.

        channels:
            Number of image channels.

        save_every:
            Save one intermediate state every N timesteps.
            Smaller values save more states.

    Returns:
        final_images:
            Final generated images of shape [B, C, H, W]

        trajectory:
            List of intermediate image tensors stored on CPU
            so they can be visualised later.
    """
    model.eval()

    # Start from pure Gaussian noise x_T
    if initial_noise is None:
        x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)
    else:
        x_t = initial_noise.to(device)

    # Store the initial noise as the first point in the trajectory
    trajectory = [x_t.detach().cpu()]

    # Reverse diffusion loop
    for t in reversed(range(len(scheduler))):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Model predicts the noise component in the current sample
        pred_noise = model(x_t, timesteps)

        # Scheduler computes one reverse step
        x_t = scheduler.sample_previous_timestep(x_t, pred_noise, timesteps)

        # Save selected intermediate steps so we can later inspect
        # how the sample evolves from noise to image
        if t % save_every == 0 or t == 0:
            trajectory.append(x_t.detach().cpu())

    return x_t, trajectory
