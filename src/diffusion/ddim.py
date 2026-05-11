"""DDIM sampling for epsilon-prediction diffusion models.

DDIM uses the same trained denoising model and beta schedule as DDPM, but
skips across a smaller set of reverse timesteps. With eta=0 it is deterministic
for a fixed initial noise tensor; eta>0 adds stochasticity.
"""

from __future__ import annotations

import torch


class DDIMSampler:
    """Sampler implementing DDIM updates on top of a DDPMScheduler."""

    def __init__(
        self,
        scheduler,
        *,
        sampling_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> None:
        if sampling_steps < 2:
            raise ValueError("sampling_steps must be at least 2")
        if sampling_steps > len(scheduler):
            raise ValueError(
                f"sampling_steps={sampling_steps} cannot exceed scheduler timesteps={len(scheduler)}"
            )
        if eta < 0:
            raise ValueError("eta must be non-negative")

        self.scheduler = scheduler
        self.sampling_steps = sampling_steps
        self.eta = eta
        self.clip_denoised = clip_denoised
        self.timesteps = self._make_timesteps()

    def _make_timesteps(self) -> list[int]:
        """Return descending DDIM timesteps, always including T-1 and 0."""
        timesteps = torch.linspace(
            0,
            len(self.scheduler) - 1,
            steps=self.sampling_steps,
            dtype=torch.float64,
        )
        timesteps = timesteps.round().long().unique(sorted=True)
        if timesteps.numel() != self.sampling_steps:
            raise ValueError(
                "sampling_steps produced duplicate timestep indices; choose fewer DDIM steps"
            )
        return list(reversed(timesteps.tolist()))

    def _extract_alpha_cumprod(
        self,
        timesteps: torch.Tensor,
        *,
        default_value: float,
        target: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = timesteps >= 0
        safe_timesteps = timesteps.clamp(min=0)
        values = self.scheduler.alpha_cumprod[safe_timesteps].view(-1, 1, 1, 1)
        default = torch.full_like(values, default_value)
        return torch.where(valid_mask.view(-1, 1, 1, 1), values, default).to(target.device)

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        timesteps: torch.Tensor,
        previous_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one DDIM reverse update from x_t to x_previous_t."""
        alpha_t = self._extract_alpha_cumprod(
            timesteps,
            default_value=1.0,
            target=x_t,
        )
        alpha_prev = self._extract_alpha_cumprod(
            previous_timesteps,
            default_value=1.0,
            target=x_t,
        )

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        if self.clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        sigma = self.eta * torch.sqrt(
            ((1.0 - alpha_prev) / (1.0 - alpha_t)).clamp_min(0.0)
            * (1.0 - (alpha_t / alpha_prev)).clamp_min(0.0)
        )
        direction_scale = torch.sqrt((1.0 - alpha_prev - sigma.square()).clamp_min(0.0))
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + direction_scale * pred_noise

        if self.eta > 0:
            noise = torch.randn_like(x_t)
            nonzero_mask = (previous_timesteps >= 0).float().view(-1, 1, 1, 1)
            x_prev = x_prev + nonzero_mask * sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        *,
        image_size: int,
        batch_size: int,
        channels: int = 3,
        device: str = "cpu",
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a batch of images using DDIM reverse sampling."""
        model.eval()
        if initial_noise is None:
            x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)
        else:
            x_t = initial_noise.to(device)
            batch_size = x_t.shape[0]

        for index, timestep in enumerate(self.timesteps):
            previous_timestep = self.timesteps[index + 1] if index + 1 < len(self.timesteps) else -1
            timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            previous_timesteps = torch.full(
                (batch_size,),
                previous_timestep,
                device=device,
                dtype=torch.long,
            )
            pred_noise = model(x_t, timesteps)
            x_t = self.step(x_t, pred_noise, timesteps, previous_timesteps)

        return x_t
