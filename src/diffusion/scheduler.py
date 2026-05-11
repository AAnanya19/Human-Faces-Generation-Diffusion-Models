"""DDPM scheduler for unconditional image generation."""

import math
import torch


class DDPMScheduler:
    """Precompute diffusion coefficients and expose noising/sampling steps."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        noise_schedule: str = "linear",
        noise_max_beta: float = 0.999,
        cosine_s: float = 0.008,
        device: str = "cpu",
    ):
        """Initialise the schedule and derived coefficients."""
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.noise_schedule = noise_schedule.lower()
        self.noise_max_beta = noise_max_beta

       
        self.betas = self._make_beta_schedule(
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )

        self.alphas = 1.0 - self.betas

        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        # For t = 0, the previous cumulative product is defined as 1.0.
        self.alpha_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]],
            dim=0,
        )

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)

        # Posterior variance for the reverse process.
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def _make_beta_schedule(
        self,
        beta_start: float,
        beta_end: float,
        cosine_s: float,
    ) -> torch.Tensor:
        """Build the beta schedule."""
        if self.noise_schedule == "linear":
            return torch.linspace(
                beta_start,
                beta_end,
                self.timesteps,
                device=self.device,
            )

        if self.noise_schedule == "cosine":
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps, device=self.device)
            alpha_cumprod = torch.cos(
                ((x / self.timesteps) + cosine_s)
                / (1.0 + cosine_s)
                * math.pi
                * 0.5
            ) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1.0 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            return betas.clamp(min=1e-8, max=self.noise_max_beta)

        raise ValueError(
            f"Unknown noise_schedule '{self.noise_schedule}'."
        )

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise using the closed-form DDPM equation."""
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].view(
            -1, 1, 1, 1
        )

        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t

    @torch.no_grad()
    def sample_previous_timestep(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Take one reverse DDPM step."""
        beta_t = self.betas[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].view(
            -1, 1, 1, 1
        )
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[timesteps].view(-1, 1, 1, 1)

        model_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        )

        posterior_variance_t = self.posterior_variance[timesteps].view(-1, 1, 1, 1)

        noise = torch.randn_like(x_t)

        nonzero_mask = (timesteps > 0).float().view(-1, 1, 1, 1)

        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise
        return x_prev

    def __len__(self) -> int:
        """Return the total number of timesteps."""
        return self.timesteps
