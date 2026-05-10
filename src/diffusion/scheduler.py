# diffusion/scheduler.py
"""
Diffusion Scheduler (cosine noise schedule)

This file:
- Defines the noise schedule used in diffusion training and sampling
- Implements the forward diffusion process
- Implements one reverse denoising step

References:
- Ho et al., "Denoising Diffusion Probabilistic Models", 2020
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", 2021
  (cosine schedule, Section 3.2)

Note:
- This is a simple scheduler for unconditional image generation.
- It includes only parts needed for:
    1. adding noise during training
    2. stepping backwards during sampling

Noise schedule — cosine (Nichol & Dhariwal 2021):
    f(t) = cos( (t/T + s) / (1 + s) * pi/2 )^2
    alpha_bar_t = f(t) / f(0)
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}   (clipped to 0.9999)

    Compared to the original linear beta schedule, the cosine schedule keeps
    alpha_bar_t higher for longer, which means images retain more structure
    at intermediate timesteps. This leads to better sample quality.

Forward process:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    where epsilon ~ N(0, I)

Reverse mean used during sampling:
    mu_theta(x_t, t) = 1/sqrt(alpha_t) * (
        x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t)
    )

Then we sample:
    x_{t-1} ~ N(mu_theta(x_t, t), posterior_variance_t * I)

This implementation assumes:
- the model predicts noise epsilon_theta(x_t, t)
- timesteps are integer indices in [0, T-1]
"""

import math

import torch


class DDPMScheduler:
    """
    This class precomputes all diffusion coefficients once in __init__,
    then provides methods for:
    - adding noise to clean images during training
    - taking a reverse denoising step during sampling
    """

    def __init__(
        self,
        timesteps: int = 1000,
        device: str = "cpu",
        s: float = 0.008,
    ):
        """
        Args:
            timesteps:
                Total number of diffusion steps T.

            s:
                Small offset for the cosine schedule (Nichol & Dhariwal, 2021).
                Prevents beta from being too small near t=0. Default 0.008.
        """
        self.timesteps = timesteps
        self.device = torch.device(device)

        # 1. Cosine noise schedule (Nichol & Dhariwal, 2021)
        # ------------------------------------------------------------------
        # Compute alpha_bar_t directly from the cosine function, then derive
        # betas from consecutive ratios. This gives a much smoother schedule
        # than a linear beta ramp: signal decays slowly at first and faster
        # near the end, keeping more image structure at mid-timesteps.
        #
        #   f(t) = cos( (t/T + s) / (1 + s) * pi/2 )^2
        #   alpha_bar_t = f(t) / f(0)
        #   beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}

        steps = timesteps + 1  # need t = 0, 1, ..., T
        t = torch.linspace(0, timesteps, steps, device=device) / timesteps
        f = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2

        # alpha_cumprod for t = 1 ... T (shape [T])
        alpha_cumprod_full = f / f[0]
        self.alpha_cumprod = alpha_cumprod_full[1:].clamp(min=1e-8)

        # Derive betas: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        # Clip to 0.9999 to avoid numerical issues at the final timestep.
        self.betas = (1.0 - self.alpha_cumprod / alpha_cumprod_full[:-1]).clamp(max=0.9999)
        self.alphas = 1.0 - self.betas

        # alpha_cumprod from the previous timestep; alpha_cumprod_prev[0] = 1
        # because before any noise is added the image is still perfectly clean.
        self.alpha_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]],
            dim=0,
        )

        # 2. Precompute square roots
        # ------------------------------------------------------------------
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # During reverse sampling, we use 1 / sqrt(alpha_t)
        # in the mean formula for p(x_{t-1} | x_t).
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)

        # 3. Posterior variance for reverse sampling
        # ------------------------------------------------------------------
        # beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to a clean image x_0 to obtain x_t.

        Implements the forward diffusion equation:

            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        
        - During training, we dont simulate all steps one by one.
        - Instead, we jump directly from x_0 to x_t using this formula.
        - This is the standard DDPM training approach.

        Args:
            x_start:
                Clean images x_0 of shape [B, C, H, W]

            noise:
                Gaussian noise epsilon with same shape as x_start

            timesteps:
                Tensor of timestep indices of shape [B]
                Each image in the batch can have a different timestep.

        Returns:
            Noisy images x_t of shape [B, C, H, W]
        """
        # Gather the precomputed coefficient for each image in batch.
        # Since timesteps has shape [B], indexing gives [B], and we reshape to
        # [B, 1, 1, 1] so broadcasting works across channels and spatial dimensions.
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].view(
            -1, 1, 1, 1
        )

        # Apply the forward noising equation.
        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t

    @torch.no_grad()
    def sample_previous_timestep(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one reverse denoising step:
            x_t -> x_{t-1}

        Reverse mean formula:
            mu_theta(x_t, t) =
                1 / sqrt(alpha_t) * (
                    x_t - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise
                )

        Then:
            x_{t-1} = mu_theta(x_t, t) + sqrt(posterior_variance_t) * z
            where z ~ N(0, I), except when t = 0

        Why there is no noise at t = 0:
        - At the final denoising step we want to return the final sample
          not inject randomness again.

        Args:
            x_t:
                Current noisy image at timestep t, shape [B, C, H, W]

            pred_noise:
                Model prediction for the noise in x_t, same shape as x_t

            timesteps:
                Current timestep indices, shape [B]

        Returns:
            A sample for x_{t-1}, shape [B, C, H, W]
        """
        # Extract the schedule values needed for the current timestep of each image
        # and reshape so they broadcast across channels and spatial dimensions.
        beta_t = self.betas[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].view(
            -1, 1, 1, 1
        )
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[timesteps].view(-1, 1, 1, 1)
        

        # Compute the mean of p(x_{t-1} | x_t)
        # ------------------------------------------------------------------
        # This is reverse DDPM update
        #
        # - x_t contains both img signal and noise
        # - pred_noise estimates the noise component
        # - subtracting that estimated noise moves us to cleaner img
        model_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        )

        # Posterior variance determines how much random noise to inject at this step.
        posterior_variance_t = self.posterior_variance[timesteps].view(-1, 1, 1, 1)

        # Sample standard Gaussian noise with same shape as the img.
        noise = torch.randn_like(x_t)

        # We only add noise when t > 0.
        # For the last step (t == 0), we should return the mean directly.
        nonzero_mask = (timesteps > 0).float().view(-1, 1, 1, 1)

        # Final reverse update:
        # x_{t-1} = mean + noise_term
        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise
        return x_prev

    def __len__(self) -> int:
        """
        Allow len(scheduler) to return the total number of timesteps.
        """
        return self.timesteps


class DDIMScheduler(DDPMScheduler):
    """
    DDIM sampler (Song et al., 2020).

    Uses the same trained model and noise schedule as DDPM but replaces the
    stochastic reverse step with a deterministic (or low-noise) update that
    operates on a small subsequence of timesteps.

    Key properties:
    - eta = 0  → fully deterministic (DDIM)
    - eta = 1  → recovers DDPM variance
    - ddim_steps << T → much faster sampling (e.g. 50 steps vs 1000)

    Reverse update for one DDIM step (t → t_prev):

        pred_x0 = (x_t - sqrt(1 - ā_t) * eps_theta) / sqrt(ā_t)
        x_{t_prev} = sqrt(ā_{t_prev}) * pred_x0
                   + sqrt(1 - ā_{t_prev} - σ²) * eps_theta
                   + σ * ε

    where σ = eta * sqrt((1 - ā_{t_prev}) / (1 - ā_t)) * sqrt(1 - ā_t / ā_{t_prev})
    """

    def __init__(
        self,
        timesteps: int = 1000,
        device: str = "cpu",
        s: float = 0.008,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ):
        """
        Args:
            ddim_steps:
                Number of denoising steps to actually perform (<<T for speed).

            eta:
                Controls stochasticity. 0 = deterministic DDIM, 1 = DDPM-level noise.
        """
        super().__init__(timesteps, device, s)
        self.ddim_steps = ddim_steps
        self.eta = eta

        step_ratio = timesteps // ddim_steps
        # Evenly spaced subsequence in descending order, e.g. [999, 979, …, 19]
        self.ddim_timesteps = list(reversed(range(step_ratio - 1, timesteps, step_ratio)))

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        """
        One DDIM reverse step: x_t → x_{t_prev}.

        Args:
            x_t:      Current noisy image, shape [B, C, H, W].
            pred_noise: Model noise prediction at timestep t, same shape.
            t:        Current (higher) timestep index.
            t_prev:   Target (lower) timestep index; use -1 to signal t_prev = 0
                      boundary where alpha_bar_prev should be 1.0.
        """
        abar_t = self.alpha_cumprod[t]
        abar_prev = (
            self.alpha_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=self.device)
        )

        # Reconstruct x0 estimate from x_t and predicted noise
        pred_x0 = (x_t - torch.sqrt(1.0 - abar_t) * pred_noise) / torch.sqrt(abar_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # DDIM sigma — 0 when eta=0 (fully deterministic)
        sigma_t = (
            self.eta
            * torch.sqrt((1.0 - abar_prev) / (1.0 - abar_t))
            * torch.sqrt(1.0 - abar_t / abar_prev)
        )

        # Direction pointing toward x_t
        dir_xt = torch.sqrt(1.0 - abar_prev - sigma_t ** 2) * pred_noise

        noise = torch.randn_like(x_t) if self.eta > 0.0 else 0.0

        return torch.sqrt(abar_prev) * pred_x0 + dir_xt + sigma_t * noise