# diffusion/scheduler.py
"""
DDPM Scheduler

This file:
- Defines the noise schedule used in diffusion training and sampling
- Implements the forward diffusion process 
- Implements one reverse denoising step 

Reference
- Ho et al., "Denoising Diffusion Probabilistic Models", 2020
- The scheduler is inspired by standard DDPM implementations
  used in tutorials and libraries such as Hugging Face Diffusers,
  but is reimplemented in a smaller and clearer form
  for this task.

 Note:
- This is a simple scheduler for unconditional image generation.
- It includes only parts needed for:
    1. adding noise during training
    2. stepping backwards during sampling

Mathematical summary:
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
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu",
    ):
        """
        Args:
            timesteps:
                Total number of diffusion steps T.     

            beta_start:
                The first beta value in the variance schedule.
                This should be small because early noising steps should be gentle.

            beta_end:
                The last beta value in the variance schedule.
                This is larger because later steps can add more noise.

        """
        self.timesteps = timesteps
        self.device = torch.device(device)

       
        # 1. Create the beta schedule
        # ------------------------------------------------------------------
        # beta_t controls how much noise is added at each timestep.
        # We use a simple linear schedule from beta_start to beta_end.
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # alpha_t is the amount of signal retained at each step.
        self.alphas = 1.0 - self.betas

        # alpha_bar_t (cumulative product of alphas)
        # tells us how much of the original image survives after repeatedly
        # applying  forward process up to timestep t.
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        # We store alpha_cumprod from the previous timestep.
        # For t = 0, we define alpha_cumprod_prev = 1.0
        # because before any noise is added, the image is still perfectly clean.
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
        # In DDPM, the reverse process uses a Gaussian with:
        #   mean = model-dependent term
        #   variance = posterior_variance_t
        #  Posterior variance is:
        #   beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        #
        # This is used when sampling x_{t-1} from x_t.
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