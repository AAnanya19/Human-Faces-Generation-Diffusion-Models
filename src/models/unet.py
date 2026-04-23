# models/unet.py
"""
DDPM U-Net (simple implementation)

This file:
- Defines the denoising model used in diffusion training
- Takes a noisy image x_t and timestep t as input
- Predicts the noise epsilon_theta(x_t, t)

Reference:
- The overall design follows the standard encoder-decoder U-Net idea
  used in diffusion models and image-to-image architectures
- The timestep conditioning follows the common DDPM approach of
  embedding t and injecting that information into convolution blocks
- This implementation is written from scratch in a smaller and simpler
  form for the butterfly task

Note:
- This is a compact U-Net for unconditional image generation
- It is designed for a small toy task 
- It uses:
    1. sinusoidal timestep embeddings
    2. convolution blocks with time conditioning
    3. downsampling and upsampling with skip connections

Model summary:
Input:
    x_t  -> noisy image tensor of shape [B, C, H, W]
    t    -> timestep tensor of shape [B]

Output:
    predicted noise of shape [B, C, H, W]

This implementation assumes:
- the scheduler handles the diffusion equations
- the training loop compares predicted noise with the true noise
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Creates sinusoidal embeddings for diffusion timesteps.

    This is similar in spirit to positional encodings used in transformers,
    but here it is used to tell the network how much noise is present
    in the image at timestep t.

    Input:
        timesteps: [B]

    Output:
        embedding: [B, embedding_dim]
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert integer timesteps into sinusoidal embeddings.
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2

        # Build the frequency scales used for sine/cosine encoding
        exponent = -math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * exponent)

        # Expand timesteps so each one is multiplied by all frequencies
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

        # Concatenate sine and cosine parts
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        # If embedding_dim is odd, pad one extra dimension
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros((embedding.size(0), 1), device=device)], dim=1
            )

        return embedding


class ConvBlock(nn.Module):
    """
    A basic convolution block with timestep conditioning.

    Structure:
        Conv -> GroupNorm -> SiLU
        + projected time embedding
        Conv -> GroupNorm -> SiLU

    Why this block exists:
    - image features are processed through convolutions
    - timestep information is projected and added to the feature maps
    - this allows the model to adapt its behaviour depending on t
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act1 = nn.SiLU()

        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.SiLU()

        # If the number of channels changes, use a 1x1 projection for residual path
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one conditioned convolution block.

        Args:
            x:
                Input feature map of shape [B, C, H, W]

            time_emb:
                Timestep embedding of shape [B, time_dim]

        Returns:
            Output feature map of shape [B, out_channels, H, W]
        """
        residual = self.residual_conv(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        # Project timestep embedding and add it to the feature map
        # Reshape from [B, C] to [B, C, 1, 1] so it can broadcast spatially
        time_term = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + residual


class DownBlock(nn.Module):
    """
    One downsampling stage of the U-Net.

    This block:
    - processes features with two conv blocks
    - returns a skip connection
    - downsamples the feature map for the next stage
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()

        self.block1 = ConvBlock(in_channels, out_channels, time_dim)
        self.block2 = ConvBlock(out_channels, out_channels, time_dim)

        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        """
        Returns:
            downsampled_features:
                Feature map after spatial downsampling

            skip:
                Feature map used later in the decoder path
        """
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    One upsampling stage of the U-Net.

    This block:
    - upsamples the lower-resolution feature map
    - concatenates the matching skip connection from the encoder
    - processes the combined features with two conv blocks
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

        self.block1 = ConvBlock(out_channels + skip_channels, out_channels, time_dim)
        self.block2 = ConvBlock(out_channels, out_channels, time_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Current decoder feature map

            skip:
                Matching encoder feature map

            time_emb:
                Timestep embedding

        Returns:
            Refined upsampled feature map
        """
        x = self.upsample(x)

        # Concatenate skip connection along the channel dimension
        x = torch.cat([x, skip], dim=1)

        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        return x


class UNet(nn.Module):
    """
    A compact U-Net for DDPM noise prediction.

    Architecture:
    - input projection
    - timestep embedding
    - two downsampling stages
    - bottleneck
    - two upsampling stages
    - output projection back to image channels

    This version is intentionally small and suitable for:
    - butterfly toy task
    - 64x64 or 128x128 images
    - initial from-scratch DDPM experiments
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 256,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Timestep embedding
        # ------------------------------------------------------------------
        # First create sinusoidal embeddings, then pass them through
        # a small MLP so the model can learn a more useful time representation.
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ------------------------------------------------------------------
        # 2. Initial image projection
        # ------------------------------------------------------------------
        # Project the RGB image into the first feature space.
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ------------------------------------------------------------------
        # 3. Encoder / downsampling path
        # ------------------------------------------------------------------
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)     # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim) # 128 -> 256

        # ------------------------------------------------------------------
        # 4. Bottleneck
        # ------------------------------------------------------------------
        self.mid_block1 = ConvBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_block2 = ConvBlock(base_channels * 4, base_channels * 4, time_dim)

        # ------------------------------------------------------------------
        # 5. Decoder / upsampling path
        # ------------------------------------------------------------------
        self.up1 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 2,
            time_dim=time_dim,
        )

        self.up2 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels * 2,
            out_channels=base_channels,
            time_dim=time_dim,
        )

        # ------------------------------------------------------------------
        # 6. Final output projection
        # ------------------------------------------------------------------
        # Map decoder features back to the same number of channels as the input image.
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion U-Net.

        Args:
            x:
                Noisy image tensor x_t of shape [B, C, H, W]

            timesteps:
                Timestep tensor of shape [B]

        Returns:
            Predicted noise tensor of shape [B, C, H, W]
        """
        # ------------------------------------------------------------------
        # 1. Compute timestep embeddings
        # ------------------------------------------------------------------
        time_emb = self.time_embedding(timesteps)

        # ------------------------------------------------------------------
        # 2. Initial projection
        # ------------------------------------------------------------------
        x = self.init_conv(x)

        # ------------------------------------------------------------------
        # 3. Encoder path
        # ------------------------------------------------------------------
        x, skip1 = self.down1(x, time_emb)
        x, skip2 = self.down2(x, time_emb)

        # ------------------------------------------------------------------
        # 4. Bottleneck
        # ------------------------------------------------------------------
        x = self.mid_block1(x, time_emb)
        x = self.mid_block2(x, time_emb)

        # ------------------------------------------------------------------
        # 5. Decoder path
        # ------------------------------------------------------------------
        x = self.up1(x, skip2, time_emb)
        x = self.up2(x, skip1, time_emb)

        # ------------------------------------------------------------------
        # 6. Final output
        # ------------------------------------------------------------------
        x = self.final_conv(x)
        return x