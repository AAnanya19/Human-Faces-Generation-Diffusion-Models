# models/unet.py
"""
DDPM U-Net (improved implementation)

This file:
- Defines the denoising model used in diffusion training
- Takes a noisy image x_t and timestep t as input
- Predicts the noise epsilon_theta(x_t, t)

What this model consists of:
1. A sinusoidal timestep embedding
   - tells the network how noisy the current image is

2. Residual convolution blocks
   - more stable than plain stacked convolutions
   - help feature reuse and make deeper models easier to train

3. Downsampling and upsampling stages
   - allow the network to learn both local details and larger structure

4. Skip connections
   - preserve fine spatial detail from earlier layers
   - core part of the U-Net idea

5. Self attention in the lower resolution part of the network
   - helps the model reason over larger spatial relationships



References:
- Ho et al., "Denoising Diffusion Probabilistic Models", 2020
- The general U-Net structure follows the standard encoder decoder design
- The use of timestep conditioning and attention is inspired by common
  DDPM implementations and the Hugging Face diffusion tutorial,
  but this implementation is written from scratch for this project

Model summary:
Input:
    x_t -> noisy image tensor of shape [B, C, H, W]
    t   -> timestep tensor of shape [B]

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

    This gives the network a continuous representation of timestep t.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.embedding_dim // 2

        exponent = -math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * exponent)

        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros((embedding.size(0), 1), device=device)], dim=1
            )

        return embedding


class ResidualBlock(nn.Module):
    """
    Residual convolution block with timestep conditioning.

    Structure:
        GroupNorm -> SiLU -> Conv
        add projected time embedding
        GroupNorm -> SiLU -> Conv
        + residual connection

    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add timestep information after the first convolution.
        # This lets the block adapt its feature processing based on t.
        time_term = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + residual


class SelfAttentionBlock(nn.Module):
    """
    Lightweight self-attention block for 2D feature maps.

    
    - convolution is very good at local patterns
    - attention helps connect information across distant spatial regions
    - useful for global butterfly structure and symmetry
    """

    def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]

        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)

        return attn_out + residual


class DownBlock(nn.Module):
    """
    One encoder stage.

    This block:
    - applies two residual blocks
    - optionally applies attention
    - stores a skip connection
    - downsamples the feature map
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        use_attention: bool = False,
    ):
        super().__init__()

        self.block1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.block2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.attn = SelfAttentionBlock(out_channels) if use_attention else nn.Identity()

        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        x = self.attn(x)

        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    One decoder stage.

    This block:
    - upsamples the lower resolution feature map
    - concatenates the corresponding skip connection
    - applies two residual blocks
    - optionally applies attention
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        use_attention: bool = False,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

        self.block1 = ResidualBlock(out_channels + skip_channels, out_channels, time_dim)
        self.block2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.attn = SelfAttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        x = self.attn(x)

        return x


class UNet(nn.Module):
    """
     U Net for DDPM noise prediction.

    Compared with the previoius implemtation this one :
    - uses a wider channel progression
    - uses residual blocks instead of plain conv blocks
    - includes attention at deeper stages

    
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 256,
    ):
        super().__init__()

        # Time embedding pipeline
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial projection from RGB image to feature space
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder path
        # 64 -> 128 -> 256 -> 256
        self.down1 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            time_dim=time_dim,
            use_attention=False,
        )

        self.down2 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            time_dim=time_dim,
            use_attention=False,
        )

        self.down3 = DownBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            time_dim=time_dim,
            use_attention=True,
        )

        # Bottleneck
        self.mid_block1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attn = SelfAttentionBlock(base_channels * 4)
        self.mid_block2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        # Decoder path
        self.up1 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 2,
            time_dim=time_dim,
            use_attention=True,
        )

        self.up2 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels * 2,
            out_channels=base_channels,
            time_dim=time_dim,
            use_attention=False,
        )

        self.up3 = UpBlock(
            in_channels=base_channels,
            skip_channels=base_channels,
            out_channels=base_channels,
            time_dim=time_dim,
            use_attention=False,
        )

        # Final projection back to RGB noise prediction
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
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
        # Encode timestep information
        time_emb = self.time_embedding(timesteps)

        # Initial image projection
        x = self.init_conv(x)

        # Encoder
        x, skip1 = self.down1(x, time_emb)
        x, skip2 = self.down2(x, time_emb)
        x, skip3 = self.down3(x, time_emb)

        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # Decoder
        x = self.up1(x, skip3, time_emb)
        x = self.up2(x, skip2, time_emb)
        x = self.up3(x, skip1, time_emb)

        # Final prediction
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return x