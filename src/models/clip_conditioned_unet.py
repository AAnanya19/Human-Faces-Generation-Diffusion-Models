"""DDPM U-Net with CLIP text conditioning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
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
                [embedding, torch.zeros((embedding.size(0), 1), device=device)],
                dim=1,
            )

        return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        *,
        dropout: float = 0.0,
        groups: int = 8,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_term = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class SelfAttentionBlock(nn.Module):
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
        x = x.view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return attn_out + residual


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        *,
        num_heads: int = 4,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "encoder_hidden_states must have shape [batch, sequence, context_dim]"
            )

        b, c, h, w = x.shape
        residual = x

        query = self.norm(x).view(b, c, h * w).transpose(1, 2)
        context = encoder_hidden_states.to(device=query.device, dtype=query.dtype)

        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = ~text_attention_mask.to(query.device).bool()

        attn_out, _ = self.attn(
            query,
            context,
            context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return attn_out + residual


class DownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        context_dim: int,
        *,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
        use_cross_attention: bool,
        attention_heads: int,
        add_downsample: bool,
    ):
        super().__init__()
        blocks = []
        current_in = in_channels
        for _ in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    current_in,
                    out_channels,
                    time_dim,
                    dropout=dropout,
                )
            )
            current_in = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.attn = (
            SelfAttentionBlock(out_channels, num_heads=attention_heads)
            if use_attention
            else nn.Identity()
        )
        self.cross_attn = (
            CrossAttentionBlock(
                out_channels,
                context_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
            if use_cross_attention
            else None
        )
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if add_downsample
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, time_emb)
        x = self.attn(x)
        if self.cross_attn is not None:
            x = self.cross_attn(x, encoder_hidden_states, text_attention_mask)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        *,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
        attention_heads: int,
        add_upsample: bool,
    ):
        super().__init__()
        self.upsample = (
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if add_upsample
            else nn.Identity()
        )

        blocks = []
        current_in = out_channels + skip_channels
        for _ in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    current_in,
                    out_channels,
                    time_dim,
                    dropout=dropout,
                )
            )
            current_in = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.attn = (
            SelfAttentionBlock(out_channels, num_heads=attention_heads)
            if use_attention
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, time_emb)
        x = self.attn(x)
        return x


class CLIPConditionedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 256,
        context_dim: int = 768,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        attention_resolutions: tuple[int, ...] = (16, 8),
        cross_attention_resolutions: tuple[int, ...] = (16, 8),
        attention_heads: int = 4,
        image_size: int = 64,
    ):
        super().__init__()
        if len(channel_mults) < 2:
            raise ValueError("channel_mults must contain at least two entries")

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_channels * mult for mult in channel_mults]
        for channel in channels:
            if channel % attention_heads != 0:
                raise ValueError(
                    f"Channel count {channel} must be divisible by attention_heads={attention_heads}."
                )

        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        down_stages = []
        resolution = image_size
        in_ch = channels[0]
        for idx, out_ch in enumerate(channels):
            down_stages.append(
                DownStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_dim=time_dim,
                    context_dim=context_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    use_attention=resolution in attention_resolutions,
                    use_cross_attention=resolution in cross_attention_resolutions,
                    attention_heads=attention_heads,
                    add_downsample=idx < len(channels) - 1,
                )
            )
            in_ch = out_ch
            if idx < len(channels) - 1:
                resolution //= 2
        self.down_stages = nn.ModuleList(down_stages)

        bottleneck_channels = channels[-1]
        self.mid_block1 = ResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            time_dim,
            dropout=dropout,
        )
        self.mid_attn = (
            SelfAttentionBlock(bottleneck_channels, num_heads=attention_heads)
            if resolution in attention_resolutions
            else nn.Identity()
        )
        self.mid_cross_attn = (
            CrossAttentionBlock(
                bottleneck_channels,
                context_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
            if resolution in cross_attention_resolutions
            else None
        )
        self.mid_block2 = ResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            time_dim,
            dropout=dropout,
        )

        up_stages = []
        current_channels = bottleneck_channels
        current_resolution = resolution
        for idx in reversed(range(len(channels))):
            out_ch = channels[idx]
            up_stages.append(
                UpStage(
                    in_channels=current_channels,
                    skip_channels=channels[idx],
                    out_channels=out_ch,
                    time_dim=time_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    use_attention=current_resolution in attention_resolutions,
                    attention_heads=attention_heads,
                    add_upsample=idx < len(channels) - 1,
                )
            )
            current_channels = out_ch
            if idx < len(channels) - 1:
                current_resolution *= 2
        self.up_stages = nn.ModuleList(up_stages)

        self.final_norm = nn.GroupNorm(8, channels[0])
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps)

        x = self.init_conv(x)

        skips = []
        for stage in self.down_stages:
            x, skip = stage(x, time_emb, encoder_hidden_states, text_attention_mask)
            skips.append(skip)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        if self.mid_cross_attn is not None:
            x = self.mid_cross_attn(x, encoder_hidden_states, text_attention_mask)
        x = self.mid_block2(x, time_emb)

        for stage, skip in zip(self.up_stages, reversed(skips)):
            x = stage(x, skip, time_emb)

        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)
        return x


__all__ = ["CLIPConditionedUNet", "CrossAttentionBlock"]
