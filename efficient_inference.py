"""DRCT (Dense Residual Connected Transformer) efficient inference script.

This module provides optimized inference for DRCT super-resolution models with:
- Memory format optimization (channels_last)
- Mixed precision support (fp32/fp16/bf16)
- Batched tile processing for large images
- CUDA stream overlap for improved throughput
- Efficient padding strategies
"""

import argparse
import glob
import json
import logging as pylog
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch._dynamo as dynamo
import torch._logging as tlog
import torch.cuda.streams
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor
from tqdm.auto import tqdm

# Configure logging for better debugging
os.environ.setdefault("TORCH_LOGS", "+dynamo,graph_breaks,guards,recompiles,inductor,output_code,cudagraphs")
warnings.simplefilter("default")
pylog.basicConfig(level=pylog.INFO, format="%(levelname)s %(name)s: %(message)s")

# Make compilation failures visible
dynamo.config.suppress_errors = False
dynamo.config.verbose = True

# Enable torch logging if available
try:
    tlog.set_logs(
        dynamo="debug",
        graph_breaks=True,
        guards=True,
        recompiles=True,
        inductor="debug",
        cudagraphs=True,
        output_code=True,
    )
except Exception:
    pass

# Write debug artifacts
try:
    inductor.debug = True
    inductor.print_kernel_source = True
    inductor.debug_dir = "torch_compile_debug"
except Exception:
    pass

#############################################
#           Model Utility Functions         #
#############################################


def log(msg):
    """Log message through tqdm to avoid output conflicts."""
    try:
        tqdm.write(str(msg))
    except Exception:
        print(msg, flush=True)


def audit_buffers(model):
    """Check for CPU buffers that would break compilation."""
    bad = [(n, b.device, b.dtype) for n, b in model.named_buffers() if not b.is_cuda]
    if bad:
        raise RuntimeError(f"CPU buffers found: {bad}")


def _dry_run_compile(model, device, tile, amp_dtype=torch.bfloat16, mode="reduce-overhead"):
    x = torch.zeros(1, 3, tile, tile, device=device).contiguous(memory_format=torch.channels_last)
    # list graph breaks, then force fullgraph capture
    try:
        from torch._dynamo import explain
        # Use the new API: explain(model)(x) instead of explain(model, x)
        explanation = explain(model)(x)
        print(f"Graph breaks: {explanation.graph_break_count}")
    except Exception as e:
        print(f"[dynamo.explain] Skipping analysis: {str(e)[:100]}")
    m = torch.compile(model, mode=mode, fullgraph=True, dynamic=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        _ = m(x)  # raises if any break remains


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.

    Args:
        x: Input tensor
        drop_prob: Probability of dropping path
        training: Whether in training mode

    Returns:
        Output tensor with paths potentially dropped
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: Optional[float] = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        """Initialize channel attention.

        Args:
            num_feat: Number of input features
            squeeze_factor: Channel reduction factor
        """
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    """Channel Attention Block with compression."""

    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        """Initialize CAB.

        Args:
            num_feat: Number of features
            compress_ratio: Feature compression ratio
            squeeze_factor: Channel attention squeeze factor
        """
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert single int to 2-tuple."""
    return (x, x) if isinstance(x, int) else x


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    """Initialize tensor with truncated normal distribution."""
    nn.init.trunc_normal_(tensor, std=std)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition image into windows.

    Args:
        x: Input tensor (B, H, W, C)
        window_size: Window size

    Returns:
        Windows tensor (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition.

    Args:
        windows: Windows tensor
        window_size: Window size
        H: Height
        W: Width

    Returns:
        Merged tensor (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#############################################
#           Core Model Modules              #
#############################################


class WindowAttention(nn.Module):
    """Window-based multi-head self attention module."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Initialize WindowAttention.

        Args:
            dim: Number of input channels
            window_size: Window size (height, width)
            num_heads: Number of attention heads
            qkv_bias: Whether to add bias to qkv projection
            qk_scale: Override default qk scale
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # tuple (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            .permute(2, 0, 1)
            .contiguous()
        )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """Initialize SwinTransformerBlock.

        Args:
            dim: Number of input channels
            input_resolution: Input resolution (H, W)
            num_heads: Number of attention heads
            window_size: Window size
            shift_size: Shift size for shifted window attention
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to add bias to qkv
            qk_scale: Override default qk scale
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            act_layer: Activation layer
            norm_layer: Normalization layer
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # Pre-calculate mask but don't register as buffer yet (will move to device later)
            attn_mask = None  # Will be created on-demand with correct device
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: Tuple[int, int], device=None) -> torch.Tensor:
        """Calculate attention mask for shifted window attention."""
        H, W = x_size
        # Create mask on the correct device from the start
        img_mask = torch.zeros((1, H, W, 1), device=device if device is not None else "cpu")
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            # Use cached mask if available and on correct device
            if self.attn_mask is not None:
                attn_windows = self.attn(x_windows, mask=self.attn_mask)
            elif self.shift_size > 0:
                # Create mask on demand with correct device
                if self.attn_mask is None:
                    self.attn_mask = self.calculate_mask(self.input_resolution, device=x.device)
                attn_windows = self.attn(x_windows, mask=self.attn_mask)
            else:
                attn_windows = self.attn(x_windows, mask=None)
        else:
            # Different size, calculate new mask
            if self.shift_size > 0:
                attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size, device=x.device))
            else:
                attn_windows = self.attn(x_windows, mask=None)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RDG(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        drop_path,
        norm_layer,
        gc,
        patch_size,
        img_size,
    ):
        super(RDG, self).__init__()

        self.swin1 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
        )
        self.adjust1 = nn.Conv2d(dim, gc, 1)

        self.swin2 = SwinTransformerBlock(
            dim + gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + gc) % num_heads),
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
        )
        self.adjust2 = nn.Conv2d(dim + gc, gc, 1)

        self.swin3 = SwinTransformerBlock(
            dim + 2 * gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 2 * gc) % num_heads),
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
        )
        self.adjust3 = nn.Conv2d(dim + gc * 2, gc, 1)

        self.swin4 = SwinTransformerBlock(
            dim + 3 * gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 3 * gc) % num_heads),
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=1,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
        )
        self.adjust4 = nn.Conv2d(dim + 3 * gc, gc, 1)

        self.swin5 = SwinTransformerBlock(
            dim + 4 * gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 4 * gc) % num_heads),
            window_size=window_size,
            shift_size=0,
            mlp_ratio=1,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
        )
        self.adjust5 = nn.Conv2d(dim + 4 * gc, dim, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pe = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.pue = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x: torch.Tensor, xsize: Tuple[int, int]) -> torch.Tensor:
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, xsize), xsize))))
        x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))))
        x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize))))
        x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize))))
        x5 = self.pe(self.adjust5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize)))
        return x5 * 0.2 + x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super(Upsample, self).__init__(*m)


class DRCT(nn.Module):
    """Dense Residual Connected Transformer for image super-resolution."""

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        gc=32,
        **kwargs,
    ):
        super(DRCT, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            # Register as buffer so it moves to device automatically
            self.register_buffer("mean", torch.Tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer("mean", torch.zeros(1, 1, 1, 1))
        self.upscale = upscale
        self.upsampler = upsampler

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # FIX: Add patch_unembed to match forward_features
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RDG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                num_heads=num_heads[i_layer],
                window_size=window_size,
                depth=0,
                shift_size=window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                gc=gc,
                img_size=img_size,
                patch_size=patch_size,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DRCT model.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Super-resolved image tensor
        """
        # No need to move mean as it's now a buffer that moves with the model
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x


#############################################
#         Inference & Main Script           #
#############################################

image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


def get_sorted_files_by_size(input_path: str, image_extensions: Tuple[str, ...]) -> List[str]:
    """
    Get a list of files sorted by file size (ascending).

    Args:
        input_path: Path to the directory containing images
        image_extensions: Tuple of image extensions to look for (e.g., ('*.jpg', '*.png'))

    Returns:
        List of file paths sorted by size (smallest to largest)
    """
    input_path_obj = Path(input_path)
    case_insensitive_patterns = [str(input_path_obj / ext.lower()) for ext in image_extensions] + [
        str(input_path_obj / ext.upper()) for ext in image_extensions
    ]

    input_files = []
    for pattern in case_insensitive_patterns:
        input_files.extend(glob.glob(pattern))
    input_files = list(dict.fromkeys(input_files))
    input_files = sorted(input_files, key=lambda x: Path(x).stat().st_size)
    return input_files


def check_ampere_gpu() -> None:
    """
    Check if the GPU supports NVIDIA Ampere or later and enable TF32 in PyTorch if it does.
    """
    if not torch.cuda.is_available():
        print("No GPU detected, running on CPU.")
        return
    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) supports NVIDIA Ampere or later, enabled TF32 in PyTorch."
            )
        else:
            gpu_name = torch.cuda.get_device_name(device)
            print(f"{gpu_name} (compute capability {major}.{minor}) does not support NVIDIA Ampere or later.")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.preferred_linalg_library = "cusolver"
    except Exception as e:
        print(f"Error occurred while checking GPU: {e}")


def test(img_lq: torch.Tensor, model: nn.Module, args: argparse.Namespace, window_size: int) -> torch.Tensor:
    """
    Perform inference on input image tensor.

    Args:
        img_lq: Input low-quality image tensor
        model: DRCT model instance
        args: Parsed command line arguments
        window_size: Size of processing window

    Returns:
        Output high-quality image tensor
    """
    if args.tile is None:
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16
            if args.precision == "bf16" and torch.cuda.is_available()
            else torch.float16
            if args.precision == "fp16" and torch.cuda.is_available()
            else torch.float32,
        ):
            output = model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

        coords = [(y, x) for y in h_idx_list for x in w_idx_list]
        weight = None  # Add Hann weight later if needed
        E = torch.zeros(b, c, h * sf, w * sf, device=img_lq.device, dtype=torch.float32)
        W = torch.zeros_like(E)

        # Create CUDA streams for overlap
        streams = (
            [torch.cuda.Stream() for _ in range(min(args.streams, len(coords)))]
            if args.streams > 1 and torch.cuda.is_available()
            else [torch.cuda.current_stream()]
        )
        stream_idx = 0

        for i in range(0, len(coords), args.tile_batch_size):
            ysxs = coords[i : i + args.tile_batch_size]
            batch = torch.cat([img_lq[..., y : y + tile, x : x + tile] for (y, x) in ysxs], dim=0)

            current_stream = streams[stream_idx % len(streams)]
            stream_idx += 1

            with torch.cuda.stream(current_stream):
                with torch.inference_mode(), torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16
                    if args.precision == "bf16" and torch.cuda.is_available()
                    else torch.float16
                    if args.precision == "fp16" and torch.cuda.is_available()
                    else torch.float32,
                ):
                    outs = model(batch)

                for j, (y, x) in enumerate(ysxs):
                    y0, y1 = y * sf, (y + tile) * sf
                    x0, x1 = x * sf, (x + tile) * sf
                    out = outs[j]
                    if weight is None:
                        E[..., y0:y1, x0:x1] += out
                        W[..., y0:y1, x0:x1] += 1.0
                    else:
                        E[..., y0:y1, x0:x1] += out * weight
                        W[..., y0:y1, x0:x1] += weight

        # Wait for all streams to complete
        if args.streams > 1 and torch.cuda.is_available():
            for stream in streams:
                stream.synchronize()
        output = E / W
    return output


def main() -> None:
    """
    Main function to run DRCT inference on input images.
    Parses command line arguments and processes all images in the input directory.
    """
    parser = argparse.ArgumentParser(
        description="DRCT Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="input test image folder")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="weights/net_g_latest.pth",
        help="path to model checkpoint",
    )
    parser.add_argument("--output", type=str, default=None, help="output folder")
    parser.add_argument("--scale", type=int, default=4, help="scale factor: 1, 2, 3, 4")
    parser.add_argument(
        "--tile",
        type=int,
        default=256,
        help="Tile size, -1 for no tile during inference (inference on whole image)",
    )
    parser.add_argument("--tile_overlap", type=int, default=16, help="Overlapping of different tiles")
    parser.add_argument(
        "--tile_batch_size",
        type=int,
        default=1,
        help="Mini-batch size for tile processing to reduce memory usage",
    )
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality (0-100)")
    parser.add_argument(
        "--compile",
        choices=["off", "default", "reduce", "max"],
        default="off",
        help="torch.compile mode: off, default, reduce-overhead, max-autotune (off by default)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default=None,
        help="Precision: fp32, fp16, bf16 (auto-detected if None)",
    )
    parser.add_argument("-skip", "--skip_completed", action="store_true", help="skip completed images")
    parser.add_argument(
        "--streams",
        type=int,
        default=1,
        help="Number of CUDA streams for H2D/compute overlap (default: 1)",
    )
    args = parser.parse_args()
    print(f"Running inference with args:\n{json.dumps(args.__dict__, indent=4)}")

    if args.tile < 1:
        args.tile = None

    check_ampere_gpu()

    # Set precision
    if args.precision is None:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere or later
                args.precision = "bf16"
            else:
                args.precision = "fp16"
        else:
            args.precision = "fp32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRCT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6] * 12,
        embed_dim=180,
        num_heads=[6] * 12,
        gc=32,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    # Load state dict with strict=False to handle missing buffer
    model.load_state_dict(checkpoint["params"], strict=False)
    model.eval()
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    if args.compile != "off":
        print(f"Compiling model with mode: {args.compile}")
        if args.compile == "default":
            mode = "default"
        elif args.compile == "reduce":
            mode = "reduce-overhead"
        else:  # max
            mode = "max-autotune"
        try:
            audit_buffers(model)  # fail early on stray CPU buffers

            # use the actual tile, not 64, to avoid recompiles and mask mismatches
            tile = int(args.tile) if getattr(args, "tile", None) else 64

            # dry run: fullgraph=True to surface the latest break loudly
            # Note: dry run can fail with fake OOM or other issues, so we make it optional
            if os.environ.get("SKIP_DRY_RUN", "0") != "1":
                try:
                    _dry_run_compile(model, device, tile, amp_dtype=torch.bfloat16, mode=mode)
                    print("Dry run succeeded - model is fully compilable!")
                except Exception as e:
                    print(f"Dry run failed (expected): {str(e)[:100]}...")
                    print("Proceeding with partial compilation...")

            # real compile for execution: keep shapes static for cudagraphs
            model = torch.compile(model, mode=mode, dynamic=False)

            # warmup on exact tile, channels_last, and same AMP dtype
            if args.input and Path(args.input).is_dir() and len(list(Path(args.input).iterdir())) > 1:
                print("Running warmup inference...")
                dummy = torch.zeros(1, 3, tile, tile, device=device).contiguous(memory_format=torch.channels_last)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(dummy)

        except Exception as e:
            print(f"Compilation failed: {e}, falling back to eager mode")

    window_size = 16

    out_dir = Path(args.output) if args.output is not None else Path(args.input) / "upscaled-DRCT-outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = get_sorted_files_by_size(args.input, image_extensions)
    for path in tqdm(input_files, desc="inference"):
        imgname = Path(path).stem
        out_path = out_dir / f"{imgname}.jpg"

        if args.skip_completed and out_path.exists():
            print(f"Skipping completed image: {out_path.name}")
            continue

        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().pin_memory()
            img = img.unsqueeze(0).contiguous(memory_format=torch.channels_last).to(device, non_blocking=True)

            _, _, h_old, w_old = img.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old

            # Use efficient padding with reflect mode
            img = F.pad(img, (0, w_pad, 0, h_pad), mode="reflect")

            output = test(img, model, args, window_size)
            output = output[..., : h_old * args.scale, : w_old * args.scale]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        except Exception as error:
            print("Error", error, imgname)
        else:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(str(out_path), output, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    
    # Save inference configuration
    config_path = out_dir / "inference_config.json"
    with open(config_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
    print(f"Inference configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
