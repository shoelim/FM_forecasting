import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Tuple

from model.layers import ResidualBlock, UpBlock
import numpy as np

import os
import scipy
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from torchvision.utils import make_grid, save_image
from skimage.metrics import structural_similarity as cal_ssim


def normalize(in_channels, **kwargs):
    if in_channels < 32:
        return nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x):
    return x*torch.sigmoid(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 64):  #32
        super(Encoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        residual_layers = []
        ch_mult = [1, 2, 4, 4]
        for i in range(len(ch_mult) - 1):
            in_ch = ch_mult[i] * mid_channels
            out_ch = ch_mult[i + 1] * mid_channels
            residual_layers.append(ResidualBlock(
                in_ch, out_ch, downsample_factor=2, norm_layer=normalize))
        self.residuals = nn.Sequential(*residual_layers)

        attn_ch = ch_mult[-1] * mid_channels
        self.pre_attn_residual = ResidualBlock(attn_ch, attn_ch, downsample_factor=1, norm_layer=normalize)
        self.attn_norm = normalize(attn_ch)
        self.attn = nn.MultiheadAttention(embed_dim=attn_ch, num_heads=1, batch_first=True)
        self.post_attn_residual = ResidualBlock(attn_ch, attn_ch, downsample_factor=1, norm_layer=normalize)

        self.out_norm = normalize(attn_ch)
        self.out_conv = nn.Conv2d(attn_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        :param images: [b, c, h, w]
        """

        x = self.conv_in(images)
        x = self.residuals(x)

        x = self.pre_attn_residual(x)
        z = self.attn_norm(x)
        h = z.size(2)
        z = rearrange(z, "b c h w -> b (h w) c")
        z, _ = self.attn(query=z, key=z, value=z)
        z = rearrange(z, "b (h w) c -> b c h w", h=h)
        x = x + z
        x = self.post_attn_residual(x)

        x = self.out_norm(x)
        x = swish(x)
        x = self.out_conv(x)

        return x
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 256):
        super(Decoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.pre_attn_residual = ResidualBlock(mid_channels, mid_channels, downsample_factor=1, norm_layer=normalize)
        self.attn_norm = normalize(mid_channels)
        self.attn = nn.MultiheadAttention(embed_dim=mid_channels, num_heads=1, batch_first=True)
        self.post_attn_residual = ResidualBlock(mid_channels, mid_channels, downsample_factor=1, norm_layer=normalize)

        residual_layers = []
        ch_div = [1, 2, 4, 4]
        
        for i in range(len(ch_div) - 1):
            in_ch = mid_channels // ch_div[i]
            out_ch = mid_channels // ch_div[i + 1]
            residual_layers.append(nn.Sequential(
                ResidualBlock(in_ch, out_ch, downsample_factor=1, norm_layer=normalize),
                UpBlock(out_ch, out_ch, scale_factor=2, upscaling_mode="nearest")))
        self.residuals = nn.Sequential(*residual_layers)

        out_ch = mid_channels // ch_div[-1]
        self.out_norm = normalize(out_ch)
        self.out_conv = nn.Conv2d(out_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        :param images: [b, c, h, w]
        """

        x = self.conv_in(images)

        x = self.pre_attn_residual(x)
        z = self.attn_norm(x)
        h = z.size(2)
        z = rearrange(z, "b c h w -> b (h w) c")
        z, _ = self.attn(query=z, key=z, value=z)
        z = rearrange(z, "b (h w) c -> b c h w", h=h)
        x = x + z
        x = self.post_attn_residual(x)

        x = self.residuals(x)

        x = self.out_norm(x)
        x = swish(x)
        x = self.out_conv(x)
        
        return x #torch.tanh(x)


class Model_AE(nn.Module):
    def __init__(self, state_size=4, in_channels=1, out_channels=1, enc_mid_channels=64, dec_mid_channels=256):
        super(Model_AE, self).__init__()

        self.sigma = 0.0000001
        state_size = state_size 
        
        self.encoder = Encoder(in_channels=in_channels, out_channels=state_size, mid_channels=enc_mid_channels)
        self.decoder = Decoder(in_channels=state_size, out_channels=out_channels, mid_channels=dec_mid_channels)


    def forward(self, observations: torch.Tensor, option: int=1):
        """

        :param observations: [b, num_observations, num_channels, height, width]
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        assert num_observations > 2

        # Sample target snapshots and conditioning
        target_snapshots_indices = torch.randint(low=2, high=num_observations, size=[batch_size])
        target_snapshots = observations[torch.arange(batch_size), target_snapshots_indices]
        reference_snapshots_indices = target_snapshots_indices - 1
        reference_snapshots = observations[torch.arange(batch_size), reference_snapshots_indices]
        conditioning_snapshots_indices = torch.cat([torch.randint(low=0, high=s - 1, size=[1]) for s in target_snapshots_indices], dim=0)
        conditioning_snapshots = observations[torch.arange(batch_size), conditioning_snapshots_indices]   
        
        # Encoder
        # input snapshots: [b, num_channels, height, width]
        input_snapshots = torch.stack([target_snapshots, reference_snapshots, conditioning_snapshots], dim=1)
        flat_input_snapshots = rearrange(input_snapshots, "b n c h w -> (b n) c h w")
        flat_latents = self.encoder(flat_input_snapshots)
        
        # Decode latents so that we can train the AE
        reconstructed_flat_snapshots = self.decoder(flat_latents)
        reconstructed_snapshots = rearrange(reconstructed_flat_snapshots, "(b n) c h w -> b n c h w", n=3)

        return input_snapshots, reconstructed_snapshots

    

