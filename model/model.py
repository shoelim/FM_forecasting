import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torchdiffeq import odeint
from tqdm import tqdm

from typing import Tuple

from model.layers import ResidualBlock, UpBlock
from model.layers.position_encoding import build_position_encoding
import numpy as np

import os
import scipy
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import scipy.stats as stats


def normalize(in_channels, **kwargs):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x*torch.sigmoid(x)

def metric(pred, true):        
    min_val = min(pred.min(), true.min())
    max_val = max(pred.max(), true.max())
    data_range = max_val - min_val
            
    ssim, psnr = 0, 0
            
    # Iterate over batch and sequence length
    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            if pred.shape[2] == 1:
                # For 1-channel grayscale images, no need to swap axes or use multichannel
                ssim += cal_ssim(pred[b, f, 0], true[b, f, 0], data_range=data_range)
                psnr += PSNR(pred[b, f, 0], true[b, f, 0], data_range=data_range)
            else:
                # For multi-channel images (e.g., RGB), use channel_axis parameter
                # pred[b, f] and true[b, f] are assumed to have shape (B, L, C, H, W)
                ssim += cal_ssim(pred[b, f], true[b, f], data_range=data_range, channel_axis=0)
                psnr += PSNR(pred[b, f], true[b, f], data_range=data_range)
                
    # Average SSIM and PSNR over the batch and sequence
    ssim = ssim / (pred.shape[0] * pred.shape[1])
    psnr = psnr / (pred.shape[0] * pred.shape[1])
            
    return ssim, psnr


def timestamp_embedding(timesteps, dim, scale=200, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param scale: a premultiplier of timesteps
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: whether to repeat only the values in timesteps along the 2nd dim
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = scale * timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(scale * timesteps, 'b -> b d', d=dim)
    return embedding


class VectorFieldRegressor(nn.Module):
    def __init__(
            self,
            depth: int,
            mid_depth: int,
            state_size: int,
            state_res: Tuple[int, int],
            inner_dim: int,
            out_norm: str = "ln",
            reference: bool = True):
        super(VectorFieldRegressor, self).__init__()

        self.state_size = state_size
        self.state_height = state_res[0]
        self.state_width = state_res[1]
        self.inner_dim = inner_dim
        self.reference = reference

        self.position_encoding = build_position_encoding(self.inner_dim, position_embedding_name="learned")

        self.project_in = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            nn.Linear(3 * self.state_size if self.reference else 2 * self.state_size, self.inner_dim)
            
        )

        self.time_projection = nn.Sequential(
            nn.Linear(1, 256), 
            nn.ReLU(),
            nn.Linear(256, self.inner_dim)
        )

        def build_layer(d_model: int):
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4 * d_model,
                dropout=0.05,
                activation="gelu",
                norm_first=True,
                batch_first=True)

        self.in_blocks = nn.ModuleList()
        self.mid_blocks = nn.Sequential(*[build_layer(self.inner_dim) for _ in range(mid_depth)])
        self.out_blocks = nn.ModuleList()
        for i in range(depth):
            self.in_blocks.append(build_layer(self.inner_dim))
            self.out_blocks.append(nn.ModuleList([
                nn.Linear(2 * self.inner_dim, self.inner_dim),
                build_layer(self.inner_dim)]))

        if out_norm == "ln":
            self.project_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.GELU(),
                nn.LayerNorm(self.inner_dim),
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
        elif out_norm == "bn":
            self.project_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.GELU(),
                nn.BatchNorm2d(self.inner_dim),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
        else:
            raise NotImplementedError

    def forward(
            self,
            input_latents: torch.Tensor,
            reference_latents: torch.Tensor,
            conditioning_latents: torch.Tensor,
            index_distances: torch.Tensor,
            timestamps: torch.Tensor) -> torch.Tensor:
        """

        :param input_latents: [b, c, h, w]
        :param reference_latents: [b, c, h, w]
        :param conditioning_latents: [b, c, h, w]
        :param index_distances: [b]
        :param timestamps: [b]
        :return: [b, c, h, w]
        """

        # Fetch timestamp tokens
        t = timestamp_embedding(timestamps, dim=self.inner_dim).unsqueeze(1)
        
        # Calculate position embedding
        pos = self.position_encoding(input_latents)
        pos = rearrange(pos, "b c h w -> b (h w) c")

        # Calculate distance embeddings
        dist = self.time_projection(torch.log(index_distances).unsqueeze(1)).unsqueeze(1)
        
        # Build input tokens
        if self.reference:
            x = self.project_in(torch.cat([input_latents, reference_latents, conditioning_latents], dim=1))
        else:
            x = self.project_in(torch.cat([input_latents, conditioning_latents], dim=1))
        x = x + pos + dist
        x = torch.cat([t, x], dim=1)

        # Propagate through the main network
        hs = []
        for block in self.in_blocks:
            x = block(x)
            hs.append(x.clone())
        x = self.mid_blocks(x)
        for i, block in enumerate(self.out_blocks):
            x = block[1](block[0](torch.cat([hs[-i - 1], x], dim=-1)))

        # Project to output
        out = self.project_out(x[:, 1:])

        return out


class Model(nn.Module):
    def __init__(self, encoder=None, decoder=None, state_size=4, state_res=[8,8], ours_sigma=0.01, sigma_min = 0.001, sigma_sam=0.0):
        super(Model, self).__init__()

        self.sigma = ours_sigma 
        self.sigma_min = sigma_min
        self.sigma_sam = sigma_sam
        state_size = state_size # state size
        
        self.encoder = encoder
        self.decoder = decoder 

        self.vector_field_regressor = VectorFieldRegressor(
            state_size=state_size,
            state_res=state_res,
            inner_dim=512,
            depth=4,
            mid_depth=5,
            out_norm='bn',
            reference=True
        )

    def forward(self, observations: torch.Tensor, option: str):
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
        # input snapshots: [b, 3, num_channels, height, width]
        with torch.no_grad():
            self.encoder.eval()
            input_snapshots = torch.stack([target_snapshots, reference_snapshots, conditioning_snapshots], dim=1)
            flat_input_snapshots = rearrange(input_snapshots, "b n c h w -> (b n) c h w")
            flat_latents = self.encoder(flat_input_snapshots)
        
        # reshape flat latents so that latents are of dimension (b, 3, latent_channels, latent_res1, latent_res2) 
        latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=3)
        
        target_latents = latents[:, 0]
        reference_latents = latents[:, 1]
        conditioning_latents = latents[:, 2]
        
        # Sample input latents and calculate target vectors
        noise = torch.randn_like(target_latents).to(target_latents.dtype).to(target_latents.device)
        if option == 'river':
            river_sigma_min = 0.0000001
            timestamps = torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device)
            input_latents = (1 - (1 - river_sigma_min) * timestamps) * noise + timestamps * target_latents
            target_vectors = (target_latents - (1 - self.sigma) * input_latents) / (1 - (1 - self.sigma) * timestamps)
        elif option == 'ours':
            timestamps = torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device)
            sigma = self.sigma
            sigma_min = self.sigma_min
            interpolated_vectors = timestamps * target_latents + (1 - timestamps) * reference_latents
            input_latents = (torch.sqrt(timestamps * (1 - timestamps) * sigma**2 + sigma_min**2)) * noise + interpolated_vectors
            target_vectors = target_latents - reference_latents + 0.5 * (sigma**2) * (1 - 2 * timestamps) * (input_latents - interpolated_vectors) / (sigma**2 * timestamps * (1 - timestamps) + sigma_min**2)
        elif option == 'stoch_interp':
            eps = 0.001
            sigma = 0.1
            timestamps = eps + (1 - eps) * torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device) 
            input_latents = (torch.sqrt(timestamps) * (1 - timestamps)) * sigma * noise + torch.square(timestamps) * target_latents + (1 - timestamps) * reference_latents
            interpolated_vectors = torch.square(timestamps) * target_latents + (1 - timestamps) * reference_latents 
            target_vectors = 2 * timestamps * target_latents - reference_latents + (input_latents - interpolated_vectors) * (1/ (2 * timestamps) - 1/(1-timestamps)) 
        elif option == 'stoch_interp_2':
            eps = 0.001
            sigma = 0.1
            timestamps = eps + (1 - eps) * torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device) 
            input_latents = (torch.sqrt(timestamps) * (1 - timestamps)) * sigma * noise + timestamps * target_latents + (1 - timestamps) * reference_latents
            interpolated_vectors = timestamps * target_latents + (1 - timestamps) * reference_latents 
            target_vectors = target_latents - reference_latents + (input_latents - interpolated_vectors) * (1/ (2 * timestamps) - 1/(1-timestamps)) 
        elif option == 've_sde':
            eps = 0.00001
            sigma_min = 0.01 
            sigma_max = 0.1
            timestamps = torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device) * (1 - eps) 
            ct = sigma_min * torch.sqrt( torch.pow(sigma_max/sigma_min, 2*(1-timestamps)) - 1)
            input_latents = ct * noise + target_latents
            alpha = np.log(sigma_max/sigma_min) * torch.pow(sigma_max, 2*(1-timestamps)) / ( torch.pow(sigma_min, 2*(1-timestamps)) - torch.pow(sigma_max, 2*(1-timestamps) ))
            target_vectors = target_latents + alpha * (input_latents -  target_latents) 
        elif option == 'vp_sde':
            eps = 0.00001
            beta_min = 0.1
            beta_max = 20.0
            timestamps = torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device) * (1 - eps) 
            t_prime = beta_min + (1 - timestamps) * (beta_max - beta_min) 
            t = (1-timestamps) * beta_min + torch.pow(1-timestamps, 2) * (beta_max - beta_min) / 2 
            input_latents = target_latents * torch.exp(-0.5*t) + torch.sqrt(1-torch.exp(-t)) * noise 
            target_vectors = -0.5 * t_prime * (torch.exp(-t) * input_latents - torch.exp(-0.5*t) * target_latents) / (1-torch.exp(-t)) 
        else:
            raise ValueError('Invalid option for probability density path!')
       
        # Calculate time distances
        index_distances = (reference_snapshots_indices - conditioning_snapshots_indices).to(input_latents.device)

        # Predict vectors
        reconstructed_vectors = self.vector_field_regressor(
            input_latents=input_latents,
            reference_latents=reference_latents,
            conditioning_latents=conditioning_latents,
            index_distances=index_distances,
            timestamps=timestamps.squeeze(3).squeeze(2).squeeze(1))

       
        return target_vectors, reconstructed_vectors

    @torch.no_grad()
    def generate_snapshots(
            self,
            observations: torch.Tensor,
            num_condition_snapshots: int = None, 
            num_snapshots: int = None,
            steps: int = 100,
            warm_start: float = 0.0,
            past_horizon: int = -1,
            verbose: bool = False,
            solver: str = "rk4",
            option: str = "ours") -> torch.Tensor:
        """
        Generates num_snapshots snapshots conditioned on observations

        :param observations: [b, num_observations, num_channels, height, width]
        :param num_snapshots: number of snapshots to generate
        :param warm_start: part of the integration path to jump to
        :param steps: number of steps for sampling
        :param past_horizon: number of snapshots to condition on
        :param verbose: whether to display loading bar
        """

        # Encoder
        self.encoder.eval()
        flat_observations = rearrange(observations, "b n c h w -> (b n) c h w")
        flat_latents = self.encoder(flat_observations)
        latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=num_condition_snapshots)
        b, n, c, h, w = latents.shape
        if n == 1:
             latents = latents[:, [0, 0]]          
            
        # Generate future latents
        gen = tqdm(range(num_snapshots), desc="Generating snapshots", disable=not verbose, leave=False)
        for _ in gen:
            def f(t: torch.Tensor, y: torch.Tensor):
                lower_bound = 0 if past_horizon == -1 else min(0, latents.size(1) - past_horizon)
                higher_bound = latents.size(1) - 1

                # Sample conditioning and reference
                conditioning_latents_indices = torch.randint(low=lower_bound, high=higher_bound, size=[b])
                conditioning_latents = latents[torch.arange(b), conditioning_latents_indices]
                reference_latents = latents[:, -1]

                # Calculate index distances
                index_distances = (higher_bound - conditioning_latents_indices).to(y.device)

                # Calculate vectors
                return self.vector_field_regressor(
                    input_latents=y,
                    reference_latents=reference_latents,
                    conditioning_latents=conditioning_latents,
                    index_distances=index_distances,
                    timestamps=t * torch.ones(b).to(latents.device))

            # Initialize with noise
            noise = torch.randn([b, c, h, w]).to(latents.device)
            if option == 'river':
                y0 = (1 - (1 - self.sigma) * warm_start) * noise + warm_start * latents[:, -1]
            elif option == 'ours':
                y0 = latents[:, -1] + self.sigma_sam * noise 
            elif option == 'stoch_interp':
                y0 = latents[:, -1] 
            elif option == 'stoch_interp_2':
                y0 = latents[:, -1] 
            elif option == 'vp_sde' or option == 've_sde':
                y0 = noise
            else:
                raise ValueError('Invalid option for probability density path!')
                
            # Solve ODE
            next_latents = odeint(
                f,
                y0,
                t=torch.linspace(warm_start, 1, int((1 - warm_start) * steps)).to(y0.device),
                method=solver
            )[-1]
            
            latents = torch.cat([latents, next_latents.unsqueeze(1)], dim=1)

        # Close loading bar
        gen.close()

        if n == 1:
            latents = latents[:, 1:]

        # Decode to image space
        flat_latents = rearrange(latents, "b n c h w -> (b n) c h w")
        reconstructed_flat_images = self.decoder(flat_latents)
        reconstructed_observations = rearrange(reconstructed_flat_images, "(b n) c h w -> b n c h w", b=b)
        
        return reconstructed_observations

