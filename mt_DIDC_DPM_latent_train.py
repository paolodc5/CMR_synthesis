import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchinfo import summary

from accelerate import Accelerator

import numpy as np
import json
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import Image

from diffusers import DDPMScheduler, UNet2DModel, get_cosine_schedule_with_warmup
from datasets import FastDatasetDIDC
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS
from utils import set_reproducibility, sanitize_config, multiclass_dice_loss


@dataclass
class TrainingConfig:
    run_name: str = "DDPM_conditional_train"
    data_path: str = "./DIDC_multiclass_coro_v2_prep"
    exp_dir = './experiments/DIDCV2/diffusion' 

    target_size: int = 128
    val_fraction: float = 0.2
    num_input_classes: int = 22
    num_fg_channels: int = 4
    unet_layers_per_block: int = 2
    unet_down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    unet_up_block_types: tuple = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    unet_blocks_out_channels: tuple = (128, 256, 512, 512)

    train_batch_size: int = 16
    eval_batch_size: int = 4 
    batch_size_per_gpu: int = 8
 
    num_epochs: int = 100
    conditional_generation: bool = True
    num_train_timesteps: int = 1000
    num_sample_steps: int = 150
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 250

    save_image_epochs: int = 5
    save_model_epochs: int = 5
    
    num_workers: int = 8
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    notes: str = "Diffusion model first CONDITIONAL training more epochs, meant as benchmark"
    seed: int = 187
    
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        self.gradient_accumulation_steps = max(
            1, 
            self.train_batch_size // (self.batch_size_per_gpu * self.num_gpus)
        )
        self.exp_dir = os.path.join(self.exp_dir, f"{datetime.now().strftime('%Y%m%d_%H%M')}_{self.run_name}")
        self.fg_channels = self.num_fg_channels if self.conditional_generation else 0



class DDPM(torch.nn.Module):
    def __init__(self, unet, autoencoder, autoencoder_fg, noise_scheduler, config):
        super().__init__()
        self.unet = unet
        self.autoencoder = autoencoder
        self.autoencoder_fg = autoencoder_fg
        self.noise_scheduler = noise_scheduler
        self.config = config
        self.scale_factor = getattr(self.config, "scale_factor", 0.18215)  # default value from SD


        self.autoencoder.eval()  
        self.autoencoder_fg.eval()  
        for param in self.autoencoder.parameters():
            param.requires_grad = False  
        for param in self.autoencoder_fg.parameters():
            param.requires_grad = False  


    @torch.no_grad()
    def encode_to_latent(self, x, is_fg=False):
        if is_fg:
            latent = self.autoencoder_fg.encode(x)  
        else:
            latent = self.autoencoder.encode(x)
    
        return latent.latent_dist.sample()  # return the sample from the latent distribution        

    def forward(self, batch):
        # encode input mask
        input_mask = batch['multiClassMask']  # shape (B, H, W)
        input_mask = F.one_hot(input_mask, num_classes=self.config.num_input_classes).permute(0, 3, 1, 2).float()  # shape (B, C_in, H, W)
        z0 = self.encode_to_latent(input_mask)  # shape (B, C_encoded, H_enc, W_enc)        
        # scale z0 to match the expected input range of the noise scheduler (e.g., [-1, 1])
        z0 = z0 * 2 - 1  # assuming the autoencoder outputs in [0, 1], scale to [-1, 1]


        # encode the foreground 
        fg = batch['input_label']  # shape (B, C_fg, H, W)
        enc_fg = self.encode_to_latent(fg, is_fg=True)  # shape (B, C_encoded, H_enc, W_enc)
        
        # compute noise and add to the input mask
        B = z0.shape[0]
        timesteps = torch.randint(0, self.config.num_train_timesteps, (B,), device=z0.device).long()
        noise = torch.randn_like(z0)
        z_t = self.noise_scheduler.add_noise(z0, noise, timesteps)

        # concatenate the noisy input mask with the conditioning encoded foreground
        cond_input = torch.cat([z_t, enc_fg], dim=1)  # shape (B, 2*C_encoded, H_enc, W_enc)

        # forward pass through the UNet
        unet_output = self.unet(cond_input, timesteps)
        loss = F.mse_loss(unet_output.sample, noise)

        return loss


class Sampler:
    def __init__(self, model, noise_scheduler, config):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.config = config

    @torch.no_grad()
    def sample(self, conditioning_fg):
        B, C_fg, H, W = conditioning_fg.shape
        latent_shape = (B, self.config.unet_blocks_out_channels[0], H // 8, W // 8)  # assuming 3 downsamplings with factor 2 each
        z_t = torch.randn(latent_shape, device=conditioning_fg.device)  # start from pure noise

        for t in tqdm(reversed(range(self.config.num_sample_steps)), desc="Sampling"):
            timesteps = torch.full((B,), t * (self.config.num_train_timesteps // self.config.num_sample_steps), device=conditioning_fg.device, dtype=torch.long)
            cond_input = torch.cat([z_t, conditioning_fg], dim=1)  # shape (B, C_encoded*2, H_enc, W_enc)
            unet_output = self.model.unet(cond_input, timesteps)
            noise_pred = unet_output.sample

            z_t = self.noise_scheduler.step(noise_pred, t * (self.config.num_train_timesteps // self.config.num_sample_steps), z_t)

        return z_t

if __name__ == "__main__":
    # test train_val_step
    pass