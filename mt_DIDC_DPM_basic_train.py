import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

import torch
import torch.nn.functional as F
from torchinfo import summary
import torch.nn.functional as F

from accelerate import Accelerator

import cv2
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime

from diffusers import DDPMScheduler, UNet2DModel, get_cosine_schedule_with_warmup
from datasets import RAMDatasetDIDC, LazyDatasetDIDC
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS
from utils import set_reproducibility


@dataclass
class TrainingConfig:
    data_path: str = "./New_dictionary"
    num_workers: int = 8
    remap_nn: bool = True
    threshold_classes: int = 50
    min_blob_size: int = 10
    target_size: int = 128
    val_fraction: float = 0.2
    train_batch_size: int = 16
    eval_batch_size: int = 16  
    num_epochs: int = 50
    batch_size_per_gpu: int = 2
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    notes: str = "Diffusion model"
    seed: int = 187
    
    gradient_accumulation_steps: int = 1 

    def __post_init__(self):
        self.gradient_accumulation_steps = max(
            1, 
            self.train_batch_size // (self.batch_size_per_gpu * self.num_gpus)
        )

set_reproducibility(TrainingConfig.seed)
print(TrainingConfig.seed)


def training_step(batch, model, num_classes, optimizer, noise_scheduler, accelerator):
    model
    model.train()

    clean_images = batch['multiClassMask']  # Shape: (B, C, H, W)
    clean_images = F.one_hot(clean_images.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)
    clean_images = clean_images * 2.0 - 1.0  # Scale to [-1, 1]

    batch_size = clean_images.size(0)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device).long()

    noise = torch.randn_like(clean_images.float())
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    noise_pred = model(noisy_images, timesteps).sample
    loss = F.mse_loss(noise_pred, noise)

    with accelerator.accumulate(model):
        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.detach().item()


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare accelerator model
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            loss = training_step(batch, model, len(train_dataloader.dataset.new_labels), optimizer, noise_scheduler, accelerator)

            progress_bar.update(1)
            logs = {"loss": loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            # implement sampling pipeline

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # implement evaluate() to see how well your model is doing, and optionally save generated images
                pass

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                # implement saving logic
                pass



def main():
    config = TrainingConfig()
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="sigmoid")
    model = UNet2DModel(sample_size=384,  in_channels=22,   out_channels=22, layers_per_block=2,block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),)

    dataset = LazyDatasetDIDC(config.data_path, grouping_rules=GROUPING_RULES, new_labels=NEW_LABELS)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    model_summary = summary(model, input_data=(torch.randn(1, 22, 384, 384), torch.tensor([0])))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=(len(train_dataloader) * config.num_epochs))