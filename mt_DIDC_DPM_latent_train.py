import os
import sys
import logging
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL, get_cosine_schedule_with_warmup

# Import custom (assumendo che esistano nel tuo environment)
from datasets import FastDatasetDIDC
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS
from utils import set_reproducibility, sanitize_config, multiclass_dice_loss

logger = get_logger(__name__, log_level="INFO")

@dataclass
class TrainingConfig:
    run_name: str = "LDM_fg"
    data_path: str = "./DIDC_multiclass_coro_v2_prep"
    exp_dir: str = "./experiments/DIDCV2/diffusion"
    vae_pretrained_path: str = "./experiments/DIDC_VAE/20260302_1211_VAE_KL_train_no_adv/checkpoint_epoch" 
    vae_fg_pretrained_path: str = "./experiments/DIDC_VAE/20260304_0905_VAE_KL_train_fg/checkpoint_epoch" # Optional separate VAE for foreground masks, if None the same VAE will be used for both images and masks

    val_fraction: float = 0.2
    num_input_classes: int = 22
        
    # UNet Config
    unet_down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    unet_up_block_types: tuple = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    unet_blocks_out_channels: tuple = (128, 256, 512, 512)
    unet_layers_per_block: int = 2

    # Training params
    train_batch_size: int = 32
    eval_batch_size: int = 4 
    batch_size_per_gpu: int = 32
    num_epochs: int = 100
    conditional_generation: bool = True
    num_train_timesteps: int = 1000
    num_sample_steps: int = 150
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 250

    # Logging & Saving
    save_image_epochs: int = 5
    save_model_epochs: int = 1
    num_workers: int = 8
    mixed_precision: str = "fp16" 
    notes: str = "Latent Diffusion model conditional training with custom VAE and custom VAE for foreground masks"
    seed: int = 187
    
    gradient_accumulation_steps: int = 1
    run_dir: str = ""

    def __post_init__(self):
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.gradient_accumulation_steps = max(1, self.train_batch_size // (self.batch_size_per_gpu * num_gpus))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_dir = os.path.join(self.exp_dir, f"{timestamp}_{self.run_name}")

class LatentDiffusionTrainer:
    def __init__(self, config: TrainingConfig, model, vae, noise_scheduler, optimizer, lr_scheduler, train_loader, val_loader, accelerator, new_labels, vae_fg=None):
        self.config = config
        self.model = model
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.scale_factor = vae.config.scaling_factor
        self.new_labels = new_labels
        self.latent_channels = vae.config.latent_channels
        self.vae_fg = vae_fg

        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 'sample_dice_loss': [], 'lr': []
        }
        self.fixed_sample_batch = next(iter(self.val_loader))

    @torch.no_grad()
    def encode_to_latent(self, x, is_fg=False, use_mode=False):
        """Handles scaling and encoding of both clean images and foreground masks into the VAE latent space."""
        x = x * 2.0 - 1.0 # Pixel from [0,1] to [-1,1]
        
        if self.vae_fg and is_fg:
            posterior = self.vae_fg.encode(x).latent_dist
        else:
            posterior = self.vae.encode(x).latent_dist # Same VAE for both images and fg masks
        
        if use_mode or is_fg:
            z = posterior.mode()
        else:
            z = posterior.sample()
            
        return z * self.scale_factor

    def _preprocess_batch(self, batch):
        """Extracts pixels and foreground masks, prepares them fot eh UNet"""
        clean_images = batch['multiClassMask'] 
        fg_masks = batch['input_label'] 

        clean_images = F.one_hot(clean_images.long(), num_classes=self.config.num_input_classes).permute(0, 3, 1, 2).float() 
        fg_masks = fg_masks.float() 

        # Map original foreground masks to the multi tissue label space expected by the VAE
        B, _, H, W = fg_masks.shape        
        mapped_fg_masks = torch.zeros(
            (B, self.config.num_input_classes, H, W), 
            device=fg_masks.device, 
            dtype=fg_masks.dtype
        )

        if not self.vae_fg:
            fg_indices = [self.new_labels.index(label) for label in ['Background', 'LV_blood_pool', 'LV_Myocardium', 'RV_blood_pool_myocardium']]
            mapped_fg_masks[:, fg_indices, :, :] = fg_masks
        else:
            mapped_fg_masks = fg_masks

        # encoding
        z0 = self.encode_to_latent(clean_images, is_fg=False, use_mode=False)
        enc_fg = self.encode_to_latent(mapped_fg_masks, is_fg=True, use_mode=True)
        
        return z0, enc_fg

    def step(self, batch, is_training=True):
        z0, enc_fg = self._preprocess_batch(batch)
        batch_size = z0.size(0)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device).long()
        noise = torch.randn_like(z0)
        z_t = self.noise_scheduler.add_noise(z0, noise, timesteps)

        # Concat in the latent space
        try:
            net_input = torch.cat([z_t, enc_fg], dim=1) if self.config.conditional_generation else z_t
        except Exception as e:
            logger.error(f"Error during concatenation: z_t shape {z_t.shape}, enc_fg shape {enc_fg.shape}")
            raise e


        with torch.set_grad_enabled(is_training):
            noise_pred = self.model(net_input, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

        if is_training: 
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.lr_scheduler.step()

        return loss.detach().item()

    @torch.no_grad()
    def generate_and_log_samples(self, epoch):
        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # foreground encoding in latent space
        _, enc_fg = self._preprocess_batch(self.fixed_sample_batch)
        actual_batch_size = enc_fg.size(0)

        # Noise generation (latent shape)
        noise_shape = (actual_batch_size, self.latent_channels, enc_fg.shape[2], enc_fg.shape[3])
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
        z_t = torch.randn(noise_shape, generator=generator, device=self.accelerator.device)

        self.noise_scheduler.set_timesteps(self.config.num_sample_steps, device=self.accelerator.device)

        # Reverse diffusion process (sampling) with conditioning on the encoded foreground masks
        for t in tqdm(self.noise_scheduler.timesteps, desc=f"Sampling E{epoch}", leave=False, disable=not self.accelerator.is_local_main_process):
            net_input = torch.cat([z_t, enc_fg], dim=1) if self.config.conditional_generation else z_t 
            noise_pred = unwrapped_model(net_input, t).sample
            z_t = self.noise_scheduler.step(noise_pred, t, z_t).prev_sample 

        # Decoding step
        z_0_rescaled = z_t / self.scale_factor
        output_logits = self.vae.decode(z_0_rescaled).sample

        dice_loss = multiclass_dice_loss(output_logits, self.fixed_sample_batch['multiClassMask'].long())
        if isinstance(dice_loss, torch.Tensor): dice_loss = dice_loss.item()
        logger.info(f"Epoch {epoch} - Sample Dice Loss: {dice_loss:.4f}")

        # Tracking and image saving
        x_classes = torch.argmax(output_logits, dim=1) 
        tb_images = x_classes.unsqueeze(1).float() / (self.config.num_input_classes - 1) 

        for tracker in self.accelerator.trackers: 
            if tracker.name == "tensorboard":
                tracker.writer.add_images("generated_samples", tb_images, epoch)
        
        masks_np = x_classes.cpu().numpy()
        scale_factor = 255.0 / (self.config.num_input_classes - 1) if self.config.num_input_classes > 1 else 1.0
        masks_np_visual = (masks_np * scale_factor).astype(np.uint8)
        
        image_dir = os.path.join(self.config.run_dir, "samples")
        os.makedirs(image_dir, exist_ok=True)
        for i, mask_array in enumerate(masks_np_visual):
            img = Image.fromarray(mask_array, mode="L")
            img.save(os.path.join(image_dir, f"epoch_{epoch:04d}_sample_{i:02d}.png"))

        return dice_loss

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.config.run_dir, f"checkpoint_epoch")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_path)
        self.noise_scheduler.save_pretrained(checkpoint_path)

        with open(os.path.join(self.config.run_dir, f"metrics_history.json"), "w") as f:
            json.dump(self.metrics_history, f, indent=4)

    def train(self):
        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            avg_loss = 0.0
            progress_bar = tqdm(total=len(self.train_loader), disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch}")

            for step, batch in enumerate(self.train_loader):
                loss = self.step(batch, is_training=True) # Actual training step
                avg_loss += loss
                
                current_lr = self.lr_scheduler.get_last_lr()[0]
                logs = {"loss": loss, "lr": current_lr}
                self.accelerator.log(logs, step=global_step)

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
                global_step += 1

            avg_loss /= len(self.train_loader)
            self.metrics_history['train_loss'].append(avg_loss)
            self.metrics_history['lr'].append(self.lr_scheduler.get_last_lr()[0])
            progress_bar.close()

            if self.accelerator.is_main_process: 
                self.model.eval()
                val_loss = 0.0
                for val_step, val_batch in enumerate(self.val_loader):
                    val_loss += self.step(val_batch, is_training=False) # Actual validation step

                val_loss /= len(self.val_loader)
                self.metrics_history['val_loss'].append(val_loss)
                self.accelerator.log({"val_loss": val_loss}, step=global_step)
                
                logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

                if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                    dice_loss = self.generate_and_log_samples(epoch)
                    self.metrics_history['sample_dice_loss'].append(dice_loss)
                    self.accelerator.log({"sample_dice_loss": dice_loss}, step=global_step)

                if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
                    self.save_checkpoint(epoch)


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    config = TrainingConfig()
    set_reproducibility(config.seed)
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.run_dir,
    )
    
    if accelerator.is_main_process:
        setup_logger(config.run_dir)
        accelerator.init_trackers(f"tb_tracker_train", config=sanitize_config(asdict(config)))
    
    logger.info(f"Experiment directory: {config.run_dir}")

    logger.info("Loading pretrained VAE...")
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained_path).to(accelerator.device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    logger.info("VAE loaded and frozen.")

    if config.vae_fg_pretrained_path:
        logger.info("Loading separate pretrained VAE for foreground masks...")
        vae_fg = AutoencoderKL.from_pretrained(config.vae_fg_pretrained_path).to(accelerator.device)
        vae_fg.eval()
        for param in vae_fg.parameters():
            param.requires_grad = False
        logger.info("Foreground VAE loaded and frozen.")
    else:
        vae_fg = None
        logger.info("No separate VAE for foreground masks specified, using the same VAE for both images and masks.")


    dataset_config_path = os.path.join(config.data_path, "dataset_config.json")
    if os.path.exists(dataset_config_path):
        with open(dataset_config_path, "r") as f:
            dataset_config = json.load(f)
    else:
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

    all_files = sorted(list(set([
        f.replace('_fg.npy', '').replace('_mask.npy', '') 
        for f in os.listdir(config.data_path) if f.endswith('.npy')
    ])))
    train_files, val_files = train_test_split(all_files, test_size=config.val_fraction, random_state=config.seed)

    if accelerator.is_main_process:
        with open(f"{config.run_dir}/train_val_split.json", "w") as f:
            json.dump({'train_indices': train_files, 'val_indices': val_files}, f, indent=4)
        with open(f"{config.run_dir}/grouping_rules_and_labels.json", "w") as f:
            json.dump({'grouping_rules': GROUPING_RULES, 'new_labels': NEW_LABELS}, f, indent=4)
        with open(f"{config.run_dir}/training_config.json", "w") as f:
            json.dump({**asdict(config), **dataset_config}, f, indent=4)

    train_dataset = FastDatasetDIDC(config.data_path, file_list=train_files)
    val_dataset = FastDatasetDIDC(config.data_path, file_list=val_files)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size_per_gpu, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size_per_gpu, shuffle=False, num_workers=config.num_workers)

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="sigmoid")
    
    # parameters that depend on the VAE config
    in_channels = vae.config.latent_channels if config.conditional_generation else vae.config.latent_channels # Handle number of channels
    if config.vae_fg_pretrained_path:
        in_channels += vae_fg.config.latent_channels if config.conditional_generation else 0 # Add channels for foreground conditioning if using separate VAE
    out_channels = vae.config.latent_channels

    # UNet model for noise prediction
    model = UNet2DModel(
        sample_size=vae.config.sample_size, 
        in_channels=in_channels, 
        out_channels=out_channels,
        layers_per_block=config.unet_layers_per_block, 
        block_out_channels=config.unet_blocks_out_channels,
        down_block_types=config.unet_down_block_types,
        up_block_types=config.unet_up_block_types
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=(len(train_dataloader) * config.num_epochs))

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    trainer = LatentDiffusionTrainer(
        config=config,
        model=model,
        vae=vae,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        accelerator=accelerator,
        new_labels=dataset_config['new_labels_used'],
        vae_fg=vae_fg
    )
    
    logger.info("Start training LDM ... ")
    trainer.train()
    logger.info("End training without errors.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during training.")
        sys.exit(1)