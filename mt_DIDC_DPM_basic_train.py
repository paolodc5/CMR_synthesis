import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator

import numpy as np
import json
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import Image

from diffusers import DDPMScheduler, UNet2DModel, get_cosine_schedule_with_warmup
from datasets import LazyDatasetDIDC
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS
from utils import set_reproducibility, sanitize_config, multiclass_dice_loss


@dataclass
class TrainingConfig:
    run_name: str = "DDPM_conditional_train"
    data_path: str = "./New_dictionary"
    num_workers: int = 8
    remap_nn: bool = True
    threshold_classes: int = 50
    min_blob_size: int = 10
    target_size: int = 128
    val_fraction: float = 0.2
    num_input_classes: int = 22
    num_fg_channels: int = 4
    unet_layers_per_block: int = 2
    unet_down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    unet_up_block_types: tuple = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    unet_blocks_out_channels: tuple = (128, 256, 512, 512)
    rm_black_slices: bool = True
    remap_nn: bool = True
    train_batch_size: int = 16
    eval_batch_size: int = 4  
    num_epochs: int = 170
    conditional_generation: bool = True
    num_train_timesteps: int = 1000
    num_sample_steps: int = 50
    batch_size_per_gpu: int = 8
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 250
    save_image_epochs: int = 10
    save_model_epochs: int = 8
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    notes: str = "Diffusion model first CONDITIONAL training more epochs, logging dice loss on generated samples"
    seed: int = 187
    
    gradient_accumulation_steps: int = 1
    exp_dir = './experiments/DIDC' 

    def __post_init__(self):
        self.gradient_accumulation_steps = max(
            1, 
            self.train_batch_size // (self.batch_size_per_gpu * self.num_gpus)
        )
        self.exp_dir = os.path.join(self.exp_dir, f"{datetime.now().strftime('%Y%m%d_%H%M')}_{self.run_name}")
        self.fg_channels = self.num_fg_channels if self.conditional_generation else 0

def train_val_step(batch, model, num_classes, noise_scheduler, accelerator, lr_scheduler=None, optimizer=None, is_training=True, conditional_generation=False):
    assert not (is_training and optimizer is None), "Optimizer must be provided for training step"
    assert not (not is_training and optimizer is not None), "Optimizer should not be provided for validation step"

    clean_images = batch['multiClassMask']  # Shape: (B, C, H, W)
    fg_masks = batch['input_label'] 

    clean_images = F.one_hot(clean_images.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)
    clean_images = clean_images * 2.0 - 1.0  # Scale to [-1, 1]
    fg_masks = fg_masks.float() * 2.0 - 1.0 

    batch_size = clean_images.size(0)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device).long()

    noise = torch.randn_like(clean_images.float())
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

    if conditional_generation:
        net_input = torch.cat([noisy_images, fg_masks], dim=1)
    else:
        net_input = noisy_images

    with torch.set_grad_enabled(is_training):
        noise_pred = model(net_input, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

    if is_training: 
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

    return loss.detach().item()

def sample_and_save_images(model, noise_scheduler, config, epoch, num_classes, accelerator, val_batch):
    """
    Generates images from pure noise.
    """
    model.eval()

    # foreground
    fg_masks = val_batch['input_label'].float() * 2.0 - 1.0
    actual_batch_size = config.batch_size_per_gpu

    # Generate noise tensor 
    noise_shape = (actual_batch_size, num_classes, config.target_size, config.target_size)
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
    x = torch.randn(noise_shape, generator=generator, device=accelerator.device)

    # Set the noise scheduler timesteps for sampling
    noise_scheduler.set_timesteps(config.num_sample_steps, device=accelerator.device)

    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps, desc=f"Epoch {epoch} Sampling", leave=False, file=sys.__stderr__):
            net_input = torch.cat([x, fg_masks], dim=1) if config.conditional_generation else x # conditional generation input if enabled
            noise_pred = model(net_input, t).sample
            x = noise_scheduler.step(noise_pred, t, x).prev_sample # Standard DDPM step, scheduler subtracts the predicted noise

    dice_loss = multiclass_dice_loss(x, val_batch['multiClassMask'].long())
    if isinstance(dice_loss, torch.Tensor): dice_loss = dice_loss.item()
    print(f"Epoch {epoch} - Sample Dice Loss: {dice_loss:.4f}")

    x_classes = torch.argmax(x, dim=1) # Shape becomes (Batch, H, W)

    # Save images to TensorBoard
    tb_images = x_classes.unsqueeze(1).float() / (num_classes - 1) # Now between 0 and 1 for TensorBoard visualization
    for tracker in accelerator.trackers: # direct access to TensorBoard SummaryWriter
        if tracker.name == "tensorboard":
            tracker.writer.add_images("generated_samples", tb_images, epoch)
    
    # Save images to disk (adapted for multi-class masks)
    masks_np = x_classes.cpu().numpy()
    scale_factor = 255.0 / (num_classes - 1) if num_classes > 1 else 1.0
    masks_np_visual = (masks_np * scale_factor).astype(np.uint8)
    
    image_dir = os.path.join(config.exp_dir, "samples")
    os.makedirs(image_dir, exist_ok=True)
    
    for i, mask_array in enumerate(masks_np_visual):
        img = Image.fromarray(mask_array, mode="L")
        img.save(os.path.join(image_dir, f"epoch_{epoch:04d}_sample_{i:02d}.png"))

    return masks_np, dice_loss

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'sample_dice_loss': [],
        'lr': [],
    }
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.exp_dir),
    )

    config_dict = asdict(config)
    if accelerator.is_main_process:
        if config.exp_dir is not None:
            os.makedirs(config.exp_dir, exist_ok=True)
        accelerator.init_trackers("tb_tracker_train", config=sanitize_config(config_dict))

    # Prepare accelerator model
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    fixed_sample_batch = next(iter(val_dataloader))

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, file=sys.__stderr__)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        
        avg_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            loss = train_val_step(batch, 
                                  model, 
                                  config.num_input_classes, 
                                  noise_scheduler=noise_scheduler, 
                                  accelerator=accelerator, 
                                  lr_scheduler=lr_scheduler, 
                                  optimizer=optimizer, 
                                  conditional_generation=config.conditional_generation, 
                                  is_training=True
                                  )
            # WARNING: to be fixed, logging every epoch instead of every step, otherwise it becomes too verbose !!!!!!!!!!!!!!!
            avg_loss += loss
            logs = {"loss": loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_step += 1

        # save metrics history for this epoch
        avg_loss /= len(train_dataloader)
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['lr'].append(lr_scheduler.get_last_lr()[0])
        metrics_history['current_epoch'] = epoch
            

        if accelerator.is_main_process:
            # validation
            model.eval()
            
            val_loss = 0.0
            for val_batch in val_dataloader:
                val_loss += train_val_step(val_batch, 
                                           model, 
                                           config.num_input_classes, 
                                           noise_scheduler=noise_scheduler, 
                                           accelerator=accelerator, 
                                           conditional_generation=config.conditional_generation, 
                                           is_training=False
                                           )
            val_loss /= len(val_dataloader)
            metrics_history['val_loss'].append(val_loss)
            accelerator.log({"val_loss": val_loss}, step=global_step)
            print(f"Epoch {epoch} - Training loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
            unwrapped_model = accelerator.unwrap_model(model)

            # Sample and save images every few epochs
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                _, dice_loss = sample_and_save_images(unwrapped_model, noise_scheduler, config, epoch, config.num_input_classes, accelerator, fixed_sample_batch)
                metrics_history['sample_dice_loss'].append(dice_loss)
                accelerator.log({"sample_dice_loss": dice_loss}, step=global_step)

            # Save model checkpoint every few epochs
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                checkpoint_path = os.path.join(config.exp_dir, f"checkpoint_epoch")
                
                unwrapped_model.save_pretrained(checkpoint_path)
                noise_scheduler.save_pretrained(checkpoint_path)

                with open(os.path.join(config.exp_dir, f"metrics_history.json"), "w") as f:
                    json.dump(metrics_history, f, indent=4)

def main():
    config = TrainingConfig()
    
    set_reproducibility(config.seed)
    os.makedirs(config.exp_dir, exist_ok=True)

    log_file = open(f"{config.exp_dir}/log.txt", "w")
    sys.stdout = log_file
    sys.stderr = log_file
    print('Experiment directory:', config.exp_dir)
    
    all_files = sorted([f for f in os.listdir(config.data_path) if f.endswith('.npy')])
    train_files, val_files = train_test_split(all_files, test_size=config.val_fraction, random_state=config.seed)

    split_info = {
        'train_indices': train_files,
        'val_indices': val_files
    }

    with open(f"{config.exp_dir}/train_val_split.json", "w") as f:
        json.dump(split_info, f, indent=4)

    with open (f"{config.exp_dir}/grouping_rules_and_labels.json", "w") as f:
        json.dump({
            'grouping_rules': GROUPING_RULES,
            'new_labels': NEW_LABELS
        }, f, indent=4)

    with open (f"{config.exp_dir}/training_config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)
    
    train_dataset = LazyDatasetDIDC(config.data_path, 
                                    GROUPING_RULES, 
                                    NEW_LABELS, 
                                    target_size=(config.target_size, config.target_size), 
                                    num_input_classes=config.num_fg_channels, 
                                    rm_black_slices=config.rm_black_slices, 
                                    remap_nn=config.remap_nn, 
                                    threshold_classes=config.threshold_classes, 
                                    min_blob_size=config.min_blob_size,
                                    file_list=train_files)
    
    val_dataset =  LazyDatasetDIDC(config.data_path, 
                                   GROUPING_RULES, 
                                   NEW_LABELS, 
                                   target_size=(config.target_size, config.target_size), 
                                   num_input_classes=config.num_fg_channels, 
                                   rm_black_slices=config.rm_black_slices, 
                                   remap_nn=config.remap_nn, 
                                   threshold_classes=config.threshold_classes, 
                                   min_blob_size=config.min_blob_size,
                                   file_list=val_files)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size_per_gpu, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size_per_gpu, shuffle=False, num_workers=config.num_workers)



    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="sigmoid")
    
    in_channels = config.num_input_classes + config.num_fg_channels if config.conditional_generation else config.num_input_classes
    model = UNet2DModel(sample_size=config.target_size,  
                        in_channels=in_channels, 
                        out_channels=config.num_input_classes, 
                        layers_per_block=config.unet_layers_per_block, 
                        block_out_channels=config.unet_blocks_out_channels,
                        down_block_types=config.unet_down_block_types,
                        up_block_types=config.unet_up_block_types
                        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=(len(train_dataloader) * config.num_epochs))

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    log_file.close()

if __name__ == '__main__':
    main()
