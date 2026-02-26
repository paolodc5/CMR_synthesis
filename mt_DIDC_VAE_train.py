import os
import sys
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
import torchvision
from diffusers import AutoencoderKL, get_cosine_schedule_with_warmup
from datasets import FastDatasetDIDC
from gan_basic import DiscriminatorPatchGAN
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS
from utils import multiclass_dice_loss, set_reproducibility, sanitize_config, sanitize_config

@dataclass
class VAETrainingConfig:
    run_name: str = "VAE_KL_train"
    data_path: str = "./DIDC_multiclass_coro_v2_prep_old"
    num_workers: int = 8
    target_size: int = 384
    val_fraction: float = 0.2
    num_input_classes: int = 22
    train_batch_size: int = 16
    eval_batch_size: int = 4  
    batch_size_per_gpu: int = 4
    num_epochs: int = 170
    
    latent_channels: int = 4 # Compresses the input mask (22 channels) into a 4-channel latent space (bottleneck)
    kl_weight: float = 1e-5  # weight for KL divergence in the loss function
    layers_per_block: int = 2
    block_out_channels: tuple = (64, 128, 256)
    down_block_types: tuple = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")
    up_block_types: tuple = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")
    
    num_fg_channels: int = 4

    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 250
    save_image_epochs: int = 10
    save_model_epochs: int = 8
    mixed_precision: str = "fp16"
    seed: int = 187
    notes: str = "VAE with KL divergence loss, Targeting 4 foreground classes with remapping and thresholding. NEW dataset coroV2"
    gradient_accumulation_steps: int = 1
    exp_dir = './experiments/DIDC_VAE' 
    adv_train: bool = False # Whether to use adversarial training with a discriminator (PatchGAN) alongside the VAE. If True, discriminator and its optimizer will be initialized and used in the training loop.

    def __post_init__(self):
        self.gradient_accumulation_steps = max(1, self.train_batch_size // (self.batch_size_per_gpu * self.num_gpus))
        self.exp_dir = os.path.join(self.exp_dir, f"{datetime.now().strftime('%Y%m%d_%H%M')}_{self.run_name}")

def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max=0.75):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight

def train_val_step(batch, model, num_classes, accelerator, optimizer=None, lr_scheduler=None, is_training=True, kl_weight=1e-5):
    assert not (is_training and optimizer is None), "Optimizer must be provided for training step"

    target_classes = batch['multiClassMask'].long() # Shape: (B, H, W)
    if target_classes.dim() == 4:
        target_classes = target_classes.squeeze(1)

    clean_images = F.one_hot(target_classes, num_classes=num_classes).permute(0, 3, 1, 2).float() 
    clean_images = clean_images * 2.0 - 1.0 

    with torch.set_grad_enabled(is_training):
        posterior = model.encode(clean_images).latent_dist
        z = posterior.sample()
        reconstructed_logits = model.decode(z).sample

        recon_loss = F.cross_entropy(reconstructed_logits, target_classes) 
        dice_loss = multiclass_dice_loss(reconstructed_logits, target_classes)
        kl_loss = posterior.kl().mean()
        loss = recon_loss + kl_weight * kl_loss

    if is_training: # real training logic with gradient accumulation and optimizer step
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

    return loss.detach().item(), recon_loss.detach().item(), kl_loss.detach().item(), dice_loss.detach().item()

# Currently not used
def compute_perceptual_loss(discr_real, discr_fake, criterion_L1):
    """
    Function to compute perceptual loss between features from the generated image and features from the ground truth image.
    
    :param discr_real: List of tuples containing discriminator outputs and features for real images
    :param discr_fake: List of tuples containing discriminator outputs and features for fake images
    :param criterion_L1: Loss function to compute the perceptual loss (e.g., L1Loss or MSELoss)
    :return: Computed perceptual loss averaged across all feature map pairs
    """
    total_loss = 0.0
    for real_pred, fake_pred in zip(discr_real, discr_fake):
        real_feat = real_pred[1] # Get features from the discriminator's forward pass on real pairs
        fake_feat = fake_pred[1] # Get features from the discriminator's forward pass on fake pairs
        
        loss_discr = 0.0
        for rf, ff in zip(real_feat, fake_feat):
            loss_layer = criterion_L1(ff, rf.detach()) # Compute L1 loss between fake and real features
            loss_discr += loss_layer
            
        total_loss += loss_discr # just a sum not a mean
    return total_loss

def train_vae_adv_step(batch, model, discriminator, num_classes, accelerator, 
                       opt_vae, opt_disc, global_step, 
                       disc_start_step=10000, kl_weight=1e-6):
    
    target_classes = batch['multiClassMask'].long()
    if target_classes.dim() == 4: target_classes = target_classes.squeeze(1)

    clean_images = F.one_hot(target_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()
    clean_images = clean_images * 2.0 - 1.0

    posterior = model.encode(clean_images).latent_dist
    z = posterior.sample()
    reconstructed_logits = model.decode(z).sample

    recon_loss = F.cross_entropy(reconstructed_logits, target_classes)
    dice_loss = multiclass_dice_loss(reconstructed_logits, target_classes)
    kl_loss = posterior.kl().mean()
    
    real_features = discriminator(clean_images)
    fake_features = discriminator(reconstructed_logits)
    p_loss = compute_perceptual_loss(real_features, fake_features, torch.nn.L1Loss()) 
    
    v_loss = recon_loss + dice_loss + kl_weight * kl_loss + p_loss

    if global_step >= disc_start_step:
        logits_fake = discriminator(reconstructed_logits)
        g_loss = -torch.mean(logits_fake[0]) # Adversarial loss for generator (VAE decoder)

        last_layer = model.decoder.conv_out.weight
        
        v_loss_grads = torch.autograd.grad(v_loss, last_layer, retain_graph=True)[0]
        g_loss_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        
        d_weight = torch.norm(v_loss_grads) / (torch.norm(g_loss_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        
        total_vae_loss = v_loss + d_weight * g_loss
    else:
        d_weight = torch.tensor(0.0)
        g_loss = torch.tensor(0.0)
        total_vae_loss = v_loss

    opt_vae.zero_grad()
    accelerator.backward(total_vae_loss)
    opt_vae.step()


    if global_step >= disc_start_step:
        logits_real = discriminator(clean_images.contiguous().detach())
        logits_fake_d = discriminator(reconstructed_logits.contiguous().detach())

        # Hinge loss
        loss_real = torch.mean(F.relu(1. - logits_real[0]))
        loss_fake = torch.mean(F.relu(1. + logits_fake_d[0]))
        d_loss = 0.5 * (loss_real + loss_fake)

        opt_disc.zero_grad()
        accelerator.backward(d_loss)
        opt_disc.step()
    else:
        d_loss = torch.tensor(0.0)

    d_weight_value = d_weight.item() if isinstance(d_weight, torch.Tensor) else d_weight

    return total_vae_loss.item(), recon_loss.item(), kl_loss.item(), dice_loss.item(), d_loss.item(), d_weight_value

def sample_and_save_reconstructions(model, config, epoch, num_classes, accelerator, val_batch):
    """
    Pass a real batch for visualization
    """
    model.eval()
    
    gt_classes = val_batch['multiClassMask'].long()
    if gt_classes.dim() == 4:
        gt_classes = gt_classes.squeeze(1)
    
    clean_images_oh = F.one_hot(gt_classes, num_classes=num_classes).permute(0, 3, 1, 2).float() 
    clean_images_oh = (clean_images_oh * 2.0 - 1.0).to(accelerator.device)

    with torch.no_grad():
        posterior = model.encode(clean_images_oh).latent_dist
        z = posterior.mode() # At inference time use just the mean (mode) of the distribution for deterministic output
        reconstructed_logits = model.decode(z).sample
            
    # Tensors for tensorboard logging 
    reconstructed_classes_tensor = torch.argmax(reconstructed_logits, dim=1).cpu()
    gt_classes_tensor = gt_classes.cpu()

    gt_tb = gt_classes_tensor.unsqueeze(1).float() / (num_classes - 1)
    recon_tb = reconstructed_classes_tensor.unsqueeze(1).float() / (num_classes - 1)
    
    combined_tb = torch.cat([gt_tb, recon_tb], dim=3) # dim=3 to create a side-by-side image (B, 1, H, 2*W)
    grid = torchvision.utils.make_grid(combined_tb, nrow=2, padding=2, pad_value=1.0) 
    
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_image("Val_Reconstructions/GT_vs_Pred", grid, epoch)

    # Tensors to numpy to save as images
    reconstructed_classes = reconstructed_classes_tensor.numpy()
    gt_classes_np = gt_classes_tensor.numpy()
    image_dir = os.path.join(config.exp_dir, "samples")
    os.makedirs(image_dir, exist_ok=True)
    
    scale_factor = 255.0 / (num_classes - 1) if num_classes > 1 else 1.0
    
    for i in range(gt_classes_np.shape[0]):
        gt_vis = (gt_classes_np[i] * scale_factor).astype(np.uint8)
        recon_vis = (reconstructed_classes[i] * scale_factor).astype(np.uint8)
        combined_img = np.concatenate((gt_vis, recon_vis), axis=1) 
        img = Image.fromarray(combined_img, mode="L")
        img.save(os.path.join(image_dir, f"epoch_{epoch:04d}_sample_{i:02d}_(GT_vs_Recon).png"))

def train_loop(config, model, optimizer, train_dataloader, val_dataloader, lr_scheduler, adv_train=False, discriminator=None, opt_disc=None):
    assert not adv_train or (adv_train and discriminator is not None and opt_disc is not None), "Discriminator and its optimizer must be provided for adversarial training"
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.exp_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(config.exp_dir, exist_ok=True)
        accelerator.init_trackers("tb_tracker_vae", config=sanitize_config(asdict(config)))

    metrics_hystory = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_dice_loss': [],
        'train_d_loss': [],
        'train_d_weight': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_dice_loss': [],
        'val_d_loss': [],
        'val_d_weight': [],
    }

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    fixed_sample_batch = next(iter(val_dataloader))
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, file=sys.__stderr__)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        for step, batch in enumerate(train_dataloader):
            if adv_train:
                total_loss, recon_loss, kl_loss, dice_loss, d_loss, d_weight = train_vae_adv_step(
                    batch, model, discriminator, config.num_input_classes, accelerator, 
                    optimizer, opt_disc, global_step, disc_start_step=0, kl_weight=config.kl_weight
                )
            else:
                total_loss, recon_loss, kl_loss, dice_loss, d_loss, d_weight = train_val_step(
                    batch, model, config.num_input_classes, accelerator, 
                    optimizer=optimizer, lr_scheduler=lr_scheduler, is_training=True, kl_weight=config.kl_weight
                )
            
            logs = {"Loss/Total": total_loss, "Loss/Recon": recon_loss, "Loss/KL": kl_loss, "Loss/Dice": dice_loss, "Loss/Discriminator": d_loss, "Weight/Discriminator": d_weight, "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=global_step)

            metrics_hystory['train_loss'].append(total_loss)
            metrics_hystory['train_recon_loss'].append(recon_loss)
            metrics_hystory['train_kl_loss'].append(kl_loss)
            metrics_hystory['train_dice_loss'].append(dice_loss)
            metrics_hystory['train_d_loss'].append(d_loss)
            metrics_hystory['train_d_weight'].append(d_weight)
            metrics_hystory['current_epoch'] = epoch

            progress_bar.set_postfix(loss=total_loss, recon=recon_loss)
            progress_bar.update(1)
            global_step += 1

        if accelerator.is_main_process:
            model.eval()
            val_loss, val_recon, val_kl, val_dice, val_d_loss, val_d_weight = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            for val_batch in val_dataloader:
                if adv_train:
                    v_tot, v_rec, v_kl, v_dice, v_d_loss, v_d_weight = train_vae_adv_step(
                        val_batch, model, discriminator, config.num_input_classes, accelerator, 
                        opt_vae=None, opt_disc=None, global_step=global_step, 
                        disc_start_step=0, kl_weight=config.kl_weight
                    )
                else:
                    v_tot, v_rec, v_kl, v_dice, v_d_loss, v_d_weight = train_val_step(
                        val_batch, model, config.num_input_classes, accelerator, 
                        is_training=False, kl_weight=config.kl_weight
                    )
                val_loss += v_tot
                val_recon += v_rec
                val_kl += v_kl
                val_dice += v_dice
                val_d_loss += v_d_loss
                val_d_weight += v_d_weight
                
            val_loss /= len(val_dataloader)
            val_recon /= len(val_dataloader)
            val_kl /= len(val_dataloader)
            val_dice /= len(val_dataloader)
            val_d_loss /= len(val_dataloader)
            val_d_weight /= len(val_dataloader)
            
            accelerator.log({"Val_Loss/Total": val_loss, "Val_Loss/Recon": val_recon, "Val_Loss/KL": val_kl, "Val_Loss/Dice": val_dice, "Val_Loss/Discriminator": val_d_loss, "Val_Weight/Discriminator": val_d_weight}, step=global_step)
            print(f"Epoch {epoch} | Val Recon Loss: {val_recon:.4f} | Val KL Loss: {val_kl:.4f} | Val Dice Loss: {val_dice:.4f} | Val D Loss: {val_d_loss:.4f}")

            metrics_hystory['val_loss'].append(val_loss)
            metrics_hystory['val_recon_loss'].append(val_recon)
            metrics_hystory['val_kl_loss'].append(val_kl)
            metrics_hystory['val_dice_loss'].append(val_dice)
            metrics_hystory['val_d_loss'].append(val_d_loss)
            metrics_hystory['val_d_weight'].append(val_d_weight)

            unwrapped_model = accelerator.unwrap_model(model)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                sample_and_save_reconstructions(unwrapped_model, config, epoch, config.num_input_classes, accelerator, fixed_sample_batch)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                checkpoint_path = os.path.join(config.exp_dir, f"checkpoint_epoch")
                unwrapped_model.save_pretrained(checkpoint_path)

                with open(os.path.join(config.exp_dir, f"metrics_history.json"), "w") as f:
                    json.dump(metrics_hystory, f, indent=4)

def main():
    config = VAETrainingConfig()

    set_reproducibility(config.seed)
    os.makedirs(config.exp_dir, exist_ok=True)
    log_file = open(f"{config.exp_dir}/log.txt", "w")
    sys.stdout = log_file
    sys.stderr = log_file
    print('Experiment directory:', config.exp_dir)

    dataset_config_path = os.path.join(config.data_path, "dataset_config.json")
    if os.path.exists(dataset_config_path):
        with open(dataset_config_path, "r") as f:
            dataset_config = json.load(f)
        print("Dataset configuration loaded successfully")
    else:
        raise FileNotFoundError(f"Dataset configuration file not found at {dataset_config_path}. Please run the dataset preprocessing script first.")

    all_files = sorted([f for f in os.listdir(config.data_path) if f.endswith('.npy')])
    all_files = sorted(list(set([
        f.replace('_fg.npy', '').replace('_mask.npy', '') 
        for f in all_files if f.endswith('.npy')
    ]))) # compatible with all datatset versions, just extract the unique patient IDs from the file names

    train_files, val_files = train_test_split(all_files, test_size=config.val_fraction, random_state=config.seed)
    
    split_info = {
        'train_indices': train_files,
        'val_indices': val_files
    }

    with open(f"{config.exp_dir}/train_val_split.json", "w") as f:
        json.dump(split_info, f, indent=4)

    with open (f"{config.exp_dir}/grouping_rules_and_labels.json", "w") as f:
        json.dump({
            'grouping_rules': dataset_config.get('grouping_rules_used'),
            'new_labels': dataset_config.get('new_labels_used')
        }, f, indent=4)

    with open (f"{config.exp_dir}/training_config.json", "w") as f:
        json.dump({**sanitize_config(asdict(config)), **sanitize_config(dataset_config)}, f, indent=4)

    train_dataset = FastDatasetDIDC(config.data_path, file_list=train_files)
    val_dataset =  FastDatasetDIDC(config.data_path, file_list=val_files)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size_per_gpu, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size_per_gpu, shuffle=False, num_workers=config.num_workers)
    
    model = AutoencoderKL(
        in_channels=config.num_input_classes,      
        out_channels=config.num_input_classes,     
        latent_channels=config.latent_channels,    
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
        block_out_channels=config.block_out_channels,         
        layers_per_block=config.layers_per_block,
    )

    discr_model = DiscriminatorPatchGAN(in_ch=config.num_input_classes, base_ch=64, n_layers=3)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer_disc = torch.optim.AdamW(discr_model.parameters(), lr=config.learning_rate) # Placeholder, will be used if adv_train=True
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=(len(train_dataloader) * config.num_epochs))

    train_loop(config, 
               model, 
               optimizer, 
               train_dataloader, 
               val_dataloader, 
               lr_scheduler, 
               discriminator=discr_model, 
               opt_disc=optimizer_disc, 
               adv_train=config.adv_train
               )

    log_file.close()

if __name__ == '__main__':
    # main()
    # test train_vae_adv_step with dummy data
    accelerator = Accelerator()
    batch_size = 2
    num_classes = 22
    dummy_batch = {
        'multiClassMask': torch.randint(0, num_classes, (batch_size, 384, 384), device=accelerator.device)
    }
    model = AutoencoderKL(
        in_channels=num_classes,      
        out_channels=num_classes,     
        latent_channels=4,    
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128, 256),         
        layers_per_block=2,
    ).to(accelerator.device)
    discriminator = DiscriminatorPatchGAN(in_ch=num_classes, base_ch=64, n_layers=3).to(accelerator.device)
    opt_vae = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    global_step = 0
    total_vae_loss, recon_loss, kl_loss, dice_loss, d_loss, d_weight = train_vae_adv_step(
        dummy_batch, model, discriminator, num_classes, accelerator, 
        opt_vae, opt_disc, global_step, disc_start_step=0, kl_weight=1e-6
    )
    print(f"Total VAE Loss: {total_vae_loss:.4f}, Recon Loss: {recon_loss:.4f}, KL Loss: {kl_loss:.4f}, Dice Loss: {dice_loss:.4f}, Discriminator Loss: {d_loss:.4f}, Discriminator Weight: {d_weight:.4f}")