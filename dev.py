import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import cycle
from datetime import datetime
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import device, nn, optim

from utils import (load_all_data, 
                   squeeze_and_concat, 
                   filter_mask_keep_labels, 
                   multiclass_dice_loss,
                   set_reproducibility)
from datasets import MultiTissueDataset
from unet_advanced import UNetAdvanced as UNetGan
from gan_multidiscr import MultiScaleDiscriminator
from gan_basic import DiscriminatorModel
from train_utils import EarlyStopping, save_checkpoint


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_discriminator(batch, gen, discr, criterion_GAN, optim_discr, device='cpu'):
    input = batch['input_label'].to(device) # Models passed to the function should be already on the same device
    gt = batch['multiClassMask'].to(device)

    # train discriminator
    with torch.no_grad():
        gen_img_probs = gen(input).softmax(dim=1)  # Get probabilities for each class

    gt_onehot = torch.zeros_like(gen_img_probs).scatter_(1, gt.unsqueeze(1), 1.0) # Assigns 1.0 in the corresponding class channel based on gt indices (0-11)

    discr_input_real = torch.cat([input, gt_onehot], dim=1)  # Real pairs: input + gt
    discr_input_fake = torch.cat([input, gen_img_probs], dim=1)  # Fake pairs: input + generated segmentation (probs)

    discr_real = discr(discr_input_real) # Discrim forward pass on real pairs
    discr_fake = discr(discr_input_fake) # Discrim forward pass on fake pairs

    loss = compute_multiscale_loss(discr_real, target=1, criterion=criterion_GAN) + \
        compute_multiscale_loss(discr_fake, target=0, criterion=criterion_GAN)

    optim_discr.zero_grad()
    loss.backward()
    optim_discr.step()

    return loss.item()

def train_generator(batch, gen, discr, criterion_GAN, criterion_CE, optim_gen, lambda_ce, device='cpu'):
    input = batch['input_label'].to(device)
    gt = batch['multiClassMask'].to(device)
    
    gen_img = gen(input)
    gen_img_probs = gen_img.softmax(dim=1)  # Now with gradients enablesd for generator training

    discr_input_fake = torch.cat([input, gen_img_probs], dim=1)  # Fake pairs: input + generated segmentation (probs)
    discr_fake = discr(discr_input_fake) # Discrim forward pass on fake pairs


    ce_loss = criterion_CE(gen_img, gt)  # CE loss between generated segmentation and ground truth (B, 12, H, W)
    loss_gen = compute_multiscale_loss(discr_fake, target=1, criterion=criterion_GAN) + lambda_ce * ce_loss

    optim_gen.zero_grad()
    loss_gen.backward()
    optim_gen.step()

    return loss_gen.item(), ce_loss.item()

def validate_generator(val_dataloader, gen, discr, criterion_GAN, criterion_CE, lambda_ce, device='cpu'):
    gen.eval()
    discr.eval()

    total_gan_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input = batch['input_label'].to(device)
            gt = batch['multiClassMask'].to(device)

            gen_img = gen(input)
            gen_img_probs = gen_img.softmax(dim=1)

            discr_input_fake = torch.cat([input, gen_img_probs], dim=1)
            discr_fake = discr(discr_input_fake)

            # Compute losses
            ce_loss = criterion_CE(gen_img, gt)
            gan_loss = compute_multiscale_loss(discr_fake, target=1, criterion=criterion_GAN)
            dice_loss = multiclass_dice_loss(gen_img, gt) # Logits here as input

            total_gan_loss += gan_loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            num_batches += 1

    avg_gan_loss = total_gan_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_dice_loss = total_dice_loss / num_batches

    return avg_gan_loss, avg_ce_loss, avg_dice_loss

def train_gan(num_steps, n_discr_steps, n_gen_steps, val_check_interval, gen, discr, dataloader, val_dataloader, criterion_GAN, criterion_CE, optim_gen, optim_discr, lambda_ce, es, device, exp_dir):
    pbar = tqdm(range(num_steps), desc="Training step", file=sys.__stderr__)
    
    metrics_history = {
        'train_D_loss': [],
        'train_G_loss': [],
        'train_CE_loss': [],
        'val_G_loss': [],
        'val_CE_loss': [],
        'val_Dice_loss': [],
        'current_step': 0.0
    }

    inf_iter = cycle(dataloader) # generate infinite iterator on the training dataset

    for step in pbar:
        batch = next(inf_iter)

        gen.train()
        discr.train()

        loss_discr = 0.0
        for _ in range(n_discr_steps): # Nested loop to allow multiple discriminator updates per generator update
            running_loss_discr = train_discriminator(batch, gen, discr, criterion_GAN, optim_discr, device)
            loss_discr += running_loss_discr
        loss_discr /= n_discr_steps

        loss_gen = 0.0
        ce_loss = 0.0
        for _ in range(n_gen_steps): # Nested loop to allow multiple generator updates per discriminator update
            running_loss_gen, running_ce_loss = train_generator(batch, gen, discr, criterion_GAN, criterion_CE, optim_gen, lambda_ce, device)
            loss_gen += running_loss_gen
            ce_loss += running_ce_loss
        loss_gen /= n_gen_steps
        ce_loss /= n_gen_steps

        metrics_history['train_D_loss'].append(loss_discr)
        metrics_history['train_G_loss'].append(loss_gen)
        metrics_history['train_CE_loss'].append(ce_loss)
        metrics_history['current_step'] = step

        if step % val_check_interval == 0 and step > 0: # Validate every val_check_interval steps
            val_gan_loss, val_ce_loss, val_dice_loss = validate_generator(val_dataloader, gen, discr, criterion_GAN, criterion_CE, lambda_ce, device)
            metrics_history['val_G_loss'].append(val_gan_loss)
            metrics_history['val_CE_loss'].append(val_ce_loss)
            metrics_history['val_Dice_loss'].append(val_dice_loss)

            print(f"VAL: Step {step}: Val GAN Loss: {val_gan_loss:.4f}, Val CE Loss: {val_ce_loss:.4f}, Val Dice Loss: {val_dice_loss:.4f}")

            # Save checkpoint
            save_checkpoint(exp_dir, step, gen, discr, optim_gen, optim_discr, metrics_history)

            # check for early stopping based on validation dice score
            es.check_early_stop(val_dice_loss)
            if es.no_improvement_count == 0 and step > 0: # Save best model based on validation dice score
                torch.save(gen.state_dict(), f"{exp_dir}/best_generator.pth")
            if es.stop_training:
                print(f"Early stopping triggered at step {step}")
                break
        
        if step % 20 == 0: # Update progress bar every 20 steps
            print(f"Step {step}: Train D Loss: {loss_discr:.4f}, Train G Loss: {loss_gen:.4f}, Train CE Loss: {ce_loss:.4f}")
            pbar.set_postfix({
                'train_D_loss': f'{loss_discr:.4f}', 
                'train_G_loss': f'{loss_gen:.4f}',
                'train_CE_loss': f'{ce_loss:.4f}',
                'val_G_loss': f'{val_gan_loss:.4f}' if step >= val_check_interval else 'N/A',
                'val_CE_loss': f'{val_ce_loss:.4f}' if step >= val_check_interval else 'N/A',
                'val_Dice_loss': f'{val_dice_loss:.4f}' if step >= val_check_interval else 'N/A'
            })
    
    # Save metrics history to a JSON file
    with open(f"{exp_dir}/metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=4)

    return metrics_history

def compute_multiscale_loss(discr_outputs, target, criterion):
    """
    Function to compute GAN loss (BCEWithLogitsLoss) averaged across multiple discriminators in the MultiScaleDiscriminator.
    
    :param discr_outputs: List of tuples containing discriminator outputs and features (See MultiScaleDiscriminator forward method)
    :param target: integer (1 for real, 0 for fake) to create target tensors for GAN loss computation
    :param criterion: Loss function to compute the GAN loss (e.g., BCEWithLogitsLoss)
    :return: Computed GAN loss averaged across all discriminators
    """
    loss = 0.0
    for pred, _ in discr_outputs:
        target_tensor = torch.ones_like(pred) if target == 1 else torch.zeros_like(pred)
        loss += criterion(pred, target_tensor) 
    
    return loss/len(discr_outputs)










if __name__ == "__main__":
    # generate fake batch for the generator
    
    batch = {
        'input_label': torch.randn(8, 4, 256, 256, device=DEVICE), # 4 channels for the input (ED, ES, ED+ES, binary mask)
        'multiClassMask': torch.randint(0, 12, (8, 256, 256), device=DEVICE) # 12 classes for the multi-class segmentation mask (0-11)
    }

    input = batch['input_label']
    gt = batch['multiClassMask']

    print(input.shape, gt.shape)
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_CE = nn.CrossEntropyLoss() # CE loss for segmentation (better than L1 for classification tasks)
    
    gen = UNetGan(in_ch=4, num_classes=12, base_ch=64).to(DEVICE) # 4 channels for the input (ED, ES, ED+ES, binary mask), 12 channels for the output segmentation (one per class)
    discr = MultiScaleDiscriminator(in_ch=16, n_discriminators=4).to(DEVICE) # 4 channels from input + 12 channels from generated segmentation

    optim_gen = optim.Adam(gen.parameters(), lr=1e-4)
    optim_discr = optim.Adam(discr.parameters(), lr=1e-4)

    print(train_discriminator(batch, gen, discr, criterion_GAN, optim_discr, DEVICE))
    print(train_generator(batch, gen, discr, criterion_GAN, criterion_CE, optim_gen, lambda_ce=1.0, device=DEVICE))
    print(validate_generator([batch], gen, discr, criterion_GAN, criterion_CE, lambda_ce=1.0, device=DEVICE))
    
