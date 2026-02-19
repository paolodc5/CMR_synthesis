import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import cycle
from datetime import datetime
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import device, nn, optim

from utils import (multiclass_dice_loss,
                   set_reproducibility)
from datasets import DatasetDIDC, LazyDatasetDIDC
from unet_advanced import UNetAdvanced as UNetGan
from gan_multidiscr import MultiScaleDiscriminator
from train_utils import EarlyStopping, save_checkpoint
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS


DATA_FOLDER = "./New_dictionary"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 187
set_reproducibility(SEED)

RUN_NAME = "MT_DIDC_multiscale"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
EXP_DIR = f"./experiments/DIDC/{TIMESTAMP}_{RUN_NAME}"

# Hyperparameters (not best practice to be defined here)
NUM_STEPS = 15000
N_DISCR_STEPS = 1
N_GEN_STEPS = 1
VAL_CHECK_INTERVAL = 120
LAMBDA_CE = 10.0
LAMBDA_PERC = 5.0
DROPOUT_GEN = 0.3
NUM_DISCRIMINATORS = 4
GEN_LR = 1e-4
DISCR_LR = 1e-5
PATIENCE_ES = 15 # num * VAL_CHECK_INTERVAL steps with no improvement
DELTA_ES = 0.01 # minimum improvement in validation dice loss to reset early stopping counter
BATCH_SIZE = 16
VAL_FRACTION = 0.2
NOTES="Mutliscale gan on DIDC dataset FIXED with new preprocessing, dataset with remapping of 'other_tissue' pixels to their NNs, perceptual loss added and lr as originally"
PARALLEL = True

LAZY_DATASET = True
NUM_WORKERS = 8 if LAZY_DATASET else 0 # Set num_workers > 0 only for LazyDataset to speed up data loading, for in-memory dataset it can cause unnecessary overhead and potential issues with multiprocessing on some platforms (e.g., Windows)
PIN_MEMORY = False # Set pin_memory to True only for LazyDataset to speed up data transfer to GPU, for in-memory dataset it can cause unnecessary overhead
NON_BLOCKING_GPU_LOADING = PIN_MEMORY # Set non_blocking to True only for LazyDataset to allow asynchronous GPU transfers, for in-memory dataset it can cause issues if the data is already on CPU and not pinned

# preprocessing hyperparameters
REMAP_NN = True # Whether to apply the NN remapping of "Other_tissue" pixels to the most common label among their k nearest neighbors that are not "Other_tissue". This is done in the DatasetDIDC class and can be turned on/off with this flag.
THRESHOLD_CLASSES = 50 # Minimum number of pixels for a class to be kept, otherwise those pixels are set to "Other_tissue". Set to None to disable this step. 
MIN_BLOB_SIZE = 10 # Minimum size of connected components to keep for each class, smaller blobs are removed. Set to None to disable this step.
TARGET_SIZE = (384, 384) # Target size for resizing the input images and masks (H, W)


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

def train_generator(batch, gen, discr, criterion_GAN, criterion_CE, criterion_perc, optim_gen, lambda_ce, lambda_perc, device='cpu'):
    input = batch['input_label'].to(device, non_blocking=NON_BLOCKING_GPU_LOADING) # Models passed to the function should be already on the same device
    gt = batch['multiClassMask'].to(device, non_blocking=NON_BLOCKING_GPU_LOADING)
    
    gen_img = gen(input)
    gen_img_probs = gen_img.softmax(dim=1)  # Now with gradients enablesd for generator training

    discr_input_fake = torch.cat([input, gen_img_probs], dim=1)  # Fake pairs: input + generated segmentation (probs)
    discr_fake = discr(discr_input_fake) # Discrim forward pass on fake pairs

    gt_onehot = torch.zeros_like(gen_img_probs).scatter_(1, gt.unsqueeze(1), 1.0) # Assigns 1.0 in the corresponding class channel based on gt indices (0-11)
    discr_input_real = torch.cat([input, gt_onehot], dim=1)  # Real pairs: input + gt
    
    with torch.no_grad(): # Important otherwise I updated the discriminator's gradients while training the generator. 
        discr_real = discr(discr_input_real) # Discrim forward pass on real pairs

    perc_loss = compute_perceptual_loss(discr_real, discr_fake, criterion_perc)
    ce_loss = criterion_CE(gen_img, gt)  # CE loss between generated segmentation and ground truth (B, 12, H, W)
    loss_gen = compute_multiscale_loss(discr_fake, target=1, criterion=criterion_GAN) + lambda_ce * ce_loss + lambda_perc * perc_loss

    optim_gen.zero_grad()
    loss_gen.backward()
    optim_gen.step()

    return loss_gen.item(), ce_loss.item(), perc_loss.item()

def validate_generator(val_dataloader, gen, discr, criterion_GAN, criterion_CE, criterion_perc, lambda_ce, lambda_perc, device='cpu'):
    gen.eval()
    discr.eval()

    total_gan_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0
    total_perc_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input = batch['input_label'].to(device)
            gt = batch['multiClassMask'].to(device)

            gen_img = gen(input)
            gen_img_probs = gen_img.softmax(dim=1)

            discr_input_fake = torch.cat([input, gen_img_probs], dim=1)
            discr_fake = discr(discr_input_fake)

            gt_onehot = torch.zeros_like(gen_img_probs).scatter_(1, gt.unsqueeze(1), 1.0) # Assigns 1.0 in the corresponding class channel based on gt indices (0-11)
            discr_input_real = torch.cat([input, gt_onehot], dim=1)  # Real pairs: input + gt
            discr_real = discr(discr_input_real)

            # Compute losses
            ce_loss = criterion_CE(gen_img, gt)
            gan_loss = compute_multiscale_loss(discr_fake, target=1, criterion=criterion_GAN)
            dice_loss = multiclass_dice_loss(gen_img, gt) # Logits here as input
            perc_loss = compute_perceptual_loss(discr_real, discr_fake, criterion_perc)
            total_gan_loss += gan_loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            total_perc_loss += perc_loss.item()
            num_batches += 1

    avg_gan_loss = total_gan_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_dice_loss = total_dice_loss / num_batches
    avg_perc_loss = total_perc_loss / num_batches

    return avg_gan_loss, avg_ce_loss, avg_dice_loss, avg_perc_loss

def train_gan(num_steps, n_discr_steps, n_gen_steps, val_check_interval, gen, discr, dataloader, val_dataloader, criterion_GAN, criterion_CE, criterion_perc, optim_gen, optim_discr, lambda_ce, lambda_perc, es, device, exp_dir):
    pbar = tqdm(range(num_steps), desc="Training step", file=sys.__stderr__)
    
    metrics_history = {
        'train_D_loss': [],
        'train_G_loss': [],
        'train_CE_loss': [],
        'train_Perc_loss': [],
        'val_G_loss': [],
        'val_CE_loss': [],
        'val_Dice_loss': [],
        'val_Perc_loss': [],
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
        perc_loss = 0.0
        for _ in range(n_gen_steps): # Nested loop to allow multiple generator updates per discriminator update
            running_loss_gen, running_ce_loss, running_perc_loss = train_generator(batch, gen, discr, criterion_GAN, criterion_CE, criterion_perc, optim_gen, lambda_ce, lambda_perc, device)
            loss_gen += running_loss_gen
            ce_loss += running_ce_loss
            perc_loss += running_perc_loss
        loss_gen /= n_gen_steps
        ce_loss /= n_gen_steps
        perc_loss /= n_gen_steps

        metrics_history['train_D_loss'].append(loss_discr)
        metrics_history['train_G_loss'].append(loss_gen)
        metrics_history['train_CE_loss'].append(ce_loss)
        metrics_history['train_Perc_loss'].append(perc_loss)
        metrics_history['current_step'] = step

        if step % val_check_interval == 0 and step > 0: # Validate every val_check_interval steps
            val_gan_loss, val_ce_loss, val_dice_loss, val_perc_loss = validate_generator(val_dataloader, gen, discr, criterion_GAN, criterion_CE, criterion_perc, lambda_ce, lambda_perc, device)
            metrics_history['val_G_loss'].append(val_gan_loss)
            metrics_history['val_CE_loss'].append(val_ce_loss)
            metrics_history['val_Dice_loss'].append(val_dice_loss)
            metrics_history['val_Perc_loss'].append(val_perc_loss)

            print(f"VAL: Step {step}: Val GAN Loss: {val_gan_loss:.4f}, Val CE Loss: {val_ce_loss:.4f}, Val Dice Loss: {val_dice_loss:.4f}, Val Perc Loss: {val_perc_loss:.4f}")

            # Save checkpoint
            save_checkpoint(exp_dir, step, gen, discr, optim_gen, optim_discr, metrics_history)

            # Save metrics history to a JSON file
            with open(f"{exp_dir}/metrics_history.json", "w") as f:
                json.dump(metrics_history, f, indent=4)
            
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

def main():
    cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES')
    device_ids = [int(x) for x in cuda_env.split(',') if x.strip()] if (PARALLEL and cuda_env) else None

    os.makedirs(EXP_DIR, exist_ok=True)

    log_file = open(f"{EXP_DIR}/log.txt", "w")
    sys.stdout = log_file
    sys.stderr = log_file
    print('Experiment directory:', EXP_DIR)

    # checks on devices
    p = torch.cuda.get_device_properties(DEVICE)
    print(f"Selected device: {DEVICE}")
    print(f'Num available GPUs: ', torch.cuda.device_count())
    print(f"Device: {p.name} (Memory: {p.total_memory / 1e9:.2f} GB)")

    # Dataset
    all_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.npy')])
    train_files, val_files = train_test_split(all_files, test_size=VAL_FRACTION, random_state=SEED)

    ### Save train-val split indices for reproducibility, split is made patient-wise
    split_info = {
        'train_indices': train_files,
        'val_indices': val_files
    }
    with open(f"{EXP_DIR}/train_val_split.json", "w") as f:
        json.dump(split_info, f, indent=4)
    
    with open (f"{EXP_DIR}/grouping_rules_and_labels.json", "w") as f:
        json.dump({
            'grouping_rules': GROUPING_RULES,
            'new_labels': NEW_LABELS
        }, f, indent=4)

    train_dataset = LazyDatasetDIDC(DATA_FOLDER, GROUPING_RULES, NEW_LABELS, target_size=TARGET_SIZE, rm_black_slices=True, file_list=train_files, remap_nn=REMAP_NN, threshold_classes=THRESHOLD_CLASSES, min_blob_size=MIN_BLOB_SIZE)    
    val_dataset = LazyDatasetDIDC(DATA_FOLDER, GROUPING_RULES, NEW_LABELS, target_size=TARGET_SIZE, rm_black_slices=True, file_list=val_files, remap_nn=REMAP_NN, threshold_classes=THRESHOLD_CLASSES, min_blob_size=MIN_BLOB_SIZE)    

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


    # Training objects instantiation
    n_input_classes = 4
    n_output_classes = len(train_dataset.new_labels)
    gen = UNetGan(in_ch=n_input_classes, num_classes=n_output_classes, dropout_p=DROPOUT_GEN).to(DEVICE)
    discr = MultiScaleDiscriminator(in_ch=n_input_classes+n_output_classes, n_discriminators=NUM_DISCRIMINATORS).to(DEVICE)

    if torch.cuda.device_count() > 1 and PARALLEL:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        gen = nn.DataParallel(gen)
        discr = nn.DataParallel(discr)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_CE = nn.CrossEntropyLoss() # CE loss for segmentation (better than L1 for classification tasks)
    criterion_perc = nn.L1Loss() # L1 loss for perceptual loss on discriminator features (could also try MSELoss)

    optim_gen = optim.Adam(gen.parameters(), lr=GEN_LR)
    optim_discr = optim.Adam(discr.parameters(), lr=DISCR_LR)

    es = EarlyStopping(patience=PATIENCE_ES, delta=DELTA_ES)  

    # save config
    config = {
        'num_steps': NUM_STEPS,
        'n_discr_steps': N_DISCR_STEPS,
        'lambda_ce': LAMBDA_CE,
        'lambda_perc': LAMBDA_PERC,
        'num_discriminators': NUM_DISCRIMINATORS,
        'val_check_interval': VAL_CHECK_INTERVAL,
        'batch_size': train_dataloader.batch_size,
        'learning_rate_gen': optim_gen.param_groups[0]['lr'],
        'learning_rate_discr': optim_discr.param_groups[0]['lr'],
        'remapping_nn': REMAP_NN,
        'batch_size': BATCH_SIZE,
        'notes': NOTES,
        'patience_es': PATIENCE_ES,
        'delta_es': DELTA_ES,
        'dropout_gen': DROPOUT_GEN,
        'num_gpus': torch.cuda.device_count() if PARALLEL else 1,
        'validation_fraction': VAL_FRACTION,
        'target_size': TARGET_SIZE,
        'seed': SEED,
        'parallel': PARALLEL,
        'device_ids': device_ids,
        'threshold_classes': THRESHOLD_CLASSES,
        'min_blob_size': MIN_BLOB_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY,
        'non_blocking_gpu_loading': NON_BLOCKING_GPU_LOADING
    }

    with open(f"{EXP_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # training loop
    history = train_gan(NUM_STEPS, 
                        N_DISCR_STEPS,
                        N_GEN_STEPS, 
                        VAL_CHECK_INTERVAL, 
                        gen, 
                        discr, 
                        train_dataloader, 
                        val_dataloader,
                        criterion_GAN, 
                        criterion_CE, 
                        criterion_perc,
                        optim_gen, 
                        optim_discr, 
                        LAMBDA_CE, 
                        LAMBDA_PERC,
                        es, 
                        DEVICE,
                        EXP_DIR)
    
    log_file.close()

if __name__ == "__main__":
    main()