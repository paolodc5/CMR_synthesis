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

from utils import (load_all_data, 
                   squeeze_and_concat, 
                   filter_mask_keep_labels, 
                   multiclass_dice_loss,
                   set_reproducibility)
from datasets import DatasetDIDC
from unet_advanced import UNetAdvanced
from gan_basic import DiscriminatorModel
from train_utils import EarlyStopping, save_checkpoint
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS


DATA_FOLDER = "./New_dictionary"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 187
set_reproducibility(SEED)

RUN_NAME = "MT_DIDC_UNet"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
EXP_DIR = f"./experiments/DIDC/{TIMESTAMP}_{RUN_NAME}"

VAL_FRACTION = 0.2
TARGET_SIZE = (384, 384) # Target size for resizing the input images and masks (H, W)

# Hyperparameters (not best practice to be defined here)

PATIENCE_ES = 15 # num * VAL_CHECK_INTERVAL steps with no improvement
DELTA_ES = 0.005 # minimum improvement in validation dice loss to reset early stopping counter
BATCH_SIZE = 8
NOTES="Basic UNet training on DIDC data"
PARALLEL = True
LR = 1e-3
EPOCHS = 150

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_label']  # (B, C, H, W)
            targets = batch['multiClassMask']  # (B, H, W)

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs) 
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            n_samples += batch_size
            val_loss += loss.item() * batch_size
            dice_score += multiclass_dice_loss(outputs, targets).item() * batch_size

    average_loss = val_loss / n_samples
    average_dice_score = dice_score / n_samples

    return average_loss, average_dice_score


def train(model, dataloader, val_dataloader, criterion, optimizer, scheduler, early_stopping, epochs, device):
    model.to(device)
    model.train()
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_dice_score': [],
        'val_dice_score': [],
        'learning_rate': []
    }

    for epoch in range(epochs):
        running_loss = 0.0
        n_samples = 0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}", file=sys.__stderr__) as loop:

            for batch in loop:
                model.train()

                inputs = batch['input_label']  # (B, C, H, W)
                targets = batch['multiClassMask']  # (B, H, W)

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs) 
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = inputs.size(0)
                n_samples += batch_size
                running_loss += loss.item() * batch_size
                current_average_loss = running_loss / n_samples

                loop.set_postfix({'train_loss: ': f'{current_average_loss:.2f}',
                                'lr: ': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        scheduler.step()
        dice_score = multiclass_dice_loss(outputs, targets).item()

        metrics['train_loss'].append(current_average_loss)
        metrics['train_dice_score'].append(dice_score)

        # Check early stopping condition
        if epoch % 1 == 0:  # Validate every epoch
            val_loss, val_dice_loss = validate(model, val_dataloader, criterion, device)
            metrics['val_loss'].append(val_loss)
            metrics['val_dice_score'].append(val_dice_loss)
            metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
            early_stopping.check_early_stop(val_dice_loss)
            print(f"VAL epoch {epoch+1}/{epochs}. Val Loss: {val_loss:.4f}, Validation Dice Score: {val_dice_loss:.4f}")
        
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_average_loss:.4f}, 1-Dice Score: {dice_score:.4f}")


    return model, metrics



def main():
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

    train_dataset = DatasetDIDC(DATA_FOLDER, GROUPING_RULES, NEW_LABELS, target_size=TARGET_SIZE, rm_black_slices=True, file_list=train_files)    
    val_dataset = DatasetDIDC(DATA_FOLDER, GROUPING_RULES, NEW_LABELS, target_size=TARGET_SIZE, rm_black_slices=True, file_list=val_files)    

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetAdvanced(in_ch=4, num_classes=len(NEW_LABELS)).to(DEVICE)

    if torch.cuda.device_count() > 1 and PARALLEL:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    config = {
        'seed': SEED,
        'val_fraction': VAL_FRACTION,
        'target_size': TARGET_SIZE,
        'run_name': RUN_NAME,
        'notes': NOTES,
        'parallel': PARALLEL,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'epochs': EPOCHS,
        'patience_es': PATIENCE_ES,
        'delta_es': DELTA_ES,
        'target_size': TARGET_SIZE,
    }

    with open(f"{EXP_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=PATIENCE_ES, delta=DELTA_ES, verbose=True)

    model, metrics = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, early_stopping, EPOCHS, DEVICE)

    with open(f"{EXP_DIR}/metric_history.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    torch.save(model.state_dict(), f"{EXP_DIR}/model_final.pth")


if __name__ == "__main__":
    main()