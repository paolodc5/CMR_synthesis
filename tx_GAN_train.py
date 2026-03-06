import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import asdict, dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import FastDatasetDIDC

from accelerate import Accelerator
from accelerate.logging import get_logger

from mt_DIDC_config import LABEL2LABEL

# Setup logger
logger = get_logger(__name__, log_level="INFO")

@dataclass
class GANTrainingConfig:
    run_name: str = "PropertyGAN_bSSFP"
    exp_dir: str = "./experiments/PropertyGAN"
    
    data_path_labels: str = './DIDC_multiclass_coro_v2_prep'
    data_path_properties: str = None

    gan_type: str = "spade" # "basic", "multiscale", "spade"

    
    # Parametri bSSFP (per il simulatore fisico)
    tr: float = 4.0 
    te: float = 2.0
    alpha: float = 60.0 # Flip angle
    
    # Hyperparams
    train_batch_size: int = 8
    val_fraction: float = 0.2
    num_epochs: int = 100
    lr_gen: float = 1e-4
    lr_discr: float = 1e-4
    lambda_perceptual: float = 10.0
    lambda_physics: float = 5.0 # Peso per la loss valutata sull'immagine generata
    
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1
    seed: int = 187
    run_dir: str = ""

    def __post_init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_dir = os.path.join(self.exp_dir, f"{timestamp}_{self.run_name}")


class CustomDatasetTexturizer(FastDatasetDIDC):
    def __init__(self, config: GANTrainingConfig, file_list=None):
        super().__init__(data_path=config.data_path_labels, file_list=file_list)
        self.config = config

    def __getitem__(self, idx):
        # Carica le label originali e le proprietà
        sample = super().__getitem__(idx)
        labels = sample['multiClassMask']
        

        return labels


if __name__ == "__main__":
    config = GANTrainingConfig()

    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    print(f"Starting GAN training with config: {asdict(config)}")

    dataset = CustomDatasetTexturizer(config)
    print(f"Dataset loaded with {len(dataset)} samples")
    labels = dataset[44]
    print(f"Sample labels shape: {labels.shape}, unique labels: {torch.unique(labels)}")

