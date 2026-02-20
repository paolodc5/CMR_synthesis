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