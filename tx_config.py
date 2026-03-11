from mt_DIDC_config import LABEL2LABEL, PROPERTY_KEY, NEW_LABELS
from dataclasses import dataclass, field
import torch
from datetime import datetime
import os

@dataclass
class BSSFPConfig:
    PD_max: float = 200.
    T1_max: float = 2000.
    T2_max: float = 500.

    TR: float = 3.0
    TE: float = 1.5
    flip_angle: float = 0.6

    properties_keys: dict = field(default_factory=lambda: PROPERTY_KEY.copy()) 
    label2label: dict = field(default_factory=lambda: LABEL2LABEL.copy())
    label2idx: list = field(default_factory=lambda: NEW_LABELS.copy())


@dataclass
class DiscrConfig:
    in_ch: int = 25 # 22 ch output + 3 ch condition
    base_ch: int = 64
    use_fc: bool = False


@dataclass
class GeneratorConfig:
    in_ch: int = 22 
    num_classes: int = 3
    base_ch: int = 64
    block: str = "BasicBlock"
    pool: bool = False
    dropout_p: float = 0.3


@dataclass
class GANTrainerConfig:
    run_name: str = "texturizer_GAN_train_L2reg_gs"
    exp_dir: str = "./experiments/DIDCV2_TEXT"
    
    data_path: str = "./DIDC_multiclass_coro_v2_prep_2"
    autoenc_path = str = ""
    
    # Hyperparams
    train_batch_size: int = 36
    batch_size_per_gpu: int = 12
    num_workers: int = 8

    val_fraction: float = 0.2
    num_epochs: int = 100
    lr_gen: float = 1e-4
    lr_discr: float = 1e-5
    lambda_properties: float = 10.0
    lambda_physics: float = 5.0
    lambda_reg: float = 0.1 
    warmup_steps_gen: int = 500

    max_sample_statistics: int = 5000 # max number of samples to compute the 99th percentile scale factor for normalization
    global_scale: float = 1.0 # placeholder

    mixed_precision: str = 'fp16'  # 'no', 'fp16', or 'bf16' 
    log_image_epochs = 1

    seed: int = 187
    run_dir: str = ''
    notes: str = "Training GAN with higher lr for generator and L2 regularization on offsets"
    gradient_accumulation_steps: int = 1

    bssfp_model: BSSFPConfig = field(default_factory=BSSFPConfig)
    discr_model: DiscrConfig = field(default_factory=DiscrConfig)
    gen_model: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_dir = os.path.join(self.exp_dir, f"{timestamp}_{self.run_name}")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.gradient_accumulation_steps = max(1, self.train_batch_size // (self.batch_size_per_gpu * num_gpus))


@dataclass
class UnetTrainerConfig:
    run_name: str = "texturizer_UNet_train_L2reg"
    exp_dir: str = "./experiments/DIDCV2_TEXT"
    
    data_path: str = "./DIDC_multiclass_coro_v2_prep_2"
    
    # Hyperparams
    train_batch_size: int = 36
    batch_size_per_gpu: int = 12
    num_workers: int = 8

    val_fraction: float = 0.2
    num_epochs: int = 100
    lr: float = 1e-4 

    lambda_prop: float = 10.0   
    lambda_physics: float = 5.0
    lambda_reg: float = 0.1

    max_sample_statistics: int = 5000 # max number of samples to compute the 99th percentile scale factor for normalization
    global_scale: float = 1.0 # placeholder

    mixed_precision: str = "fp16"
    log_image_epochs = 1
    seed: int = 187
    notes: str = "Training UNet with reconstruction loss (physics and property) with L2 regularization on offsets"
    run_dir: str = ""
    gradient_accumulation_steps: int = 1

    bssfp_model: BSSFPConfig = field(default_factory=BSSFPConfig)
    gen_model: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_dir = os.path.join(self.exp_dir, f"{timestamp}_{self.run_name}")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.gradient_accumulation_steps = max(1, self.train_batch_size // (self.batch_size_per_gpu * num_gpus))
