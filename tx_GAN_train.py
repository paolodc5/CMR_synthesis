import os
import sys
import json
import logging

import numpy as np
import math
import matplotlib.pyplot as plt

from datetime import datetime
from dataclasses import asdict, dataclass, field
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from datasets import FastDatasetDIDC
from gan_basic import DiscriminatorModel
from unet_advanced import UNetAdvanced as GeneratorModel
from mt_DIDC_config import GROUPING_RULES, LABEL2LABEL, PROPERTY_KEY, NEW_LABELS
from utils import setup_logger, set_reproducibility

from accelerate import Accelerator
from accelerate.logging import get_logger


# Setup logger
logger = get_logger(__name__, log_level="INFO")

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
    run_name: str = "texturizer_GAN_train"
    exp_dir: str = "./experiments/DIDCV2_TEXT"
    
    data_path: str = "./DIDC_multiclass_coro_v2_prep_2"
    autoenc_path = str = ""
    
    # Hyperparams
    train_batch_size: int = 32
    batch_size_per_gpu: int = 12
    num_workers: int = 8

    val_fraction: float = 0.2
    num_epochs: int = 100
    lr_gen: float = 1e-4
    lr_discr: float = 1e-4
    lambda_perceptual: float = 10.0
    lambda_physics: float = 5.0 # Peso per la loss valutata sull'immagine generata

    mixed_precision: str = "fp16"

    log_image_epochs = 1

    seed: int = 187
    run_dir: str = ""
    gradient_accumulation_steps: int = 1

    bssfp_model: BSSFPConfig = field(default_factory=BSSFPConfig)
    discr_model: DiscrConfig = field(default_factory=DiscrConfig)
    gen_model: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_dir = os.path.join(self.exp_dir, f"{timestamp}_{self.run_name}")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.gradient_accumulation_steps = max(1, self.train_batch_size // (self.batch_size_per_gpu * num_gpus))


class CustomDatasetTexturizer(FastDatasetDIDC):
    def __init__(self, data_path: str, file_list: list[str]=None):
        super().__init__(data_path=data_path, file_list=file_list)

    def __getitem__(self, idx):
        # load the original sample and take only the mask
        original_sample = super().__getitem__(idx)
        label = original_sample['multiClassMask']

        # load mri slice from the same pat/slice
        pat_id, slice_idx = self.samples[idx]
        pat_path = os.path.join(self.data_path, f"{pat_id}_img.npy")
        props_path = os.path.join(self.data_path, f"{pat_id}_props.npy")

        mri_slice = np.load(pat_path, mmap_mode='r')[slice_idx]

        try: 
            props_slice = np.load(props_path, mmap_mode='r')[slice_idx] # if slices are the first dimension
        except IndexError:
            props_slice = np.load(props_path, mmap_mode='r')[..., slice_idx]

        mri_slice_tensor = torch.from_numpy(mri_slice.copy()).float()
        props_slice_tensor = torch.from_numpy(props_slice.copy()).float()

        return {'input_label': label, 'mri_slice': mri_slice_tensor, 'props_slice': props_slice_tensor}


class bSSFPSimulator(nn.Module):
    def __init__(self, config: BSSFPConfig):
        super().__init__()

        self.PD_max = config.PD_max
        self.T1_max = config.T1_max
        self.T2_max = config.T2_max

        self.TR = config.TR
        self.TE = config.TE
        
        self.flip_angle = config.flip_angle

        # pre-computations
        self.sin_alpha = math.sin(self.flip_angle)
        self.cos_alpha = math.cos(self.flip_angle)

        # Build the LUT and register it as a buffer (so it gets moved to device with the model)
        self._build_lut(config)

    @torch.no_grad()
    def get_offsets_from_absolute(self, absolute_props: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Extracts the offset maps (GT) starting from the absolute properties.

        absolute_props: Tensor (B, 3, H, W) with the values [PD, T1, T2] in absolute terms.
        masks: Tensor (B, 1, H, W) or (B, H, W) with the multi-tissue labels.
        Returns: Tensor (B, 3, H, W) with the normalized pure offsets.
        """
        if masks.ndim == 4:
            masks = torch.argmax(masks, dim=1)
        init_props = self._initialize_tissue_maps(masks)
        
        max_vals = torch.tensor(
            [self.PD_max, self.T1_max, self.T2_max], 
            device=absolute_props.device, 
            dtype=torch.float32
        ).view(1, 3, 1, 1) # reshape for broadcasting
        
        absolute_props_norm = absolute_props / max_vals
        offsets = absolute_props_norm - init_props
        
        return offsets

    def _build_lut(self, config):
        """Pre-computes label mappings for initial values of PD, T1, T2"""
        num_labels = len(config.label2idx)
        lut = torch.zeros((num_labels, 3), dtype=torch.float32) # 3 properties: PD, T1, T2 for each pixel value (label)
        
        for old_label, new_label in config.label2label.items():
            idx = config.label2idx.index(old_label)
            props = config.properties_keys[new_label]
            
            lut[idx, 0] = props[0] / self.PD_max
            lut[idx, 1] = props[1] / self.T1_max
            lut[idx, 2] = props[2] / self.T2_max

        self.register_buffer('tissue_lut', lut)

    def _initialize_tissue_maps(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Creates initial PD, T1, T2 maps based on the input segmentation mask using the pre-computed LUT.
        mask: Tensor (B, 1, H, W) or (B, H, W)
        """
        if mask.dim() == 4:
            mask = mask.squeeze(1)
            
        mask = mask.long() # must be long type for indexing
        init_prop = self.tissue_lut[mask]   # (B, H, W) -> (B, H, W, 3)
        if mask.ndim == 4:
            init_prop = init_prop.argmax(1)

        init_prop = init_prop.permute(0, 3, 1, 2) # (B, H, W, 3) -> (B, 3, H, W)

        return init_prop

    def _bssfp_signal_model(self, offsets: torch.Tensor, init_values: torch.Tensor) -> torch.Tensor:
        """Physical bSSFP model that simulates the MRI signal based on the input tissue properties.
        offsets: Tensor (B, 3, H, W) - predicted offsets for PD, T1, T2
        init_values: Tensor (B, 3, H, W) - initial PD, T1, T2 maps from the LUT
        Returns: Simulated MRI slice (B, H, W)
        """
        # [...,0:1,...] to keep the channel dimension for broadcasting
        PD = torch.clamp(offsets[:, 0:1, :, :] + init_values[:, 0:1, :, :], 0.001, 50.0) * self.PD_max
        T1 = torch.clamp(offsets[:, 1:2, :, :] + init_values[:, 1:2, :, :], 0.001, 50.0) * self.T1_max
        T2 = torch.clamp(offsets[:, 2:3, :, :] + init_values[:, 2:3, :, :], 0.001, 50.0) * self.T2_max
        
        num = PD * self.sin_alpha
        den = (T1 / T2 + 1.0) - self.cos_alpha * (T1 / T2 - 1.0)
        decay = torch.exp(-self.TE / T2) 
        m = torch.abs((num / den) * decay) * 0.05
        
        return m

    def forward(self, props_offest: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        init_props = self._initialize_tissue_maps(masks)
        simulated_slice = self._bssfp_signal_model(props_offest, init_props)
        
        return simulated_slice


class GANTrainer:
    def __init__(
        self, 
        config: GANTrainerConfig, 
        gen: nn.Module, 
        discr: nn.Module, 
        bssfp_sim: bSSFPSimulator,
        opt_G: torch.optim.Optimizer, 
        opt_D: torch.optim.Optimizer, 
        train_loader, 
        val_loader, 
        accelerator: Accelerator
    ):
        self.config = config
        self.gen = gen
        self.discr = discr
        self.bssfp_sim = bssfp_sim
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator

        # Loss Functions
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        
        # Tracking states
        self.global_step = 0

        logger.info("GANTrainer initialized correctly")

    def train_step(self, batch):
        """Executes a single training step for D and G"""
        original_mask = batch['input_label'].unsqueeze(1).float() # (B, 1, H, W) - add channel dimension for condition

        condition = original_mask.clone()
        if self.config.gen_model.in_ch > 1:
            condition = F.one_hot(original_mask.squeeze(1).long(), num_classes=self.config.gen_model.in_ch).permute(0, 3, 1, 2).float().contiguous() # (B, 1, H, W) -> (B, C, H, W)

        absolute_gt_props = batch['props_slice'].float()                           # (B, 3, H, W)
        gt_mri = batch['mri_slice'].unsqueeze(1).float()                           # (B, 1, H, W)

        gt_offsets = self.bssfp_sim.get_offsets_from_absolute(absolute_gt_props, condition) # retrieves the target offsets
    
        # train discriminator
        self.discr.train()
        with self.accelerator.accumulate(self.discr):
            self.opt_D.zero_grad()

            self.gen.eval()
            with torch.no_grad():
                pred_offsets_detached = self.gen(condition)

            real_pair = torch.cat([condition, gt_offsets], dim=1)
            fake_pair = torch.cat([condition, pred_offsets_detached], dim=1)
            
            combined_pair = torch.cat([real_pair, fake_pair], dim=0) # this is necessary for distributed training
            d_combined = self.discr(combined_pair)
            d_real, d_fake = torch.split(d_combined, real_pair.shape[0])
            
            loss_D_real = self.criterion_GAN(d_real, torch.ones_like(d_real))
            loss_D_fake = self.criterion_GAN(d_fake, torch.zeros_like(d_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            self.accelerator.backward(loss_D)
            self.opt_D.step()

        # train generator
        self.gen.train()
        with self.accelerator.accumulate(self.gen):
            self.opt_G.zero_grad()

            pred_offsets = self.gen(condition)
            fake_pair_for_G = torch.cat([condition, pred_offsets], dim=1)

            self.discr.eval()
            for param in self.discr.parameters():
                param.requires_grad = False

            d_fake_for_G = self.discr(fake_pair_for_G)
            loss_G_adv = self.criterion_GAN(d_fake_for_G, torch.ones_like(d_fake_for_G))
            
            for param in self.discr.parameters():
                param.requires_grad = True

            # Properties Loss (L1 over T1, T2, PD)
            loss_G_prop = self.criterion_L1(pred_offsets, gt_offsets)
            
            # Physics Loss (L1 on bSSFP simulated image vs real MRI)
            pred_img = self.bssfp_sim(pred_offsets, condition)
            loss_G_phys = self.criterion_L1(pred_img, gt_mri)
            
            loss_G = loss_G_adv + (10.0 * loss_G_prop) + (self.config.lambda_physics * loss_G_phys)
            
            self.accelerator.backward(loss_G)
            self.opt_G.step()

        return {
            "loss_D": loss_D.item(),
            "loss_G": loss_G.item(),
            "loss_G_adv": loss_G_adv.item(),
            "loss_G_prop": loss_G_prop.item(),
            "loss_G_phys": loss_G_phys.item(),
            "current_step": self.global_step,
            "current_epoch": self.global_step // len(self.train_loader)
        }

    def train(self):
        """Loop principale di training."""
        logger.info("Starting training...")
        
        target_batch_idx = len(self.val_loader) // 2 # choose a "middle batch" for validation
        self.fixed_val_batch = 0
        for i, batch in enumerate(self.val_loader):
            if i == target_batch_idx:
                self.fixed_val_batch = batch
                break


        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch}")
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                self.accelerator.log(metrics, step=self.global_step)
                progress_bar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})
                
                self.global_step += 1

                if step > 30:
                    break
            
            if self.accelerator.is_main_process and (epoch % self.config.log_image_epochs == 0 or epoch == self.config.num_epochs - 1):
                self._log_images(self.fixed_val_batch, epoch)
            
            self.save_checkpoint(epoch)

    @torch.no_grad()
    def _log_images(self, batch, epoch, n_images=8):
        """Genera e invia immagini di recap a TensorBoard."""
        if n_images > batch['input_label'].shape[0]:
            n_images = batch['input_label'].shape[0]

        self.gen.eval()
        condition = batch['input_label'].float()[:n_images]
        gt_mri = batch['mri_slice'][:n_images]
        
        if self.config.gen_model.in_ch > 1:
            condition_gen = F.one_hot(condition.squeeze(1).long(), num_classes=self.config.gen_model.in_ch).permute(0, 3, 1, 2).float().contiguous() # (B, 1, H, W) -> (B, C, H, W)

        unwrapped_gen = self.accelerator.unwrap_model(self.gen)
        unwrapped_sim = self.accelerator.unwrap_model(self.bssfp_sim)

        pred_offsets = unwrapped_gen(condition_gen)
        pred_img = unwrapped_sim(pred_offsets, condition) # (B, H, W)
        
        cond_vis = condition.unsqueeze(1) / float(self.config.gen_model.in_ch - 1)
        gt_mri_vis = gt_mri.unsqueeze(1) / (gt_mri.max() + 1e-8)
        pred_img_vis = pred_img / (pred_img.max() + 1e-8)
        
        grid = vutils.make_grid(torch.cat([cond_vis, gt_mri_vis, pred_img_vis], dim=0), nrow=4, normalize=False)
        
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_image("Val/Condition_RealMRI_FakeMRI", grid, epoch)
        
        saving_dir = os.path.join(self.config.run_dir, f"images")
        os.makedirs(saving_dir, exist_ok=True)
        saving_path = os.path.join(saving_dir, f"val_images_epoch_{epoch}.png")
        vutils.save_image(grid, saving_path)

    def save_checkpoint(self, epoch):
        """Saves model weights"""
        save_dir = os.path.join(self.config.run_dir, f"checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        
        unwrapped_gen = self.accelerator.unwrap_model(self.gen)
        unwrapped_discr = self.accelerator.unwrap_model(self.discr)
        
        torch.save(unwrapped_gen.state_dict(), os.path.join(save_dir, "generator.pth"))
        torch.save(unwrapped_discr.state_dict(), os.path.join(save_dir, "discriminator.pth"))
        

def main():
    config = GANTrainerConfig()
    set_reproducibility(config.seed)
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.run_dir
    )

    if accelerator.is_main_process:
        os.makedirs(config.run_dir, exist_ok=True)
        setup_logger(config.run_dir)
        accelerator.init_trackers("tb_tracker")
    
    dataset_config_path = os.path.join(config.data_path, "dataset_config.json")
    config_properties_path = os.path.join(config.data_path, "config_properties.json")
    if os.path.exists(dataset_config_path) and os.path.exists(config_properties_path):
        with open(dataset_config_path, "r") as f:
            dataset_config = json.load(f)
        with open(config_properties_path) as f:
            properties_config = json.load(f)
    else:
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path} or {config_properties_path}")

    all_files = sorted(list(set([f.replace('_props.npy', '') for f in os.listdir(config.data_path) if f.endswith('props.npy')])))
    train_files, val_files = train_test_split(all_files, test_size=config.val_fraction, random_state=config.seed)
    
    if accelerator.is_main_process:
        with open(f"{config.run_dir}/train_val_split.json", "w") as f:
            json.dump({'train_indices': train_files, 'val_indices': val_files}, f, indent=4)
        with open(f"{config.run_dir}/grouping_rules_and_labels.json", "w") as f:
            json.dump({'grouping_rules': GROUPING_RULES, 'new_labels': NEW_LABELS}, f, indent=4)
        with open(f"{config.run_dir}/training_config.json", "w") as f:
            json.dump({**asdict(config), **dataset_config, **properties_config}, f, indent=4)

    train_dataset = CustomDatasetTexturizer(data_path=config.data_path, file_list=train_files)
    val_dataset = CustomDatasetTexturizer(data_path=config.data_path, file_list=val_files)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_per_gpu, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size_per_gpu, shuffle=False, num_workers=config.num_workers)

    gen = GeneratorModel(**asdict(config.gen_model)) # input channels: 1 or 22
    discr = DiscriminatorModel(**asdict(config.discr_model)) # 4 channels: (1 condition + 3 offset)

    bssfp_sim = bSSFPSimulator(config.bssfp_model)
    
    opt_G = torch.optim.Adam(gen.parameters(), lr=config.lr_gen)
    opt_D = torch.optim.Adam(discr.parameters(), lr=config.lr_discr)
    
    gen, discr, opt_G, opt_D, train_loader, val_dataloader, bssfp_sim = accelerator.prepare(
        gen, discr, opt_G, opt_D, train_loader, val_dataloader, bssfp_sim
    )
    
    trainer = GANTrainer(
        config=config,
        gen=gen,
        discr=discr,
        bssfp_sim=bssfp_sim,
        opt_G=opt_G,
        opt_D=opt_D,
        train_loader=train_loader,
        val_loader=val_dataloader,
        accelerator=accelerator
    )
    
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # logger.exception cattura automaticamente il Traceback e lo scrive nel log file!
        logger.exception("Il training è crashato con il seguente errore:")
        sys.exit(1)
    
    # config = GANTrainerConfig()

    # accelerator = Accelerator(mixed_precision=config.mixed_precision)

    # print(f"Starting GAN training with config: {asdict(config)}")

    # dataset = CustomDatasetTexturizer(data_path=config.data_path)
    # print(f"Dataset loaded with {len(dataset)} samples")
    # sample = dataset[120]
    # print(f"Sample labels shape: {sample['input_label'].shape}, unique labels: {torch.unique(sample['input_label'])}")
    # print(f"Sample mri slice shape: {sample['mri_slice'].shape}. Value range: [{torch.min(sample['mri_slice'])}, {torch.max(sample['mri_slice'])}]")
    # print(f"Sample props slice shape: {sample['props_slice'].shape}. Value range: [{torch.min(sample['props_slice'])}, {torch.max(sample['props_slice'])}]")

    # # simulate a forward pass through the bSSFP simulator to check if it runs without errors
    # bssfp_simulator = bSSFPSimulator(BSSFPConfig())
    # # bssfp_simulator = accelerator.prepare(bssfp_simulator)
    # sample_input = sample['input_label'].unsqueeze(0).unsqueeze(0).float() # (1, 1, H, W)
    # sample_props = sample['props_slice'].unsqueeze(0).float() # (1, 3, H, W)

    # # get offsets from absolute properties
    # offsets = bssfp_simulator.get_offsets_from_absolute(sample_props, sample_input)
    # print(f"Offsets shape: {offsets.shape}, value range: [{torch.min(offsets)}, {torch.max(offsets)}]")

    # # simulate the bSSFP signal
    # simulated_slice = bssfp_simulator(offsets, sample_input)
    # print(f"Simulated slice shape: {simulated_slice.shape}, value range: [{torch.min(simulated_slice)}, {torch.max(simulated_slice)}]")

    # # save the simulated slice as an image for visual inspection compared to the real MRI slice
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(sample['mri_slice'], cmap='gray')
    # plt.title("Real MRI Slice")
    # plt.subplot(1, 2, 2)
    # plt.imshow(simulated_slice.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')
    # plt.title("Simulated bSSFP Slice")
    
    # os.makedirs("test", exist_ok=True)
    # plt.savefig(os.path.join("test", "bssfp_simulation_check.png"))
    # print(f"Saved bSSFP simulation check image")

    # # simulate single training step
    # gen = GeneratorModel(**asdict(config.gen_model))
    # discr = DiscriminatorModel(**asdict(config.discr_model))
    # opt_G = torch.optim.Adam(gen.parameters(), lr=config.lr_gen)
    # opt_D = torch.optim.Adam(discr.parameters(), lr=config.lr_discr)
    # trainer = GANTrainer(config, gen, discr, bssfp_simulator, opt_G, opt_D, None, None, accelerator)
    # metrics = trainer.train_step(sample)
    # print(f'train step completed: {metrics}')

