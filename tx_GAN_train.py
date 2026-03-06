import os
import sys
import json
import logging

import numpy as np
import math

from datetime import datetime
from dataclasses import asdict, dataclass, field
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from datasets import FastDatasetDIDC

from gan_basic import DiscriminatorModel

from accelerate import Accelerator
from accelerate.logging import get_logger

from mt_DIDC_config import LABEL2LABEL, PROPERTY_KEY, NEW_LABELS

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
class GANTrainerConfig:
    run_name: str = ""
    exp_dir: str = "./experiments/DIDCV2_TEXT"
    
    data_path: str = './DIDC_multiclass_coro_v2_prep_2'
    
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

    bssfp_model: BSSFPConfig = field(default_factory=BSSFPConfig)

    def __post_init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_dir = os.path.join(self.exp_dir, f"{timestamp}_{self.run_name}")


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

    def train_step(self, batch):
        """Esegue un singolo step di training per D e G."""
        condition = batch['input_label'].unsqueeze(1).float() # (B, 1, H, W)
        absolute_gt_props = batch['props_slice']              # (B, 3, H, W)
        gt_mri = batch['mri_slice']                           # (B, H, W)

        gt_offsets = self.bssfp_sim.get_offsets_from_absolute(absolute_gt_props, condition)
        
        pred_offsets = self.gen(condition)
        
        # train discriminator
        self.discr.train()
        with self.accelerator.accumulate(self.discr):
            self.opt_D.zero_grad()
            
            # Il Discriminatore valuta la coppia (Condizione, Offsets)
            real_pair = torch.cat([condition, gt_offsets], dim=1)
            fake_pair = torch.cat([condition, pred_offsets.detach()], dim=1)
            
            d_real = self.discr(real_pair)
            d_fake = self.discr(fake_pair)
            
            loss_D_real = self.criterion_GAN(d_real, torch.ones_like(d_real))
            loss_D_fake = self.criterion_GAN(d_fake, torch.zeros_like(d_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            self.accelerator.backward(loss_D)
            self.opt_D.step()

        # train generator
        self.gen.train()
        with self.accelerator.accumulate(self.gen):
            self.opt_G.zero_grad()
            
            fake_pair_for_G = torch.cat([condition, pred_offsets], dim=1)
            d_fake_for_G = self.discr(fake_pair_for_G)
            loss_G_adv = self.criterion_GAN(d_fake_for_G, torch.ones_like(d_fake_for_G))
            
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
            "loss_G_phys": loss_G_phys.item()
        }

    def train(self):
        """Loop principale di training."""
        logger.info("Iniziando il training...")
        
        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch}")
            
            for batch in progress_bar:
                metrics = self.train_step(batch)
                
                self.accelerator.log(metrics, step=self.global_step)
                progress_bar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})
                
                self.global_step += 1
            
            if self.accelerator.is_main_process and (epoch % 5 == 0 or epoch == self.config.num_epochs - 1):
                self._log_images(batch, epoch)
                self.save_checkpoint(epoch)

    @torch.no_grad()
    def _log_images(self, batch, epoch):
        """Genera e invia immagini di recap a TensorBoard."""
        self.gen.eval()
        condition = batch['input_label'].unsqueeze(1).float()[:4]
        gt_mri = batch['mri_slice'][:4].unsqueeze(1)
        
        pred_offsets = self.gen(condition)
        pred_img = self.bssfp_sim(pred_offsets, condition).unsqueeze(1) # (B, 1, H, W)
        
        cond_vis = condition / condition.max() 
        gt_mri_vis = gt_mri / (gt_mri.max() + 1e-8)
        pred_img_vis = pred_img / (pred_img.max() + 1e-8)
        
        grid = vutils.make_grid(torch.cat([cond_vis, gt_mri_vis, pred_img_vis], dim=0), nrow=4, normalize=False)
        
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_image("Val/Condition_RealMRI_FakeMRI", grid, epoch)

    def save_checkpoint(self, epoch):
        """Salva i pesi dei modelli."""
        save_dir = os.path.join(self.config.run_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        unwrapped_gen = self.accelerator.unwrap_model(self.gen)
        unwrapped_discr = self.accelerator.unwrap_model(self.discr)
        
        torch.save(unwrapped_gen.state_dict(), os.path.join(save_dir, "generator.pth"))
        torch.save(unwrapped_discr.state_dict(), os.path.join(save_dir, "discriminator.pth"))
        

def main():
    config = GANTrainerConfig()
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.run_dir
    )
    
    if accelerator.is_main_process:
        os.makedirs(config.run_dir, exist_ok=True)
        accelerator.init_trackers("tb_tracker")

    # ... (Caricamento Dataset, Dataloader) ...
    # train_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    
    # 3. Inizializza i Modelli
    # Qui entra in gioco la modularità. Purché il gen accetti (B, 1, H, W) e dia (B, 3, H, W),
    # puoi passargli una UNet, uno SPADE o un Multiscale senza toccare il Trainer!
    gen = nn.Conv2d(1, 3, 3, padding=1) # Placeholder per il tuo Generatore
    discr = nn.Conv2d(4, 1, 3, padding=1) # Placeholder per il Discriminatore (1 cond + 3 offset)
    
    bssfp_sim = bSSFPSimulator(BSSFPConfig())
    
    opt_G = torch.optim.Adam(gen.parameters(), lr=config.lr_gen)
    opt_D = torch.optim.Adam(discr.parameters(), lr=config.lr_discr)
    
    # 4. Prepara tutto con Accelerate (Device placement, DDP, mixed precision automatico)
    gen, discr, opt_G, opt_D, train_loader, bssfp_sim = accelerator.prepare(
        gen, discr, opt_G, opt_D, train_loader, bssfp_sim
    )
    
    # 5. Lancia il Trainer
    trainer = GANTrainer(
        config=config,
        gen=gen,
        discr=discr,
        bssfp_sim=bssfp_sim,
        opt_G=opt_G,
        opt_D=opt_D,
        train_loader=train_loader,
        val_loader=None, # Aggiungilo quando crei lo split
        accelerator=accelerator
    )
    
    trainer.train()



if __name__ == "__main__":
    config = GANTrainerConfig()

    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    print(f"Starting GAN training with config: {asdict(config)}")

    dataset = CustomDatasetTexturizer(data_path=config.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    sample = dataset[120]
    print(f"Sample labels shape: {sample['input_label'].shape}, unique labels: {torch.unique(sample['input_label'])}")
    print(f"Sample mri slice shape: {sample['mri_slice'].shape}. Value range: [{torch.min(sample['mri_slice'])}, {torch.max(sample['mri_slice'])}]")
    print(f"Sample props slice shape: {sample['props_slice'].shape}. Value range: [{torch.min(sample['props_slice'])}, {torch.max(sample['props_slice'])}]")

    # simulate a forward pass through the bSSFP simulator to check if it runs without errors
    bssfp_simulator = bSSFPSimulator(BSSFPConfig())
    # bssfp_simulator = accelerator.prepare(bssfp_simulator)
    sample_input = sample['input_label'].unsqueeze(0).unsqueeze(0).float() # (1, 1, H, W)
    sample_props = sample['props_slice'].unsqueeze(0).float() # (1, 3, H, W)

    # get offsets from absolute properties
    offsets = bssfp_simulator.get_offsets_from_absolute(sample_props, sample_input)
    print(f"Offsets shape: {offsets.shape}, value range: [{torch.min(offsets)}, {torch.max(offsets)}]")

    # simulate the bSSFP signal
    simulated_slice = bssfp_simulator(offsets, sample_input)
    print(f"Simulated slice shape: {simulated_slice.shape}, value range: [{torch.min(simulated_slice)}, {torch.max(simulated_slice)}]")

    # save the simulated slice as an image for visual inspection compared to the real MRI slice
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample['mri_slice'], cmap='gray')
    plt.title("Real MRI Slice")
    plt.subplot(1, 2, 2)
    plt.imshow(simulated_slice.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')
    plt.title("Simulated bSSFP Slice")
    
    os.makedirs("test", exist_ok=True)
    plt.savefig(os.path.join("test", "bssfp_simulation_check.png"))
    print(f"Saved bSSFP simulation check image")

    # simulate single training step
    # trainer = GANTrainer(config, gen, discr, bssfp_simulator, opt_G, opt_D, None, None, accelerator)
    # metrics = trainer.train_step(sample)

