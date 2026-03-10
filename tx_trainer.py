import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.utils.data._utils.collate import default_collate

import logging
from accelerate import Accelerator

import os
from tqdm import tqdm
from tx_config import GANTrainerConfig, UnetTrainerConfig
from tx_bssfps_simulator import bSSFPSimulator

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
        accelerator: Accelerator,
        logger: logging.Logger
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
        self.evaluator = ImageQualityEvaluator(device=self.accelerator.device)
        self.logger = logger
        self.logger.info("GANTrainer initialized correctly")

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

            psnr, ssim = self.evaluator.evaluate(pred_img, gt_mri)

        return {
            "Train/loss_D": loss_D.item(),
            "Train/loss_G": loss_G.item(),
            "Train/loss_G_adv": loss_G_adv.item(),
            "Train/loss_G_prop": loss_G_prop.item(),
            "Train/loss_G_phys": loss_G_phys.item(),
            "Train/current_epoch": self.global_step // len(self.train_loader),
            "Train/SSIM": ssim,
            "Train/PSNR": psnr
        }

    @torch.no_grad()
    def validate(self, epoch):
        
        self.gen.eval()
        self.discr.eval()
        
        val_metrics = {
            "loss_D": 0.0,
            "loss_G": 0.0,
            "loss_G_adv": 0.0, 
            "loss_G_prop": 0.0, 
            "loss_G_phys": 0.0, 
            "SSIM": 0.0, 
            "PSNR": 0.0
        }
        
        progress_bar = tqdm(self.val_loader, desc=f"Validation Ep {epoch}", leave=False, disable=not self.accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            original_mask = batch['input_label'].unsqueeze(1).float() 
            condition = original_mask.clone()
            if self.config.gen_model.in_ch > 1:
                condition = F.one_hot(original_mask.squeeze(1).long(), num_classes=self.config.gen_model.in_ch).permute(0, 3, 1, 2).float().contiguous() 

            absolute_gt_props = batch['props_slice'].float() 
            gt_mri = batch['mri_slice'].unsqueeze(1).float()
            gt_offsets = self.bssfp_sim.get_offsets_from_absolute(absolute_gt_props, original_mask) 

            pred_offsets = self.gen(condition)
            
            pred_img = self.bssfp_sim(pred_offsets, original_mask) 
            if pred_img.ndim == 3:
                pred_img = pred_img.unsqueeze(1)
                
            real_pair = torch.cat([condition, gt_offsets], dim=1)
            fake_pair = torch.cat([condition, pred_offsets], dim=1)
            
            combined_pair = torch.cat([real_pair, fake_pair], dim=0)
            d_combined = self.discr(combined_pair)
            d_real, d_fake = torch.split(d_combined, real_pair.shape[0])
            
            loss_D_real = self.criterion_GAN(d_real, torch.ones_like(d_real))
            loss_D_fake = self.criterion_GAN(d_fake, torch.zeros_like(d_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            loss_G_adv = self.criterion_GAN(d_fake, torch.ones_like(d_fake))
            loss_G_prop = self.criterion_L1(pred_offsets, gt_offsets)
            loss_G_phys = self.criterion_L1(pred_img, gt_mri)
            
            loss_G = loss_G_adv + (self.config.lambda_properties * loss_G_prop) + (self.config.lambda_physics * loss_G_phys)
            
            ssim_val, psnr_val = self.evaluator.evaluate(pred_img, gt_mri)
            
            val_metrics["loss_D"] += loss_D.item()
            val_metrics["loss_G"] += loss_G.item()
            val_metrics["loss_G_adv"] += loss_G_adv.item()
            val_metrics["loss_G_prop"] += loss_G_prop.item()
            val_metrics["loss_G_phys"] += loss_G_phys.item()
            val_metrics["SSIM"] += ssim_val
            val_metrics["PSNR"] += psnr_val

        num_batches = len(self.val_loader)

        final_metrics = {f"Val/{k}": v / num_batches for k, v in val_metrics.items()}
        
        self.accelerator.log(final_metrics, step=self.global_step)
        self.logger.info(f"Val Epoch {epoch} Results: " + ", ".join([f"{k}={v:.4f}" for k, v in final_metrics.items()]))
        
        self.gen.train()
        self.discr.train()
        
        return final_metrics

    def train(self):
        self.logger.info("Extracting the fixed validation batch for logging...")
        
        # target_batch_idx = len(self.val_loader) // 2+70 # choose a "middle batch" for validation
        target_batch_idx = 360
        self.fixed_val_batch = 0
        for i, batch in enumerate(self.val_loader):
            if i == target_batch_idx:
                self.fixed_val_batch = batch
                break
        
        self.logger.info(f"Using fixed validation batch number {i}")
        self.logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch}")
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                self.accelerator.log(metrics, step=self.global_step)
                progress_bar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})
                
                self.global_step += 1

            self.logger.info(f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
            if self.accelerator.is_main_process and (epoch % self.config.log_image_epochs == 0 or epoch == self.config.num_epochs - 1):
                self._log_images(self.fixed_val_batch, epoch)
            
            if self.accelerator.is_main_process:
                self.save_checkpoint(epoch)

    @torch.no_grad()
    def _log_images(self, batch, epoch, n_images=8):
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
        save_dir = os.path.join(self.config.run_dir, f"checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        
        unwrapped_gen = self.accelerator.unwrap_model(self.gen)
        unwrapped_discr = self.accelerator.unwrap_model(self.discr)
        
        torch.save(unwrapped_gen.state_dict(), os.path.join(save_dir, "generator.pth"))
        torch.save(unwrapped_discr.state_dict(), os.path.join(save_dir, "discriminator.pth"))


class UnetTrainer: 
    def __init__(
        self, 
        config: UnetTrainerConfig, 
        model: nn.Module, 
        bssfp_sim: bSSFPSimulator, 
        opt: torch.optim.Optimizer, 
        train_loader, 
        val_loader, 
        accelerator: Accelerator,
        logger: logging.Logger
    ):
        self.config = config
        self.model = model
        self.bssfp_sim = bssfp_sim
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.logger = logger
        self.criterion_L1 = nn.L1Loss()
        
        self.global_step = 0
        self.evaluator = ImageQualityEvaluator(device=self.accelerator.device)

        self.logger.info("UnetTrainer initialized correctly")

    def train_step(self, batch):
        original_mask = batch['input_label'].unsqueeze(1).float() 

        condition = original_mask.clone()
        if self.config.gen_model.in_ch > 1:
            condition = F.one_hot(original_mask.squeeze(1).long(), num_classes=self.config.gen_model.in_ch).permute(0, 3, 1, 2).float().contiguous() 

        absolute_gt_props = batch['props_slice'].float() 
        gt_mri = batch['mri_slice'].unsqueeze(1).float()

        gt_offsets = self.bssfp_sim.get_offsets_from_absolute(absolute_gt_props, condition) 

        self.model.train()
        with self.accelerator.accumulate(self.model):
            self.opt.zero_grad()

            pred_offsets = self.model(condition)
            
            loss_prop = self.criterion_L1(pred_offsets, gt_offsets)

            pred_img = self.bssfp_sim(pred_offsets, condition)
            if pred_img.ndim == 3:
                pred_img = pred_img.unsqueeze(1)
            
            loss_phys = self.criterion_L1(pred_img, gt_mri)

            loss_total = (self.config.lambda_prop * loss_prop) + (self.config.lambda_physics * loss_phys)

            self.accelerator.backward(loss_total)
            self.opt.step()

            psnr, ssim = self.evaluator.evaluate(pred_img, gt_mri)

        return {
            "Train/loss_total": loss_total.item(),
            "Train/loss_prop": loss_prop.item(),
            "Train/loss_phys": loss_phys.item(),
            "Train/SSIM": ssim,
            "Train/PSNR": psnr,
            "Train/current_epoch": self.global_step // len(self.train_loader)
        }

    @torch.no_grad()
    def validate(self, epoch):        
        self.model.eval()
        
        val_loss_total = 0.0
        val_loss_prop = 0.0
        val_loss_phys = 0.0
        val_ssim = 0.0
        val_psnr = 0.0
        
        progress_bar = tqdm(self.val_loader, desc=f"Validation Ep {epoch}", leave=False, disable=not self.accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            original_mask = batch['input_label'].unsqueeze(1).float() 
            condition = original_mask.clone()
            if self.config.gen_model.in_ch > 1:
                condition = F.one_hot(original_mask.squeeze(1).long(), num_classes=self.config.gen_model.in_ch).permute(0, 3, 1, 2).float().contiguous() 

            absolute_gt_props = batch['props_slice'].float() 
            gt_mri = batch['mri_slice'].unsqueeze(1).float()

            gt_offsets = self.bssfp_sim.get_offsets_from_absolute(absolute_gt_props, condition) 

            pred_offsets = self.model(condition)
            loss_prop = self.criterion_L1(pred_offsets, gt_offsets)

            pred_img = self.bssfp_sim(pred_offsets, condition)
            if pred_img.ndim == 3:
                pred_img = pred_img.unsqueeze(1)
            
            loss_phys = self.criterion_L1(pred_img, gt_mri)

            loss_total = (self.config.lambda_prop * loss_prop) + (self.config.lambda_physics * loss_phys)
            
            ssim_val, psnr_val = self.evaluator.evaluate(pred_img, gt_mri)
            
            val_loss_total += loss_total.item()
            val_loss_prop += loss_prop.item()
            val_loss_phys += loss_phys.item()
            val_ssim += ssim_val
            val_psnr += psnr_val

        num_batches = len(self.val_loader)
        metrics = {
            "Val/loss_total": val_loss_total / num_batches,
            "Val/loss_prop": val_loss_prop / num_batches,
            "Val/loss_phys": val_loss_phys / num_batches,
            "Val/SSIM": val_ssim / num_batches,
            "Val/PSNR": val_psnr / num_batches,
        }
        
        self.accelerator.log(metrics, step=self.global_step)
        self.logger.info(f"Val Epoch {epoch} Results: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

        self.model.train()
        
        return metrics

    def train(self):
        self.logger.info("Extracting the fixed validation batch for logging...")
        
        # target_batch_idx = len(self.val_loader) // 2-65 # choose a "middle batch" for validation
        target_batch_idx = 837
        self.fixed_val_batch = 0
        for i, batch in enumerate(self.val_loader):
            if i == target_batch_idx:
                self.fixed_val_batch = batch
                break
        self.logger.info(f"Using fixed validation batch number {i}")
        
        self.logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch}")
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                self.accelerator.log(metrics, step=self.global_step)
                progress_bar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})
                self.global_step += 1

            self.logger.info(f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
            
            self.validate(epoch)

            if self.accelerator.is_main_process and (epoch % self.config.log_image_epochs == 0 or epoch == self.config.num_epochs - 1):
                self._log_images(self.fixed_val_batch, epoch)
            
            if self.accelerator.is_main_process:
                self.save_checkpoint(epoch)
                
            self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _log_images(self, batch, epoch, n_images=8):
        if n_images > batch['input_label'].shape[0]:
            n_images = batch['input_label'].shape[0]

        condition = batch['input_label'].float()[:n_images].to(self.accelerator.device)
        gt_mri = batch['mri_slice'][:n_images].to(self.accelerator.device)
        
        if self.config.gen_model.in_ch > 1:
            condition_gen = F.one_hot(condition.squeeze(1).long(), num_classes=self.config.gen_model.in_ch).permute(0, 3, 1, 2).float().contiguous()

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_sim = self.accelerator.unwrap_model(self.bssfp_sim)
        unwrapped_model.eval()

        pred_offsets = unwrapped_model(condition_gen)
        pred_img = unwrapped_sim(pred_offsets, condition) 
        
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
        save_dir = os.path.join(self.config.run_dir, f"checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "unet.pth"))


class ImageQualityEvaluator:
    def __init__(self, device="cuda"):
        self.device = device

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

    @torch.no_grad()
    def evaluate(self, pred_img: torch.Tensor, gt_mri: torch.Tensor):
        """
        Computes SSIM and PSNR between the predicted image and the ground truth MRI.
        Both tensors must be shaped (B, C, H, W).
        """
        pred_detached = pred_img.detach()
        gt_detached = gt_mri.detach()

        # Normalization
        pred_max = pred_detached.amax(dim=(2, 3), keepdim=True) + 1e-8
        gt_max = gt_detached.amax(dim=(2, 3), keepdim=True) + 1e-8
        
        pred_norm = pred_detached / pred_max
        gt_norm = gt_detached / gt_max
        
        # Clamping to [0, 1] to ensure valid range for metrics
        pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
        gt_norm = torch.clamp(gt_norm, 0.0, 1.0)

        ssim_val = self.ssim(pred_norm, gt_norm)
        psnr_val = self.psnr(pred_norm, gt_norm)

        return ssim_val.item(), psnr_val.item()
    

