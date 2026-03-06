import os
import sys
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, asdict
from tqdm import tqdm
import json

from mt_DIDC_config import GROUPING_RULES, PROPERTY_KEY, LABEL2LABEL, NEW_LABELS

from utils import load_original_labels, setup_logger, set_reproducibility


@dataclass
class Config:
    data_dir: str = './DIDC_multiclass_coro_v2_prep_2'
    out_dir: str = './DIDC_multiclass_coro_v2_prep_2'
    
    epochs: int = 500
    lr: float = 0.001
    patience: int = 50
    min_delta: float = 1e-4

    PD_max: float = 200.0
    T1_max: float = 2000.0
    T2_max: float = 500.0
    upsample_factor: int = 1

    save_hd_images: bool = False
    slices_first: bool = True
    mapping_from_old_labels: bool = False # Whether to use the label2label mapping derived from the old grouping rules (if False, it will use the provided LABEL2LABEL mapping directly)

    seed: int = 187

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class offsetNet(nn.Module):
    def __init__(self, target_size=(384, 384)):
        super(offsetNet, self).__init__()
        self.offsets = nn.Parameter(torch.zeros(1, 3, target_size[0], target_size[1])) 
        self.scale_param = nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.offsets, torch.tanh(self.scale_param)


class PropertyGenerator:
    def __init__(self, 
                 device=None, 
                 epochs: int = 500, 
                 lr: float = 0.001, 
                 patience: int = 50, 
                 min_delta: float = 1e-4, 
                 properties_key: dict = None, 
                 label2label: dict = None, 
                 label2idx: list = None, 
                 label2label_old: dict = None,
                 pat_id: str = None,
                 save_images: bool = False,
                 slices_first: bool = True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.epochs = epochs
        self.lr = lr

        self.PD_max, self.T1_max, self.T2_max = 200., 2000., 500.

        self.TR = 3.0
        self.TE = 1.5
        self.flip_angle = 0.6

        self.patience = patience
        self.min_delta = min_delta
        
        self.properties_keys = properties_key
        self.label2idx = label2idx
        
        if label2label_old:
            self.label2label = {old_label: label2label[new_label] for old_label, new_label in label2label_old.items()} # mapping from old labels to the final labels (skipping the "new" 22 label step)
        else:
            self.label2label = label2label

        assert set(self.label2label.keys()) == set(label2idx), "All keys in label2label must be present in label2idx"

        self.pat_id = pat_id
        self.save_images = save_images
        self.slices_first = slices_first

    def _initialize_tissue_maps(self, mask):
        """Creates initial tissue starting from the mask"""
        h, w = mask.shape
        init_prop = np.zeros((1, 3, h, w), dtype=np.float32)
        
        for old_label, new_label in self.label2label.items():
            ix, iy = np.where(mask == self.label2idx.index(old_label))
            if len(ix) > 0:
                init_prop[0, 0, ix, iy] = self.properties_keys[new_label][0] / self.PD_max
                init_prop[0, 1, ix, iy] = self.properties_keys[new_label][1] / self.T1_max
                init_prop[0, 2, ix, iy] = self.properties_keys[new_label][2] / self.T2_max
                
        return torch.tensor(init_prop, dtype=torch.float32, device=self.device)

    def _bssfp_signal_model(self, offsets, init_values):
        """bSSFP physical simulator"""
        # Sums offsets with initial values, clamps to range, and rescales to physical units
        PD = torch.clamp(offsets[:, 0, ...] + init_values[:, 0, ...], 0.001, 50) * self.PD_max
        T1 = torch.clamp(offsets[:, 1, ...] + init_values[:, 1, ...], 0.001, 50) * self.T1_max
        T2 = torch.clamp(offsets[:, 2, ...] + init_values[:, 2, ...], 0.001, 50) * self.T2_max
        
        num = PD * np.sin(self.flip_angle)
        den = (T1 / T2 + 1) - np.cos(self.flip_angle) * (T1 / T2 - 1)
        decay = torch.exp(-self.TE / T2)
        
        m = torch.abs((num / den) * decay) * 0.05
        return m

    def _loss_fn(self, target_img, predicted_img, offset):
        """Error computation with L2 regularization on tissue property offsets."""
        PD_offset, T1_offset, T2_offset = offset[:, 0, ...], offset[:, 1, ...], offset[:, 2, ...]
        
        recon_error = torch.mean((predicted_img - target_img)**2)
        # L2 reg
        tissue_reg = torch.mean(PD_offset**2) * 10000 + 0.1 * torch.mean(T1_offset**2) + 0.01 * torch.mean(T2_offset**2)
        
        return recon_error + (tissue_reg * 0.001)

    def fit_slice(self, mri_slice, label_slice, slice_idx, total_slices):
        """Actual optimization for a single 2D slice."""
        target_size = mri_slice.shape
        target_image = torch.tensor(mri_slice, dtype=torch.float32, device=self.device)
        init_tissue_properties = self._initialize_tissue_maps(label_slice)
        
        # Optimization setup
        onet = offsetNet(target_size=target_size).to(self.device)
        optimizer = optim.Adam(onet.parameters(), lr=self.lr)

        onet.train()
        epoch_iter = tqdm(range(self.epochs), desc=f'Fitting Slice {slice_idx}/{total_slices}', leave=False)
        
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in epoch_iter:
            optimizer.zero_grad()
            
            tissue_offsets, scale_mag = onet()
            predicted_img = self._bssfp_signal_model(tissue_offsets, init_tissue_properties) * (1 + scale_mag)
            
            loss = self._loss_fn(target_image, predicted_img, tissue_offsets)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss - self.min_delta:
                best_loss = loss.item()
                epochs_no_improve = 0
                best_offsets = tissue_offsets
                best_scale = scale_mag
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= self.patience:
                tissue_offsets = best_offsets
                scale_mag = best_scale
                break
            
            if epoch % 50 == 0:
                epoch_iter.set_postfix_str("loss=%.4g" % loss.item())
                
        final_offsets = tissue_offsets.detach()
        final_scale = scale_mag.detach()
        
        final_props = torch.zeros_like(final_offsets)
        final_props[0, 0] = torch.clamp(final_offsets[0, 0] + init_tissue_properties[0, 0], 0.001, 50) * self.PD_max
        final_props[0, 1] = torch.clamp(final_offsets[0, 1] + init_tissue_properties[0, 1], 0.001, 50) * self.T1_max
        final_props[0, 2] = torch.clamp(final_offsets[0, 2] + init_tissue_properties[0, 2], 0.001, 50) * self.T2_max
        
        return final_props[0].cpu().numpy(), predicted_img[0].detach().cpu().numpy(), init_tissue_properties[0].cpu().numpy(), final_scale.detach().cpu().numpy(), best_loss, epoch

    def process_volume(self, mri_volume, labels_volume, out_path, upsample_factor=4):
        """Handles the entire 3D volume: Upsampling, Fitting slice by slice, and Saving."""
        mri_hd = scipy.ndimage.zoom(mri_volume, (1, upsample_factor, upsample_factor), order=1)
        labels_hd = scipy.ndimage.zoom(labels_volume, (1, upsample_factor, upsample_factor), order=0)
        
        slices, h, w = mri_hd.shape

        if self.slices_first:
            tissue_props = np.zeros((slices, 3, h, w), dtype=np.float32)
        else:
            tissue_props = np.zeros((3, h, w, slices), dtype=np.float32)
        
        os.makedirs(os.path.join(out_path, f'simulated_images/{self.pat_id}/'), exist_ok=True)

        avg_pat_loss = 0.0
        for s in range(slices):
            # Executes optimization
            props, pred_img, init_props, scale, loss, epoch = self.fit_slice(mri_hd[s], labels_hd[s], slice_idx=s, total_slices=slices)
            avg_pat_loss += loss

            if self.slices_first:
                tissue_props[s] = props
            else:
                tissue_props[..., s] = props
            
            if s == 120 or s == 180:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(mri_hd[s], cmap='gray')
                ax[0].set_title("Real MRI")
                ax[1].imshow(pred_img, cmap='gray')
                ax[1].set_title("Simulated bSSFP")
                plt.savefig(os.path.join(out_path, f'simulated_images/{self.pat_id}/MRI_{s:05d}.png'), dpi=150)
                plt.close()
        
        avg_pat_loss /= slices
        logging.info(f"Patient {self.pat_id}: Average Slice Loss = {avg_pat_loss:.4f}")

        if self.save_images:
            np.save(os.path.join(out_path, f'{self.pat_id}_MRI_matrix_HD.npy'), mri_hd)
            np.save(os.path.join(out_path, f'{self.pat_id}_labels_matrix_HD.npy'), labels_hd)
        np.save(os.path.join(out_path, f'{self.pat_id}_props.npy'), tissue_props) # shape (3, H, W, S)

def main():
    config = Config()

    set_reproducibility(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)
    setup_logger(config.out_dir, filename="computation_properties.log")

    with open(os.path.join(config.out_dir, 'config_properties.json'), 'w') as f:
        json.dump(asdict(config), f, indent=4)

    logging.info(f"Selected device: {config.device}")

    if config.mapping_from_old_labels:
        label2idx = load_original_labels(config.data_dir)
    else:
        label2idx = NEW_LABELS

    files = sorted([f for f in os.listdir(config.data_dir) if f.endswith('fg.npy')])

    pbar = tqdm(files, leave=True, desc="Processing Patients")
    for file in pbar:
        pat_id = file.replace('_fg.npy', '')
        pbar.set_description(f"Processing {pat_id}")
        
        path_mask = os.path.join(config.data_dir, pat_id + '_mask.npy')
        path_img = os.path.join(config.data_dir, pat_id + '_img.npy')

        labels_volume = np.load(path_mask)
        mri_volume = np.load(path_img)

        mri_volume = mri_volume / np.max(mri_volume) # Normalize to [0, 1] for better optimization stability
        
        if config.mapping_from_old_labels:
            label2label_old = GROUPING_RULES
        else:            
            label2label_old = None
        
        generator = PropertyGenerator(epochs=config.epochs, 
                                      lr=config.lr, 
                                      patience=config.patience, 
                                      min_delta=config.min_delta, 
                                      properties_key=PROPERTY_KEY, 
                                      label2label=LABEL2LABEL, 
                                      label2idx=label2idx, 
                                      label2label_old=label2label_old, 
                                      device=config.device,
                                      save_images=config.save_hd_images,
                                      pat_id=pat_id,
                                      slices_first=config.slices_first)
        generator.process_volume(mri_volume, labels_volume, config.out_dir, upsample_factor=config.upsample_factor)


if __name__ == "__main__":
    main()