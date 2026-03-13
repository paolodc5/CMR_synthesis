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
    data_dir: str = './DIDC_multiclass_coro_v2_prep'
    out_dir: str = './DIDC_multiclass_coro_v2_prep'
    
    epochs: int = 1500
    lr: float = 0.001
    patience: int = 50
    min_delta: float = 1e-4

    PD_max: float = 200.0
    T1_max: float = 2000.0
    T2_max: float = 500.0

    TR: float = 3.0
    TE: float = 1.5
    flip_angle: float = 0.6
    
    upsample_factor: int = 1

    lambda_reg: float = 0.01
    lambda_pd: float = 10000
    lambda_t1: float = 1
    lambda_t2: float = 0.1

    force_restart: bool = True  # If true, overrides existing results in out_dir
    save_normalized: bool = True

    batch_size: int = 256

    save_hd_images: bool = False
    slices_first: bool = True
    mapping_from_old_labels: bool = False # Whether to use the label2label mapping derived from the old grouping rules (if False, it will use the provided LABEL2LABEL mapping directly)

    seed: int = 187

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class offsetNet(nn.Module):
    def __init__(self, batch_size, target_size=(384, 384)):
        super(offsetNet, self).__init__()
        self.offsets = nn.Parameter(torch.zeros(batch_size, 3, target_size[0], target_size[1])) 
        self.scale_param = nn.Parameter(torch.zeros(batch_size, 1, 1))

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
                 slices_first: bool = True,
                 config: Config = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.epochs = epochs
        self.lr = lr

        self.PD_max, self.T1_max, self.T2_max = config.PD_max, config.T1_max, config.T2_max

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

        self.config = config

    def _initialize_tissue_maps(self, mask):
        """Creates initial tissue starting from the mask"""
        b, h, w = mask.shape
        init_prop = np.zeros((b, 3, h, w), dtype=np.float32)
        
        for old_label, new_label in self.label2label.items():
            label_idx = self.label2idx.index(old_label)
            ix = (mask == label_idx)            
            if ix.any():
                init_prop[:, 0][ix] = self.properties_keys[new_label][0] / self.PD_max
                init_prop[:, 1][ix] = self.properties_keys[new_label][1] / self.T1_max
                init_prop[:, 2][ix] = self.properties_keys[new_label][2] / self.T2_max
                
        return torch.tensor(init_prop, dtype=torch.float32, device=self.device)

    def _bssfp_signal_model(self, offsets, init_values):
        """bSSFP physical simulator"""
        PD = torch.clamp(offsets[:, 0, ...] + init_values[:, 0, ...], 0.001, 30)*self.PD_max
        T1 = torch.clamp(offsets[:, 1, ...] + init_values[:, 1, ...], 0.001, 30)*self.T1_max
        T2 = torch.clamp(offsets[:, 2, ...] + init_values[:, 2, ...], 0.001, 30)*self.T2_max

        num = PD * np.sin(self.flip_angle)
        den = (T1 / T2 + 1) - np.cos(self.flip_angle) * (T1 / T2 - 1)
        decay = torch.exp(-self.TE / T2)
        
        m = torch.abs((num / den) * decay) * 0.05
        return m

    def _loss_fn(self, target_img, predicted_img, offset):
        """Error computation with L2 regularization on tissue property offsets."""
        PD_offset, T1_offset, T2_offset = offset[:, 0, ...], offset[:, 1, ...], offset[:, 2, ...]
        
        recon_error = torch.mean((predicted_img - target_img)**2, dim=(1,2))
        # L2 reg
        tissue_reg = (torch.mean(PD_offset**2, dim=(1, 2)) * self.config.lambda_pd + 
                      torch.mean(T1_offset**2, dim=(1, 2)) * self.config.lambda_t1 + 
                      torch.mean(T2_offset**2, dim=(1, 2)) * self.config.lambda_t2)
        
        return recon_error + (tissue_reg * self.config.lambda_reg)

    def fit_batch(self, mri_batch, label_batch, start_idx, end_idx):
        b, h, w = mri_batch.shape
        target_image = torch.tensor(mri_batch, dtype=torch.float32, device=self.device)
        init_tissue_properties = self._initialize_tissue_maps(label_batch)
        
        onet = offsetNet(batch_size=b, target_size=(h, w)).to(self.device)
        optimizer = optim.Adam(onet.parameters(), lr=self.lr)

        onet.train()
        epoch_iter = tqdm(range(self.epochs), desc=f'Fitting Slices {start_idx}-{end_idx}', leave=False)
        
        # Vectorizing the early stopping (to trigger independently on each slice)
        best_loss = torch.full((b,), float('inf'), device=self.device)
        epochs_no_improve = torch.zeros(b, dtype=torch.int32, device=self.device)
        
        best_offsets = torch.zeros_like(onet.offsets)
        best_scale = torch.zeros_like(onet.scale_param)

        for epoch in epoch_iter:
            optimizer.zero_grad()
            
            tissue_offsets, scale_mag = onet()
            predicted_img = self._bssfp_signal_model(tissue_offsets, init_tissue_properties) * (1 + scale_mag)
            
            loss_per_slice = self._loss_fn(target_image, predicted_img, tissue_offsets)
            
            loss_mean = loss_per_slice.mean()
            loss_mean.backward()
            optimizer.step()

            #  independent update logic
            improved_mask = loss_per_slice < (best_loss - self.min_delta)
            
            if improved_mask.any():
                best_loss[improved_mask] = loss_per_slice[improved_mask].detach()
                best_offsets[improved_mask] = tissue_offsets[improved_mask].detach().clone() # saves the best offests and discards the rest
                best_scale[improved_mask] = scale_mag[improved_mask].detach().clone()
            
            epochs_no_improve[improved_mask] = 0
            epochs_no_improve[~improved_mask] += 1
            
            if (epochs_no_improve >= self.patience).all():
                break
            
            if epoch % 50 == 0:
                epoch_iter.set_postfix_str("loss_mean=%.4g" % loss_mean.item())
                
        uninitialized = (best_loss == float('inf'))
        if uninitialized.any():
            best_offsets[uninitialized] = tissue_offsets[uninitialized].detach()
            best_scale[uninitialized] = scale_mag[uninitialized].detach()
            
        final_props = torch.zeros_like(best_offsets)

        if self.config.save_normalized:
            final_props[:, 0] = torch.clamp(best_offsets[:, 0] + init_tissue_properties[:, 0], 0.001, 30) 
            final_props[:, 1] = torch.clamp(best_offsets[:, 1] + init_tissue_properties[:, 1], 0.001, 30) 
            final_props[:, 2] = torch.clamp(best_offsets[:, 2] + init_tissue_properties[:, 2], 0.001, 30)
        else:
            final_props[:, 0] = torch.clamp(best_offsets[:, 0] + init_tissue_properties[:, 0], 0.001, 30) * self.PD_max
            final_props[:, 1] = torch.clamp(best_offsets[:, 1] + init_tissue_properties[:, 1], 0.001, 30) * self.T1_max
            final_props[:, 2] = torch.clamp(best_offsets[:, 2] + init_tissue_properties[:, 2], 0.001, 30) * self.T2_max
        
        final_img = self._bssfp_signal_model(best_offsets, init_tissue_properties) * (1 + best_scale)
        
        # Restituiamo la loss media dei "best" giusto per logging
        return final_props.cpu().numpy(), final_img.detach().cpu().numpy(), init_tissue_properties.cpu().numpy(), best_scale.detach().cpu().numpy(), best_loss.mean().item(), epoch
    
    def process_volume(self, mri_volume, labels_volume, out_path, upsample_factor=4):
        """Handles the entire 3D volume using mini-batches."""
        mri_hd = scipy.ndimage.zoom(mri_volume, (1, upsample_factor, upsample_factor), order=1)
        labels_hd = scipy.ndimage.zoom(labels_volume, (1, upsample_factor, upsample_factor), order=0)
        
        slices, h, w = mri_hd.shape

        if self.slices_first:
            tissue_props = np.zeros((slices, 3, h, w), dtype=np.float32)
        else:
            tissue_props = np.zeros((3, h, w, slices), dtype=np.float32)
        
        os.makedirs(os.path.join(out_path, f'simulated_images/{self.pat_id}/'), exist_ok=True)

        avg_pat_loss = 0.0
        batch_size = self.config.batch_size
        num_batches = int(np.ceil(slices / batch_size))

        for b_idx in range(num_batches):
            # extract the batch
            start_idx = b_idx * batch_size
            end_idx = min((b_idx + 1) * batch_size, slices)
            
            mri_batch = mri_hd[start_idx:end_idx]
            labels_batch = labels_hd[start_idx:end_idx]

            props, pred_img, init_props, scale, loss, epoch = self.fit_batch(
                mri_batch, labels_batch, start_idx, end_idx
            )
            
            avg_pat_loss += loss * (end_idx - start_idx)

            if self.slices_first:
                tissue_props[start_idx:end_idx] = props
            else:
                # (Batch, C, H, W) -> (C, H, W, Batch)
                tissue_props[..., start_idx:end_idx] = props.transpose(1, 2, 3, 0)
            
            for i, absolute_s in enumerate(range(start_idx, end_idx)):
                if absolute_s in [120, 180]:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(mri_batch[i], cmap='gray')
                    ax[0].set_title("Real MRI")
                    ax[1].imshow(pred_img[i], cmap='gray')
                    ax[1].set_title("Simulated bSSFP")
                    plt.savefig(os.path.join(out_path, f'simulated_images/{self.pat_id}/MRI_{absolute_s:05d}.png'), dpi=150)
                    plt.close()
        
        avg_pat_loss /= slices
        logging.info(f"Patient {self.pat_id}: Average Slice Loss = {avg_pat_loss:.4f}")

        if self.save_images:
            np.save(os.path.join(out_path, f'{self.pat_id}_MRI_matrix_HD.npy'), mri_hd.astype(np.float16))
            np.save(os.path.join(out_path, f'{self.pat_id}_labels_matrix_HD.npy'), labels_hd.astype(np.uint8))
            
        np.save(os.path.join(out_path, f'{self.pat_id}_props.npy'), tissue_props.astype(np.float16)) # save in half precision to save space

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

    if not config.force_restart:
        files_to_process = []
        for file in files: 
            pat_id = file.replace('_fg.npy', '')
            props_path = os.path.join(config.out_dir, f'{pat_id}_props.npy')

            if not os.path.exists(props_path):
                files_to_process.append(file)
            
            skipped = len(files) - len(files_to_process)

        if skipped > 0:
            logging.info(f"Skipping {skipped} already processed files. Use force_restart=True to override.")
        else:
            logging.info("No existing processed files found. Starting fresh computation.")
    else:
        logging.info("force_restart=True: All existing processed files will be overridden.")


    pbar = tqdm(files, leave=True, desc="Processing Patients")
    for file in pbar:
        pat_id = file.replace('_fg.npy', '')
        pbar.set_description(f"Processing {pat_id}")
        
        path_mask = os.path.join(config.data_dir, pat_id + '_mask.npy')
        path_img = os.path.join(config.data_dir, pat_id + '_img.npy')

        labels_volume = np.load(path_mask).astype(np.int32)
        mri_volume = np.load(path_img).astype(np.float32)

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
                                      slices_first=config.slices_first, 
                                      config=config)
        generator.process_volume(mri_volume, labels_volume, config.out_dir, upsample_factor=config.upsample_factor)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("The script crashed with the following error:")
        sys.exit(1)
