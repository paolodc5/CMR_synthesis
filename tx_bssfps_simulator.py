import torch.nn as nn
from tx_config import BSSFPConfig
import math
import torch


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