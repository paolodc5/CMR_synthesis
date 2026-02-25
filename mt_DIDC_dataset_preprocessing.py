import json
import os
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS
from datasets import LazyDatasetDIDC
from tqdm import tqdm
import numpy as np
import torch

def create_mmap_dataset(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    dataset_params = {
        "source_data": source_dir,
        "target_size": 64,
        "rm_black_slices": True,
        "remap_nn": False,
        "threshold_classes": None,
        "min_blob_size": None,
        "num_input_classes": 4,
        "grouping_rules_used": GROUPING_RULES,
        "new_labels_used": NEW_LABELS
    }
    
    config_path = os.path.join(dest_dir, "dataset_config.json")
    with open(config_path, "w") as f:
        json.dump(dataset_params, f, indent=4)

    print(f"Configuration saved to {config_path}")
    
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.npy')])    
    dataset = LazyDatasetDIDC(
            data_path=dataset_params["source_data"],
            grouping_rules=GROUPING_RULES,
            new_labels=NEW_LABELS,
            rm_black_slices=dataset_params["rm_black_slices"],
            remap_nn=dataset_params["remap_nn"],
            num_input_classes=dataset_params["num_input_classes"],
            file_list=files[:2]  # It is just for class initialization, the actual process is implemented below
        )
    
    for file in tqdm(files, desc="Processing files"):
        pat_id = file.replace('.npy', '')
        path = os.path.join(source_dir, file)

        pat = np.load(path, allow_pickle=True).item()
        sums = pat['mask_foreground'].sum(axis=(0,1))
        valid_indices = np.where(sums > 0)[0]

        if len(valid_indices) == 0:
            continue

        fg_list = []    
        mask_list = []
        for idx in valid_indices:
            fg_slice = torch.from_numpy(pat['mask_foreground'][:,:,idx]).unsqueeze(0).long()
            fg_slice[(fg_slice > 3) | (fg_slice < 0)] = 0 # remove noisy labels in the foreground mask

            mask_vol = pat['interpolated_segmentation']
            if mask_vol.ndim == 1:
                mask_vol = mask_vol.reshape(fg_slice.shape[1], fg_slice.shape[2], -1, order='F')
            mask_slice = torch.from_numpy(mask_vol[:, :, idx]).unsqueeze(0).long()

            fg_proc, mask_proc = dataset.process_slice(fg_slice, mask_slice)

            fg_list.append(fg_proc)
            mask_list.append(mask_proc)
        
        fg_stack = torch.stack(fg_list, dim=0)
        mask_stack = torch.stack(mask_list, dim=0)

        np.save(os.path.join(dest_dir, f"{pat_id}_fg.npy"), fg_stack.numpy())
        np.save(os.path.join(dest_dir, f"{pat_id}_mask.npy"), mask_stack.numpy())
                
         


if __name__ == "__main__":
    source_dir = 'DIDC_multiclass_coro_v2'
    dest_dir = 'DIDC_multiclass_coro_v2_prep'
    create_mmap_dataset(source_dir, dest_dir)
    