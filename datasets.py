import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mode
from tqdm.auto import tqdm
import cv2
import sys


class MultiTissueDataset(Dataset):
    def __init__(self, data, num_classes_in=4):
        self.input_labels = data['input_labels']
        self.multiClassMasks = data['multiClassMasks']

        self.num_classes_in = num_classes_in

        # remap labels to be consecutive starting from 0 (one-hot encoding requirement), robust method
        # 0: right_ventricle + right_myocardium (merged), 1: left_ventricle, 2: left_myocardium (re-mapped), 3: background (re-mapped)
        mapper = np.arange(5, dtype=self.input_labels.dtype) 
        mapper[2] = 0  # right_ventricle + right_myocardium merged to label 0
        mapper[3] = 2
        mapper[4] = 3

        self.input_labels = mapper[self.input_labels]

    def __len__(self):
        return self.input_labels.shape[0]
    

    def __getitem__(self, idx):
        input_label    = torch.from_numpy(self.input_labels[idx]).long()
        multiClassMask = torch.from_numpy(self.multiClassMasks[idx]).long()

        input_label_one_hot = F.one_hot(input_label, num_classes=self.num_classes_in)
        input_label_one_hot = input_label_one_hot.permute(2,0,1).float()

        return {'input_label': input_label_one_hot, 'multiClassMask': multiClassMask}
    

class MultiTissueDatasetNoisyBkg(MultiTissueDataset):
    def __init__(self, data_concat, variance=1, mean=0.5, ignore_bkg=False):
        super().__init__(data_concat)
        self.background_label = 3
        self.variance = variance
        self.mean = mean
        self.ignore_bkg = ignore_bkg

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        input_label = sample['input_label']
        
        bkg_mask = input_label[self.background_label]  # Get the background mask
        noise = torch.randn_like(input_label) * np.sqrt(self.variance)  + self.mean # Generate noise of the same shape
        
        # Create a new input by replacing the background with noise
        new_input = input_label * (1 - bkg_mask) + noise * bkg_mask
        
        sample['input_label'] = new_input
        if self.ignore_bkg:
            sample['input_label'] = sample['input_label'][:-1]
        return sample


class BaseDatasetDIDC(Dataset):
    def __init__(self, data_path, grouping_rules, new_labels, target_size=(384, 384), 
                 num_input_classes=4, original_labels=None, rm_black_slices=True, remap_nn=False, threshold_classes=50, min_blob_size=10):
        
        self.data_path = data_path
        self.grouping_rules = grouping_rules
        self.new_labels = new_labels
        self.target_size = target_size
        self.rm_black_slices = rm_black_slices
        self.remap_nn = remap_nn
        self.num_input_classes = num_input_classes
        self.threshold_classes = threshold_classes
        self.min_blob_size = min_blob_size

        assert os.path.isdir(self.data_path), f"Path {self.data_path} non esiste."
        assert self.grouping_rules is not None, "Grouping rules mancanti."

        # Load original labels and generate LUT for remapping
        self.original_labels = self.load_original_labels() if original_labels is None else original_labels
        self.lut = self.generate_lut(self.grouping_rules, self.original_labels, self.new_labels)

    def process_slice(self, fg, mask):
        """
        Core transformation logic
        """
        # Resize (works for both (C, H, W) and (N, C, H, W))
        if fg.shape[-2] != self.target_size[0] or fg.shape[-1] != self.target_size[1]:
            fg = TF.resize(fg, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
            mask = TF.resize(mask, self.target_size, interpolation=TF.InterpolationMode.NEAREST)

        protected_indices = [self.original_labels.index(label) for label in ['Background', 'Other_tissue']] # channels untouched in the following steps
        other_idx = self.original_labels.index("Other_tissue")

        mask_np = mask.numpy()
        if self.min_blob_size is not None:
            mask = self.remove_cc(mask=mask_np, min_blob_size=self.min_blob_size, other_idx=self.original_labels.index("Other_tissue"), protected_indices=protected_indices)
            mask = torch.from_numpy(mask).long()

        if self.threshold_classes is not None:
            mask = self.filter_underrepresented_classes(mask=mask, min_pixels=self.threshold_classes, other_idx=other_idx, protected_indices=protected_indices)

        # NN Remapping of "Other_tissue" pixels to the most common label among their k nearest neighbors that are not "Other_tissue"
        if self.remap_nn:
            mask = self.remap_NN(mask.numpy(), other_idx)
            mask = torch.from_numpy(mask).long()

        # LUT remapping of segmentation masks to new labels (only for multi tissue maps)
        mask = self.lut[mask.long()]

        # Merge FG with segmentation masks (union) and one hot encoding
        mask = self.merge_fg_to_segm(fg, mask, self.new_labels)
        fg_one_hot = F.one_hot(fg.long(), num_classes=self.num_input_classes)
        
        # Final permutation in 3D (N, H, W, C) -> (N, C, H, W) or 2D (H, W, C) -> (C, H, W)
        if fg_one_hot.ndim == 4:
            fg_one_hot = fg_one_hot.permute(0, 3, 1, 2)
        else:
            fg_one_hot = fg_one_hot.permute(2, 0, 1)

        return fg_one_hot.float(), mask

    @staticmethod
    def remove_cc(mask, min_blob_size, other_idx, protected_indices):
        """
        Removes isolated blobs cv2.connectedComponentsWithStats. Takes and returns numpy array. Compatible with both 2D and 3D masks
        """
        protected_set = set(protected_indices)

        def process_2d_slice(slice_2d):
            unique_classes = np.unique(slice_2d)
            
            for cls in unique_classes:
                if cls in protected_set:
                    continue
                    
                bin_mask = (slice_2d == cls).astype(np.uint8)
                num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                
                for label_idx in range(1, num_labels):
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    
                    if area < min_blob_size:
                        slice_2d[labels_map == label_idx] = other_idx

        # Compatibility handled here
        if mask.ndim == 3:
            for i in range(mask.shape[0]):
                process_2d_slice(mask[i])
        elif mask.ndim == 2:
            process_2d_slice(mask)  
        return mask

    @staticmethod
    def check_black_foreground(mask):
        assert mask.ndim == 3, "Input mask must be a 3D array (D, H, W)"
        assert isinstance(mask, torch.Tensor), "Input mask must be a torch.Tensor"

        flat_mask = mask.flatten(start_dim=1)

        return flat_mask.sum(dim=1) == 0

    @staticmethod
    def generate_lut(grouping_rules, original_labels, new_labels):
        """
        Generate a Look-Up Table (LUT) for remapping original labels to new grouped labels based on provided grouping rules.
        The LUT is a numpy array where the (index +1) corresponds to the original label and the value at that index is the new grouped label index. 
        The LUT should have a length of max(original_labels) + 1 to accommodate the "background" label as 0.
        """
        lut = np.zeros(len(original_labels), dtype=int)
        for idx, label in enumerate(original_labels):
            if label in grouping_rules:
                group_name = grouping_rules[label]
                group_idx = new_labels.index(group_name)
                lut[idx] = group_idx
            else:
                print(f'Warning: Label "{label}" not found in grouping rules. Assigned to "Unknown".')
        return torch.from_numpy(lut).long()

    @staticmethod
    def merge_fg_to_segm(fg_tensor, segm_masks_tensor, new_labels):
        """
        Merge the foreground tensor with the segmentation masks tensor (which must be already remapped to the new labels)
        """
        heart_generic_idx = new_labels.index("Heart_generic")
        others_idx = new_labels.index("Blood_vessels")

        fg_mask_boolean = fg_tensor > 0

        # Set all possible heart pixels to "Others" to account for fg mask to be bigger or smaller than the original heart segmentation ("Heart_generic")
        segm_masks_tensor[segm_masks_tensor == heart_generic_idx] = others_idx
        segm_masks_tensor[fg_mask_boolean] = others_idx

        # Fg pixels remapping to match the labelling in the lut (which already includes the mapping for "LV_Myocardium", "LV_blood_pool", "RV_blood_pool_myocardium")
        # label 1: LV_blood_pool, label 2: LV_Myocardium, label 3: RV_blood_pool_myocardium
        fg_remapped = fg_tensor.clone() # to avoid problems in later steps
        fg_remapped[fg_tensor == 1] = new_labels.index("LV_blood_pool")
        fg_remapped[fg_tensor == 2] = new_labels.index("LV_Myocardium")
        fg_remapped[fg_tensor == 3] = new_labels.index("RV_blood_pool_myocardium")

        where_details = fg_tensor > 0 # Tmp mask to ignore 0 values in fg_remapped that would be remapped to the 0 index which has meaning
        segm_masks_tensor[where_details] = fg_remapped[where_details] # Final merge

        return segm_masks_tensor
    
    @staticmethod
    def filter_underrepresented_classes(mask, min_pixels, other_idx, protected_indices):
        """
        Scans the mask and subsitutes classese with less than min_pixels with other_idx, except for those in protected_indices
        Compatible with both 2D and 3D masks
        """
        protected_set = set(protected_indices)

        if mask.ndim == 3:
            for i in range(mask.shape[0]):
                slice_mask = mask[i]
                unique_vals, counts = torch.unique(slice_mask, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    v = val.item()
                    if v not in protected_set and count.item() < min_pixels:
                        slice_mask[slice_mask == v] = other_idx
                        
        elif mask.ndim == 2:
            unique_vals, counts = torch.unique(mask, return_counts=True)
            for val, count in zip(unique_vals, counts):
                v = val.item()
                if v not in protected_set and count.item() < min_pixels:
                    mask[mask == v] = other_idx

        return mask

    @staticmethod
    def remap_NN(segm_mask, other_tissue_idx, k=12):
        """
        Remap "Other_tissue" pixels in the segmentation mask to the most common label among their k nearest neighbors that are not "Other_tissue".
        """

        # If the input is 3D, apply the function slice by slice
        if segm_mask.ndim == 3:
            cleaned_slices = [BaseDatasetDIDC.remap_NN(s, other_tissue_idx, k) for s in segm_mask]
            return np.stack(cleaned_slices)
        
        # 2D case
        remap_segm = segm_mask.copy()

        valid_pixels = (segm_mask != other_tissue_idx) # piexls NOT Other_tissue
        valid_coords = np.argwhere(valid_pixels)
        valid_labels = segm_mask[valid_coords[:,0], valid_coords[:,1]] # this is for majority voting

        target_mask = segm_mask == other_tissue_idx
        target_indices = np.argwhere(target_mask) # Other tissue coordinates

        assert len(valid_coords) != 0, "No valid pixels found for NN search."

        if len(target_indices) == 0:
            print("No Other_tissue pixels found. No remapping needed.")
            return remap_segm
        else:
            tree = cKDTree(valid_coords)
            distances, nn_indices = tree.query(target_indices, k=k)
            neighbor_labels = valid_labels[nn_indices] # shape (num_target_pixels, k)

            if k > 1:
                majority_labels, _ = mode(neighbor_labels, axis=1)
                majority_labels = majority_labels.flatten()
            else:
                majority_labels = neighbor_labels.flatten()

            remap_segm[target_mask] = majority_labels # New label assignment

            return remap_segm

    def load_original_labels(self):
        original_labels = []

        if os.path.isfile(self.data_path + '/tissue_list.txt'):
            with open (self.data_path + '/tissue_list.txt', 'r') as f:
                for i, line in enumerate(f):
                    if i > 1:
                        line = line.strip().split()[-1]
                        original_labels.append(line)
            return ['Background'] + original_labels
        else:
            raise FileNotFoundError(f"Original labels file not found at {self.data_path + '/tissue_list.txt'}")

    def __len__(self):
        raise NotImplementedError("BaseDatasetDIDC is an abstract class. Implement __len__ in the subclass.")
    
    def __getitem__(self, idx):
        raise NotImplementedError("BaseDatasetDIDC is an abstract class. Implement __getitem__ in the subclass.")
    

class RAMDatasetDIDC(BaseDatasetDIDC):
    def __init__(self, *args, file_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        files = sorted(os.listdir(self.data_path)) if file_list is None else file_list
        print('Data loading in RAM...')

        fg_list = []
        mask_list = []

        for file in tqdm(files, desc="Loading files into RAM", file=sys.__stderr__):
            if file.endswith('.npy'):
                try:
                    pat = np.load(os.path.join(self.data_path, file), allow_pickle=True).item()
                    fg = torch.from_numpy(pat['mask_foreground']).permute(2,0,1).long()
                    mask = torch.from_numpy(np.round(pat['interpolated_segmentation'])).permute(2,0,1).float()

                    if self.rm_black_slices:
                        valid = ~self.check_black_foreground(fg)
                        fg = fg[valid]
                        mask = mask[valid]

                    fg, mask = self.process_slice(fg, mask) # preprocess in the loop

                    fg_list.append(fg)
                    mask_list.append(mask)
                except Exception as e:
                    print(f"Error with {file}: {e}")

        self.fg_tensor = torch.cat(fg_list, dim=0)
        self.mask_tensor = torch.cat(mask_list, dim=0)

    def __len__(self):
        return len(self.fg_tensor)

    def __getitem__(self, idx):
        return {'input_label': self.fg_tensor[idx], 'multiClassMask': self.mask_tensor[idx]}
    

class LazyDatasetDIDC(BaseDatasetDIDC):
    def __init__(self, *args, file_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        files = sorted(os.listdir(self.data_path)) if file_list is None else file_list
        print("Lazy Dataset: File...")
        
        self.samples = []
        for file in tqdm(files, desc="Indexing files and slices", file=sys.__stderr__):
            if file.endswith('.npy'):
                path = os.path.join(self.data_path, file)
                # For each file load only the non black slice indices (later) if the option is enabled
                try:
                    pat = np.load(path, allow_pickle=True).item()
                    num_slices = pat['mask_foreground'].shape[2]
                    indices = range(num_slices)

                    if self.rm_black_slices:
                        sums = pat['mask_foreground'].sum(axis=(0,1)) # check for the empty slices
                        indices = np.where(sums > 0)[0]
                    
                    for i in indices:
                        self.samples.append((path, i))
                except Exception as e:
                    print(f"Error loading file {file}: {e}")
        print(f"{len(self.samples)} samples indexed.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, slice_idx = self.samples[idx]
        
        # Load on demand the corresponding path
        pat = np.load(path, allow_pickle=True).item()
        
        # Extract slice & Convert to Tensor. Unsqueeze for later compatibility 
        fg = torch.from_numpy(pat['mask_foreground'][:, :, slice_idx]).unsqueeze(0).long()
        mask = torch.from_numpy(np.round(pat['interpolated_segmentation'][:, :, slice_idx])).unsqueeze(0).float()

        fg_proc, mask_proc = self.process_slice(fg, mask)

        if fg_proc.dim() == 4 and fg_proc.shape[0] == 1:
            fg_proc = fg_proc.squeeze(0)
        if mask_proc.dim() == 3 and mask_proc.shape[0] == 1:
            mask_proc = mask_proc.squeeze(0)

        return {'input_label': fg_proc, 'multiClassMask': mask_proc}
    

# OLDER VERSION --> here for compatibility with the analysis notebook 
class DatasetDIDC(Dataset):
    def __init__(self, data_path, grouping_rules, new_labels, target_size=(384, 384), num_input_classes=4, original_labels=None, file_list=None, rm_black_slices=True, remap_nn=False):
        self.data_path = data_path
        self.grouping_rules = grouping_rules
        self.new_labels = new_labels
        self.target_size = target_size
        self.rm_black_slices = rm_black_slices
        self.remap_nn = remap_nn

        assert os.path.isdir(self.data_path), f"Data path {self.data_path} does not exist or is not a directory."
        assert self.grouping_rules is not None, "Grouping rules must be provided for label remapping."

        self.original_labels = self.load_original_labels() if original_labels is None else original_labels
        self.lut = self.generate_lut(self.grouping_rules, self.original_labels)

        foreground_list = []
        segm_masks_list = []

        # Load all data into memory and resize on the fly if needed
        files = sorted(os.listdir(self.data_path)) if file_list is None else file_list
        for file in files:
            if file.endswith('.npy'):
                pat = np.load(self.data_path + '/' + file, allow_pickle=True).item()                

                fg = torch.from_numpy(pat['mask_foreground']).permute(2,0,1)
                mask = torch.from_numpy(pat['interpolated_segmentation']).float().permute(2,0,1)

                # Remove empty slices in the foreground if the option is enabled
                if self.rm_black_slices:
                    empty_slices_mask = self.check_black_foreground(fg)
                    fg = fg[~empty_slices_mask, ...]
                    mask = mask[~empty_slices_mask, ...]

                if fg.shape[1] != self.target_size[0] or fg.shape[2] != self.target_size[1]:
                    fg = TF.resize(fg, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
                    mask = TF.resize(mask, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
                    
                foreground_list.append(fg)
                segm_masks_list.append(mask)
        
        self.fg_tensor = torch.cat(foreground_list, dim=0).long()
        self.segm_masks_tensor = torch.cat(segm_masks_list, dim=0).long()

        # NN remapping of "Other_tissue" pixels to the most common label among their k nearest neighbors that are not "Other_tissue"
        if self.remap_nn:
            self.segm_masks_tensor = self.remap_NN(self.segm_masks_tensor.numpy(), self.original_labels.index("Other_tissue"), k=12)
            self.segm_masks_tensor = torch.from_numpy(self.segm_masks_tensor).long() 

        # remap segmentation masks using LUT (only for multi tissue maps) and merge with foreground mask (union)
        self.segm_masks_tensor = self.lut[self.segm_masks_tensor]
        self.segm_masks_tensor = self.merge_fg_to_segm(self.fg_tensor, self.segm_masks_tensor, self.new_labels)

        # One hot encode fg tensor
        num_classes = self.fg_tensor.max().item() + 1
        assert num_input_classes == num_classes, f"Number of input classes ({num_input_classes}) does not match the number of unique labels in the foreground ({num_classes})."
        
        self.fg_tensor = F.one_hot(self.fg_tensor, num_classes=num_input_classes).permute(0, 3, 1, 2)

    @staticmethod
    def check_black_foreground(mask):
        assert mask.ndim == 3, "Input mask must be a 3D array (D, H, W)"
        assert isinstance(mask, torch.Tensor), "Input mask must be a torch.Tensor"

        flat_mask = mask.flatten(start_dim=1)

        return flat_mask.sum(dim=1) == 0

    @staticmethod
    def generate_lut(grouping_rules, original_labels):
        """
        Generate a Look-Up Table (LUT) for remapping original labels to new grouped labels based on provided grouping rules.
        The LUT is a numpy array where the (index +1) corresponds to the original label and the value at that index is the new grouped label index. 
        The LUT should have a length of max(original_labels) + 1 to accommodate the "background" label as 0.
        """
        new_labels = sorted(list(set(grouping_rules.values())))
        lut = np.zeros(len(original_labels), dtype=int)
        for idx, label in enumerate(original_labels):
            if label in grouping_rules:
                group_name = grouping_rules[label]
                group_idx = new_labels.index(group_name)
                lut[idx] = group_idx
            else:
                print(f'Warning: Label "{label}" not found in grouping rules. Assigned to "Unknown".')
        return torch.from_numpy(lut).long()

    @staticmethod
    def merge_fg_to_segm(fg_tensor, segm_masks_tensor, new_labels):
        """
        Merge the foreground tensor with the segmentation masks tensor (which must be already remapped to the new labels)
        """
        heart_generic_idx = new_labels.index("Heart_generic")
        others_idx = new_labels.index("Others")

        fg_mask_boolean = fg_tensor > 0

        # Set all possible heart pixels to "Others" to account for fg mask to be bigger or smaller than the original heart segmentation ("Heart_generic")
        segm_masks_tensor[segm_masks_tensor == heart_generic_idx] = others_idx
        segm_masks_tensor[fg_mask_boolean] = others_idx

        # Fg pixels remapping to match the labelling in the lut (which already includes the mapping for "LV_Myocardium", "LV_blood_pool", "RV_blood_pool_myocardium")
        # label 1: LV_blood_pool, label 2: LV_Myocardium, label 3: RV_blood_pool_myocardium
        fg_remapped = fg_tensor.clone() # to avoid problems in later steps
        fg_remapped[fg_tensor == 1] = new_labels.index("LV_blood_pool")
        fg_remapped[fg_tensor == 2] = new_labels.index("LV_Myocardium")
        fg_remapped[fg_tensor == 3] = new_labels.index("RV_blood_pool_myocardium")

        where_details = fg_tensor > 0 # Tmp mask to ignore 0 values in fg_remapped that would be remapped to the 0 index which has meaning
        segm_masks_tensor[where_details] = fg_remapped[where_details] # Final merge

        return segm_masks_tensor
    
    @staticmethod
    def remap_NN(segm_mask, other_tissue_idx, k=12):
        """
        Remap "Other_tissue" pixels in the segmentation mask to the most common label among their k nearest neighbors that are not "Other_tissue".
        """

        # If the input is 3D, apply the function slice by slice
        if segm_mask.ndim == 3:
            print(f"Running NN filling on {segm_mask.shape[0]} slices...")
            cleaned_slices = [DatasetDIDC.remap_NN(s, other_tissue_idx, k) for s in segm_mask]
            return np.stack(cleaned_slices)
        
        # 2D case
        remap_segm = segm_mask.copy()

        valid_pixels = (segm_mask != other_tissue_idx) # piexls NOT Other_tissue
        valid_coords = np.argwhere(valid_pixels)
        valid_labels = segm_mask[valid_coords[:,0], valid_coords[:,1]] # this is for majority voting

        target_mask = segm_mask == other_tissue_idx
        target_indices = np.argwhere(target_mask) # Other tissue coordinates

        assert len(valid_coords) != 0, "No valid pixels found for NN search."

        if len(target_indices) == 0:
            print("No Other_tissue pixels found. No remapping needed.")
            return remap_segm
        else:
            tree = cKDTree(valid_coords)
            distances, nn_indices = tree.query(target_indices, k=k)
            neighbor_labels = valid_labels[nn_indices] # shape (num_target_pixels, k)

            if k > 1:
                majority_labels, _ = mode(neighbor_labels, axis=1)
                majority_labels = majority_labels.flatten()
            else:
                majority_labels = neighbor_labels.flatten()

            remap_segm[target_mask] = majority_labels # New label assignment

            return remap_segm

    def load_original_labels(self):
        original_labels = []

        if os.path.isfile(self.data_path + '/tissue_list.txt'):
            with open (self.data_path + '/tissue_list.txt', 'r') as f:
                for i, line in enumerate(f):
                    if i > 1:
                        line = line.strip().split()[-1]
                        original_labels.append(line)
            return ['Background'] + original_labels
        else:
            raise FileNotFoundError(f"Original labels file not found at {self.data_path + '/tissue_list.txt'}")

    def __len__(self):
        return self.fg_tensor.shape[0]

    def __getitem__(self, idx):
        return {'input_label': self.fg_tensor[idx].float(), 'multiClassMask': self.segm_masks_tensor[idx]}