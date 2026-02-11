import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


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


