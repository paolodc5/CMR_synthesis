import os
import numpy as np
import torch.nn.functional as F


def load_all_data(folder_path):
    """
    Loads all .npz files in a folder and groups their contents by field.
    
    Returns:
    dict: Dictionary where keys are the internal field names of the npz files 
          and values are lists containing the data from all files.
          Example: {'images': [array_file1, array_file2], 'masks': [label_file1, label_file2]}
    """
    data_dict = {}
    
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return {}
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

    if not files:
        print("No .npz files found.")
        return {}

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        try:
            with np.load(file_path) as npz_file:
                for key in npz_file.files:
                    
                    if key not in data_dict:
                        data_dict[key] = []
                    
                    data_dict[key].append(npz_file[key])
                    
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            continue

    return data_dict

def squeeze_and_concat(data_dict):
    """
    Takes a dictionary of lists of arrays, squeezes each array, and concatenates them along the first axis.
    
    Args:
    data_dict (dict): Dictionary where keys are field names and values are lists of arrays.
    
    Returns:
    dict: Dictionary where each key's value is a single concatenated array.
    """
    concatenated_dict = {}
    
    for key, list_of_arrays in data_dict.items():
        squeezed_arrays = [np.squeeze(arr) for arr in list_of_arrays]
        concatenated_dict[key] = np.concatenate(squeezed_arrays, axis=0)
    
    return concatenated_dict

def filter_mask_keep_labels(mask, keep_labels=(0, 1, 3), background_label=4):
    """Keep only specific labels in a segmentation mask.

    Pixels whose value is in ``keep_labels`` are kept unchanged; all other pixels
    are set to ``background_label``.

    Supports 2D masks (H, W) and batched masks (N, H, W).

    Args:
        mask: numpy array (or torch tensor) containing integer labels.
        keep_labels: iterable of labels to keep.
        background_label: label value to assign to all other pixels.

    Returns:
        Mask with the same shape as input.
    """

    # Accept torch tensors without making utils.py depend on torch.
    if hasattr(mask, "detach") and hasattr(mask, "cpu") and hasattr(mask, "numpy"):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    keep_labels_arr = np.array(list(keep_labels))
    out = np.where(np.isin(mask_np, keep_labels_arr), mask_np, background_label)

    # Preserve integer dtype when possible.
    if np.issubdtype(mask_np.dtype, np.integer):
        out = out.astype(mask_np.dtype, copy=False)
    return out


def multiclass_dice_loss(pred, target, smooth=1):
    """
    Computes Dice Loss for multi-class segmentation.
    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: Ground truth labels (batch_size, H, W) with integer class labels.
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.
    """
    pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
    num_classes = pred.shape[1]  # Number of classes (C)
    dice = 0  # Initialize Dice loss accumulator
    
    for c in range(num_classes):  # Loop through each class
        pred_c = pred[:, c]  # Predictions for class c
        target_c = (target == c).float() # Ground truth for class c (targets are not one-hot encoded)

        intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
        
        dice += (2. * intersection + smooth) / (union + smooth)  # Per-class Dice score

    return 1 - dice.mean() / num_classes  # Average Dice Loss across classes

