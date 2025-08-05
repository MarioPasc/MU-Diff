import numpy as np
import torch
from torch.utils.data import Dataset
import os

class BratsDataset(Dataset):
    def __init__(self, split="train", base_path="data/BRATS", target_modality="T1CE"):
        """
        Dataset for BraTS-like multi-contrast slices. 
        Loads .npy files and prepares (cond_inputs, target_output) pairs.
        target_modality: one of {"T1", "T2", "FLAIR", "T1CE"} indicating which contrast is the target.
        """
        self.base_path = base_path
        self.split = split
        all_modalities = ["T1", "T2", "FLAIR", "T1CE"]
        if target_modality not in all_modalities:
            raise ValueError(f"Invalid target_modality {target_modality}, must be one of {all_modalities}")
        # Determine modality loading order: conditions first, then target
        # Predefined orders from MU-Diff paper for reproducibility:
        orders = {
            "T1CE": ["FLAIR", "T2", "T1", "T1CE"],   # target T1CE
            "FLAIR": ["T1CE", "T1", "T2", "FLAIR"],  # target FLAIR
            "T2": ["T1CE", "T1", "FLAIR", "T2"],     # target T2
            "T1": ["FLAIR", "T1CE", "T2", "T1"]      # target T1
        }
        self.modality_order = orders.get(target_modality, [m for m in all_modalities if m != target_modality] + [target_modality])
        # Load numpy arrays for each modality in the specified order
        self.data = {}
        for mod in self.modality_order:
            file_path = os.path.join(base_path, split, f"{mod}.npy")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            # Use memory-mapping for efficiency (if large)
            self.data[mod] = np.load(file_path, mmap_mode='r')
        # All modalities should have the same number of slices:
        self.length = self.data[self.modality_order[0]].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Stack condition modality slices into a multi-channel tensor, and get target slice
        cond_imgs = []
        for mod in self.modality_order[:-1]:
            img = self.data[mod][idx]  # numpy slice (H x W)
            img_tensor = torch.from_numpy(img.astype(np.float32))
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)  # add channel dimension [1,H,W]
            cond_imgs.append(img_tensor)
        cond_stack = torch.cat(cond_imgs, dim=0)  # shape [3, H, W] for 3 condition images
        target_mod = self.modality_order[-1]
        target_img = self.data[target_mod][idx]  # numpy array of shape (H, W)
        target_tensor = torch.from_numpy(target_img.astype(np.float32)).unsqueeze(0)  # [1,H,W]
        return cond_stack, target_tensor
