import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

class BratsDataset(Dataset):
    """
    BraTS-like multi-contrast slice dataset.

    Loads 2D slices from .npy volumes per modality and returns:
        (cond_stack [C=3,H,W], target [1,H,W])

    Parameters
    ----------
    split : {"train","val","test"}
        Subdirectory under base_path.
    base_path : str
        Root path containing per-split .npy files.
    target_modality : {"T1","T2","FLAIR","T1CE"}
        Modality to predict.
    use_mmap : bool
        If True, np.memmap is used (keeps file descriptors open).
        If False, arrays are fully loaded into RAM and no FDs remain.
    dtype : np.dtype
        Numpy dtype used before conversion to float32 tensors.
    """
    ORDERS: Dict[str, List[str]] = {
        "T1CE": ["FLAIR", "T2", "T1", "T1CE"],
        "FLAIR": ["T1CE", "T1", "T2", "FLAIR"],
        "T2": ["T1CE", "T1", "FLAIR", "T2"],
        "T1": ["FLAIR", "T1CE", "T2", "T1"],
    }

    def __init__(
        self,
        split: str = "train",
        base_path: str = "data/BRATS",
        target_modality: str = "T1CE",
        use_mmap: bool = False,
        dtype: np.dtype = np.float32, # type: ignore
    ) -> None:
        all_modalities = ["T1", "T2", "FLAIR", "T1CE"]
        if target_modality not in all_modalities:
            raise ValueError(f"Invalid target_modality {target_modality}.")
        self.base_path = base_path
        self.split = split
        self.modality_order = self.ORDERS[target_modality]
        self.use_mmap = use_mmap
        self._data: Dict[str, np.ndarray] = {}

        for mod in self.modality_order:
            fp = os.path.join(base_path, split, f"{mod}.npy")
            if not os.path.isfile(fp):
                raise FileNotFoundError(fp)
            if use_mmap:
                arr = np.load(fp, mmap_mode="r")  # keeps an FD open
            else:
                arr = np.load(fp, allow_pickle=False)  # loads into RAM, no FD kept
                # ensure C-contiguous for fast torch.from_numpy
                if not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
            if arr.dtype != dtype:
                arr = arr.astype(dtype, copy=False)
            self._data[mod] = arr

        self.length = self._data[self.modality_order[0]].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        # Stack condition modality slices into a multi-channel tensor, and get target slice
        cond_imgs = []
        for mod in self.modality_order[:-1]:
            img = self._data[mod][idx]  # numpy slice (H x W)
            img_tensor = torch.from_numpy(img.astype(np.float32))
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)  # add channel dimension [1,H,W]
            # Normalize from z-score to [-1, 1] range
            # Clip z-score at ±3 sigma (covers ~99.7% of data), then scale to [-1, 1]
            img_tensor = torch.clamp(img_tensor, -3.0, 3.0) / 3.0
            cond_imgs.append(img_tensor)
        cond_stack = torch.cat(cond_imgs, dim=0)  # shape [3, H, W] for 3 condition images

        target_mod = self.modality_order[-1]
        target_img = self._data[target_mod][idx]  # numpy array of shape (H, W)
        target_tensor = torch.from_numpy(target_img.astype(np.float32)).unsqueeze(0)  # [1,H,W]
        # Normalize from z-score to [-1, 1] range
        target_tensor = torch.clamp(target_tensor, -3.0, 3.0) / 3.0
        return cond_stack, target_tensor


    def close(self) -> None:
        """Explicitly close memmap files if use_mmap=True."""
        if self.use_mmap:
            for arr in self._data.values():
                mm = getattr(arr, "_mmap", None)
                if mm is not None:
                    mm.close()

    def __del__(self) -> None:
        self.close()
