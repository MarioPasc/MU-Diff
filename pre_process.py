import os
import argparse
import numpy as np
import nibabel as nib
from typing import Dict, List

def normalize_volume(volume, mask=None):
    """
    Normalize a 3D volume using the mean and std of the brain region.
    If a brain mask is provided, use it; otherwise use all non-zero voxels.
    """
    data = volume.astype(np.float32)
    if mask is None:
        mask = data != 0  # use non-zero intensities as mask
    masked_data = data[mask]
    if masked_data.size == 0:
        mean, std = 0.0, 1.0
    else:
        mean = masked_data.mean()
        std = masked_data.std() if masked_data.std() != 0 else 1.0
    return (data - mean) / std

def extract_center_slices(volume, half_range):
    """
    Extract axial slices around the volume center index (±half_range).
    Returns a list of 2D slice arrays.
    """
    num_slices = volume.shape[2]
    center = num_slices // 2
    start = max(0, center - half_range)
    end   = min(num_slices - 1, center + half_range)
    slices = [volume[:, :, idx] for idx in range(start, end+1)]
    return slices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process dataset into slices.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the root directory of the raw dataset (with patient subfolders).")
    parser.add_argument("--output_dir", type=str, default="data/BraTS_MEN",
                        help="Output directory to save processed data (will contain train/val/test splits).")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Fraction of patients for training set.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of patients for validation set.")
    parser.add_argument("--slice_half_range", type=int, default=80,
                        help="Number of slices to take on each side of the volume center.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting patients.")
    args = parser.parse_args()

    # Prepare output folders
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    # List all patient subdirectories
    patients = sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))])
    if len(patients) == 0:
        raise RuntimeError("No patient subdirectories found in the input directory.")
    # Shuffle and split patients into train/val/test
    rng = np.random.RandomState(args.seed)
    rng.shuffle(patients)
    total = len(patients)
    n_train = int(total * args.train_ratio)
    n_val = int(total * args.val_ratio)
    n_val = min(n_val, total - n_train)  # adjust if rounding issues
    n_test = total - n_train - n_val
    train_patients = set(patients[:n_train])
    val_patients   = set(patients[n_train:n_train+n_val])
    test_patients  = set(patients[n_train+n_val:])
    # Define modality name mapping (raw filename keyword -> output name)
    modality_map = {"t1n": "T1", "t1c": "T1CE", "t2w": "T2", "t2f": "FLAIR"}
    # Initialize storage for slices of each modality per split
    slices_by_split: Dict[str, Dict[str, List[np.ndarray]]] = {
        "train": {mod: [] for mod in modality_map.values()},
        "val":   {mod: [] for mod in modality_map.values()},
        "test":  {mod: [] for mod in modality_map.values()}
    }

    for patient in patients:
        patient_dir = os.path.join(args.input_dir, patient)
        # Determine split for this patient
        if patient in train_patients:
            split = "train"
        elif patient in val_patients:
            split = "val"
        else:
            split = "test"
        # Load each modality for the patient
        patient_slices: Dict[str, List[np.ndarray]] = {mod: [] for mod in modality_map.values()}
        for fname in os.listdir(patient_dir):
            # Skip segmentation or mask files if present
            if fname.lower().endswith(("seg.nii", "seg.nii.gz")):
                continue
            # Identify modality by filename pattern
            for key, mod_name in modality_map.items():
                if f"{key}." in fname.lower():  # matches e.g. "t1c." in filename
                    img = nib.load(os.path.join(patient_dir, fname))
                    volume = img.get_fdata()
                    vol_norm = normalize_volume(volume)  # normalize intensities
                    slices = extract_center_slices(vol_norm, args.slice_half_range)
                    patient_slices[mod_name].extend(slices)
        # Append this patient's slices to the global list for its split
        for mod_name, slice_list in patient_slices.items():
            if slice_list:  # only extend if we got slices for that modality
                slices_by_split[split][mod_name].extend(slice_list)

    # Convert lists to numpy arrays and save as .npy files
    for split, mod_dict in slices_by_split.items():
        for mod_name, slice_list in mod_dict.items():
            if len(slice_list) == 0:
                continue  # no data for this modality (should not happen if data is complete)
            arr = np.stack(slice_list).astype(np.float32)
            out_path = os.path.join(args.output_dir, split, f"{mod_name}.npy")
            np.save(out_path, arr)
            print(f"Saved {split}/{mod_name}.npy with shape {arr.shape}")
