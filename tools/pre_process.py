import os
import sys
import argparse
import logging
import traceback
import numpy as np
import nibabel as nib
from numpy.lib.format import open_memmap
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger with console and optional file handler."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Clear existing handlers (important when rerunning in notebooks)
    logging.getLogger().handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    logging.getLogger().setLevel(numeric_level)
    logging.getLogger().addHandler(console_handler)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        except Exception:
            logger.exception("Failed to set up file logging at %s", log_file)


def normalize_volume(volume, mask=None):
    """
    Normalize a 3D volume using the mean and std of the brain region.
    If a brain mask is provided, use it; otherwise use all non-zero voxels.
    """
    try:
        data = volume if getattr(volume, "dtype", None) == np.float32 else volume.astype(np.float32, copy=False)
        if mask is None:
            mask = data != 0  # use non-zero intensities as mask
        masked_data = data[mask]
        if masked_data.size == 0:
            mean, std = 0.0, 1.0
        else:
            mean = masked_data.mean()
            std_val = masked_data.std()
            std = std_val if std_val != 0 else 1.0
        result = (data - mean) / std
        logger.debug("Normalized volume with mean=%.4f, std=%.4f, shape=%s", float(mean), float(std), tuple(data.shape))
        return result
    except Exception:
        logger.exception("Failed to normalize volume (shape=%s)", getattr(volume, "shape", None))
        raise


def extract_center_slices(volume, half_range):
    """
    Extract axial slices around the volume center index (±half_range).
    Returns a list of 2D slice arrays.
    """
    try:
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D")
        num_slices = volume.shape[2]
        center = num_slices // 2
        start = max(0, center - half_range)
        end = min(num_slices - 1, center + half_range)
        slices = [volume[:, :, idx] for idx in range(start, end + 1)]
        logger.debug(
            "Extracted %d slices (start=%d, end=%d, center=%d) from volume shape=%s",
            len(slices), start, end, center, tuple(volume.shape)
        )
        return slices
    except Exception:
        logger.exception("Failed to extract center slices from volume")
        raise


def _slice_bounds(depth: int, half_range: int):
    center = depth // 2
    start = max(0, center - half_range)
    end = min(depth - 1, center + half_range)
    return start, end


def main():
    parser = argparse.ArgumentParser(description="Pre-process dataset into slices.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory of the raw dataset (with patient subfolders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/BraTS_MEN",
        help="Output directory to save processed data (will contain train/val/test splits).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Fraction of patients for training set.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of patients for validation set.")
    parser.add_argument(
        "--slice_half_range",
        type=int,
        default=80,
        help="Number of slices to take on each side of the volume center.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting patients.")
    parser.add_argument(
        "--num_patients",
        type=int,
        default=None,
        help="Number of patients to include after shuffling with the given seed. Use all if omitted.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a log file.",
    )

    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)

    logger.info(
        "Starting preprocessing: input_dir=%s, output_dir=%s, train_ratio=%.2f, val_ratio=%.2f, slice_half_range=%d, seed=%d, num_patients=%s",
        args.input_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.slice_half_range,
        args.seed,
        str(args.num_patients) if args.num_patients is not None else "all",
    )

    if not os.path.isdir(args.input_dir):
        logger.error("Input directory does not exist or is not a directory: %s", args.input_dir)
        sys.exit(1)

    # Prepare output folders
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
    except Exception:
        logger.exception("Failed to create output directories under %s", args.output_dir)
        sys.exit(1)

    # List all patient subdirectories
    try:
        patients = sorted(
            [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        )
    except Exception:
        logger.exception("Failed to list patients in input_dir: %s", args.input_dir)
        sys.exit(1)

    if len(patients) == 0:
        logger.error("No patient subdirectories found in the input directory: %s", args.input_dir)
        sys.exit(1)

    if args.num_patients is not None and args.num_patients <= 0:
        logger.error("num_patients must be a positive integer; got %s", args.num_patients)
        sys.exit(1)

    # Shuffle and optionally subsample patients, then split into train/val/test
    rng = np.random.RandomState(args.seed)
    rng.shuffle(patients)

    original_total = len(patients)
    if args.num_patients is not None:
        if args.num_patients < original_total:
            patients = patients[: args.num_patients]
            logger.info(
                "Subsampled patients: %d -> %d using seed=%d", original_total, len(patients), args.seed
            )
        else:
            logger.info(
                "Requested num_patients (%d) >= available (%d); using all patients",
                args.num_patients,
                original_total,
            )

    total = len(patients)
    if total == 0:
        logger.error("No patients to process after subsampling.")
        sys.exit(1)

    n_train = int(total * args.train_ratio)
    n_val = int(total * args.val_ratio)
    n_val = min(n_val, total - n_train)  # adjust if rounding issues
    n_test = total - n_train - n_val

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    logger.info(
        "Dataset split: total=%d | train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        total,
        n_train,
        100.0 * n_train / total,
        n_val,
        100.0 * n_val / total,
        n_test,
        100.0 * n_test / total,
    )

    # Define modality name mapping (raw filename keyword -> output name)
    modality_map = {"t1n": "T1", "t1c": "T1CE", "t2w": "T2", "t2f": "FLAIR"}

    # First pass: count slices per split/modality and infer (H,W)
    counts: Dict[str, Dict[str, int]] = {s: {m: 0 for m in modality_map.values()} for s in ("train", "val", "test")}
    hw_by_mod: Dict[str, Optional[tuple]] = {m: None for m in modality_map.values()}

    logger.info("First pass: counting slices and inferring shapes...")
    for patient in patients:
        try:
            patient_dir = os.path.join(args.input_dir, patient)
            if patient in train_patients:
                split = "train"
            elif patient in val_patients:
                split = "val"
            else:
                split = "test"

            try:
                file_list = os.listdir(patient_dir)
            except Exception:
                logger.exception("[COUNT] Failed to list files in %s", patient_dir)
                continue

            for fname in file_list:
                low = fname.lower()
                if low.endswith(("seg.nii", "seg.nii.gz")):
                    continue
                for key, mod_name in modality_map.items():
                    if f"{key}." in low:
                        fpath = os.path.join(patient_dir, fname)
                        try:
                            img = nib.load(fpath)
                            shape = img.shape
                            if shape is None or len(shape) < 3:
                                logger.warning("[COUNT] Unexpected image dims for %s: %s", fpath, shape)
                                break
                            h, w, d = shape[:3]
                            if hw_by_mod[mod_name] is None:
                                hw_by_mod[mod_name] = (h, w)
                                logger.debug("[COUNT] Set shape for %s to (%d,%d)", mod_name, h, w)
                            elif hw_by_mod[mod_name] != (h, w):
                                logger.warning(
                                    "[COUNT] Skipping %s due to shape mismatch for modality %s: got (%d,%d) expected %s",
                                    fpath,
                                    mod_name,
                                    h,
                                    w,
                                    hw_by_mod[mod_name],
                                )
                                break
                            s, e = _slice_bounds(d, args.slice_half_range)
                            counts[split][mod_name] += (e - s + 1)
                        except Exception:
                            logger.exception("[COUNT] Failed to inspect %s", fpath)
                        finally:
                            break  # stop checking other keys for this file
        except Exception:
            logger.exception("[COUNT] Unexpected error for patient %s", patient)
            continue

    # Log counts
    for split in ("train", "val", "test"):
        for mod_name in modality_map.values():
            logger.info("[COUNT] split=%s, modality=%s, slices=%d, hw=%s", split, mod_name, counts[split][mod_name], hw_by_mod[mod_name])

    # Create memory-mapped .npy files for each split/modality
    logger.info("Allocating memmaps for output .npy files...")
    memmaps: Dict[tuple, np.memmap] = {}
    write_idx: Dict[tuple, int] = {}
    for split in ("train", "val", "test"):
        out_split_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)
        for mod_name in modality_map.values():
            total_slices = counts[split][mod_name]
            if total_slices <= 0:
                logger.warning("No slices for split=%s modality=%s; skipping allocation", split, mod_name)
                continue
            hw = hw_by_mod[mod_name]
            if hw is None:
                logger.warning("No shape determined for modality %s; skipping", mod_name)
                continue
            h, w = hw
            out_path = os.path.join(out_split_dir, f"{mod_name}.npy")
            try:
                mm = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(total_slices, h, w))
                memmaps[(split, mod_name)] = mm
                write_idx[(split, mod_name)] = 0
                logger.info("Allocated %s with shape (%d, %d, %d)", out_path, total_slices, h, w)
            except Exception:
                logger.exception("Failed to allocate memmap at %s", out_path)

    # Second pass: process and write slices incrementally
    logger.info("Second pass: processing volumes and writing to memmaps...")
    for patient in patients:
        try:
            patient_dir = os.path.join(args.input_dir, patient)
            if patient in train_patients:
                split = "train"
            elif patient in val_patients:
                split = "val"
            else:
                split = "test"

            logger.info("Processing patient=%s (split=%s)", patient, split)

            try:
                file_list = os.listdir(patient_dir)
            except Exception:
                logger.exception("[WRITE] Failed to list files in %s", patient_dir)
                continue

            for fname in file_list:
                low = fname.lower()
                if low.endswith(("seg.nii", "seg.nii.gz")):
                    logger.debug("Skipping segmentation file: %s", fname)
                    continue
                for key, mod_name in modality_map.items():
                    if f"{key}." in low:
                        fpath = os.path.join(patient_dir, fname)
                        mm_key = (split, mod_name)
                        if mm_key not in memmaps:
                            logger.debug("No allocation for %s; possibly zero slices. Skipping %s", mm_key, fpath)
                            break
                        try:
                            img = nib.load(fpath)
                            shape = img.shape
                            if shape is None or len(shape) < 3:
                                logger.warning("[WRITE] Unexpected image dims for %s: %s", fpath, shape)
                                break
                            h, w, d = shape[:3]
                            exp_hw = hw_by_mod[mod_name]
                            if exp_hw != (h, w):
                                logger.warning(
                                    "[WRITE] Skipping %s due to shape mismatch for modality %s: got (%d,%d) expected %s",
                                    fpath,
                                    mod_name,
                                    h,
                                    w,
                                    exp_hw,
                                )
                                break
                            volume = img.get_fdata(dtype=np.float32)
                            # Normalize and write slices directly without building large lists
                            # Compute bounds
                            s, e = _slice_bounds(d, args.slice_half_range)
                            # Normalize once
                            vol_norm = normalize_volume(volume)
                            del volume
                            mm = memmaps[mm_key]
                            idx = write_idx[mm_key]
                            for z in range(s, e + 1):
                                try:
                                    mm[idx] = vol_norm[:, :, z]
                                    idx += 1
                                except Exception:
                                    logger.exception("[WRITE] Failed writing slice z=%d from %s", z, fpath)
                                    # keep going with next slice
                                    continue
                            write_idx[mm_key] = idx
                            del vol_norm, img
                        except Exception:
                            logger.exception("[WRITE] Failed processing file %s (patient=%s, modality=%s)", fpath, patient, mod_name)
                        finally:
                            break  # matched a key; move to next file
        except Exception:
            logger.exception("[WRITE] Unexpected error while processing patient %s", patient)
            continue

    # Flush memmaps and report final counts
    for key, mm in memmaps.items():
        try:
            mm.flush()
        except Exception:
            logger.exception("Failed to flush memmap for %s", key)
    for (split, mod_name), idx in write_idx.items():
        logger.info("Final written slices for split=%s modality=%s: %d / planned %d", split, mod_name, idx, counts[split][mod_name])

    logger.info("Preprocessing completed.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # argparse or explicit sys.exit; let it pass without extra trace
        raise
    except Exception:
        logger.critical("Fatal error in preprocessing script")
        logger.critical(traceback.format_exc())
        sys.exit(1)
