import os
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips #type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, MAE, and LPIPS between prediction and ground truth images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to directory of ground truth images (png format).")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to directory of predicted images.")
    args = parser.parse_args()

    gt_files = sorted([f for f in os.listdir(args.gt_dir) if os.path.isfile(os.path.join(args.gt_dir, f))])
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if os.path.isfile(os.path.join(args.pred_dir, f))])
    # Match files by name
    common_files = [f for f in gt_files if f in pred_files]
    if len(common_files) == 0:
        raise RuntimeError("No matching image files found in the provided directories.")

    # Initialize LPIPS model (AlexNet backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex').to(device)

    total_psnr = total_ssim = total_mae = total_lpips = 0.0
    count = 0
    for fname in common_files:
        gt_path = os.path.join(args.gt_dir, fname)
        pred_path = os.path.join(args.pred_dir, fname)
        # Load images in grayscale
        gt_img = Image.open(gt_path).convert("L")
        pred_img = Image.open(pred_path).convert("L")
        gt_arr = np.array(gt_img, dtype=np.float32)
        pred_arr = np.array(pred_img, dtype=np.float32)
        # Scale pixel values to [0,1]
        gt_norm = gt_arr / 255.0
        pred_norm = pred_arr / 255.0
        # PSNR and SSIM
        psnr_val = peak_signal_noise_ratio(gt_norm, pred_norm, data_range=1.0)
        ssim_val = structural_similarity(gt_norm, pred_norm, data_range=1.0)
        # MAE
        mae_val = float(np.mean(np.abs(gt_norm - pred_norm)))
        # LPIPS (convert to 3-channel tensors in [-1,1])
        gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)   # shape [1,3,H,W]
        pred_tensor = torch.from_numpy(pred_norm).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        lpips_val = float(loss_fn(gt_tensor*2-1, pred_tensor*2-1).item())
        # Accumulate
        total_psnr  += psnr_val
        total_ssim  += ssim_val
        total_mae   += mae_val
        total_lpips += lpips_val
        count += 1

    # Compute averages
    avg_psnr  = total_psnr / count
    avg_ssim  = total_ssim / count
    avg_mae   = total_mae / count
    avg_lpips = total_lpips / count
    # Display results
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average LPIPS: {avg_lpips:.6f}")
