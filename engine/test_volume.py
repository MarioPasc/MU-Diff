import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

# Generators
from backbones.ncsnpp_generator_adagn_feat import NCSNpp, NCSNpp_adaptive

# ==============================
# Utilities copied/aligned with train.py
# ==============================

def var_func_vp(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Variance function for VP schedule."""
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    return var

def var_func_geometric(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Geometric variance schedule."""
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather and reshape schedule entries for broadcasting to x shape."""
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_time_schedule(args, device: torch.device) -> torch.Tensor:
    """t in (0,1], length = n_timestep+1, identical to train.py."""
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    return t.to(device)

def get_sigma_schedule(args, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute betas, sigmas, a_s exactly like train.py."""
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device).type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas

class Posterior_Coefficients:
    """Posterior coefficients identical to train.py."""
    def __init__(self, args, device: torch.device) -> None:
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior_combine(coefficients: Posterior_Coefficients,
                             x_0_1: torch.Tensor,
                             x_0_2: torch.Tensor,
                             x_t: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
    """Two-predictor posterior sampling, identical math to train.py."""
    def q_posterior(x_0_1, x_0_2, x_t, t):
        mean1 = extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_1 + \
                extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        mean2 = extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_2 + \
                extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        mean = (mean1 + mean2) / 2
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0_1, x_0_2, x_t, t):
        mean, _, log_var = q_posterior(x_0_1, x_0_2, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    return p_sample(x_0_1, x_0_2, x_t, t)

def sample_from_model(coefficients: Posterior_Coefficients,
                      generator1: torch.nn.Module, cond1: torch.Tensor,
                      generator2: torch.nn.Module, cond2: torch.Tensor, cond3: torch.Tensor,
                      n_time: int, x_init: torch.Tensor, T: torch.Tensor, opt) -> torch.Tensor:
    """
    Run the reverse process. Mirrors train.py ordering and dtype.
    """
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64, device=x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            # Use autocast dtype consistent with training flags
            dtype = torch.bfloat16 if getattr(opt, "use_bf16", False) else torch.float16
            with torch.autocast('cuda', dtype=dtype):
                x_0_1 = generator1(x, cond1, cond2, cond3, t_time, latent_z)
                # IMPORTANT: keep spatial dims when indexing channel
                x_0_2 = generator2(x, cond1, cond2, cond3, t_time, latent_z, x_0_1[:, [0], :, :])
                x = sample_posterior_combine(coefficients, x_0_1[:, [0], :, :], x_0_2[:, [0], :, :], x, t).detach()
    return x

# ==============================
# I/O and preprocessing
# ==============================

def robust_minmax_to_minus1_1(vol: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              pmin: float = 1.0,
                              pmax: float = 99.0) -> np.ndarray:
    """
    Map intensities to [-1, 1] using robust [pmin, pmax] percentiles over nonzero voxels.
    Falls back to full range if mask is degenerate.
    """
    data = vol.astype(np.float32, copy=False)
    m = (data != 0) if mask is None else (mask.astype(bool) & (data == data))  # ignore NaNs
    if not np.any(m):
        # degenerate: return zeros
        return np.zeros_like(data, dtype=np.float32)
    vals = data[m]
    lo = np.percentile(vals, pmin)
    hi = np.percentile(vals, pmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros_like(data, dtype=np.float32)
    x01 = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    x11 = x01 * 2.0 - 1.0
    return x11

def extract_center_slices(volume: np.ndarray, half_range: int) -> Tuple[List[np.ndarray], int, int]:
    """
    Extract axial slices around the center index (±half_range). Returns slice list and bounds.
    """
    z = volume.shape[2]
    c = z // 2
    start = max(0, c - half_range)
    end = min(z - 1, c + half_range)
    slices = [volume[:, :, idx] for idx in range(start, end + 1)]
    return slices, start, end

def reconstruct_volume_from_slices(predicted_slices: List[np.ndarray],
                                   original_shape: Tuple[int, int, int],
                                   start_slice: int, end_slice: int) -> np.ndarray:
    """
    Rebuild 3D volume from predicted slices. Non-predicted slices are zeros.
    """
    vol = np.zeros(original_shape, dtype=np.float32)
    for i, sl in enumerate(predicted_slices):
        k = start_slice + i
        if start_slice <= k <= end_slice and k < original_shape[2]:
            vol[:, :, k] = sl.astype(np.float32, copy=False)
    return vol

def load_and_preprocess_volume(file_path: str, slice_half_range: int) -> Tuple[List[np.ndarray], Tuple[int,int,int], np.ndarray, nib.Nifti1Header, int, int]:
    """
    Load NIfTI. Normalize to [-1,1] with robust min-max. Extract center slices.
    """
    img = nib.load(file_path)
    vol = img.get_fdata()
    vol_norm = robust_minmax_to_minus1_1(vol)
    slices, s0, s1 = extract_center_slices(vol_norm, slice_half_range)
    return slices, vol.shape, img.affine, img.header, s0, s1

def load_checkpoint(template: str, net: torch.nn.Module, name: str, device: torch.device) -> None:
    """
    Load a .pth saved via torch.save(model.state_dict()) from DDP, removing 'module.' prefix when present.
    """
    path = template.format(name)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = { (k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items() }
    net.load_state_dict(ckpt, strict=False)
    net.eval()

# ==============================
# Inference driver
# ==============================

def predict_volume(args) -> None:
    """
    Predict one target modality volume from three input modalities, aligned with train.py behavior.
    """
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device(f'cuda:{args.gpu_chose}')

    # Init models
    gen1 = NCSNpp(args).to(device)
    gen2 = NCSNpp_adaptive(args).to(device)

    # Load weights saved by train.py
    exp_dir = os.path.join(args.output_path, args.exp)
    ckpt_tmpl = os.path.join(exp_dir, "{}.pth")
    load_checkpoint(ckpt_tmpl, gen1, "gen_diffusive_1", device)
    load_checkpoint(ckpt_tmpl, gen2, "gen_diffusive_2", device)

    # Diffusion coefficients
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    # Modality order identical to BratsDatasetWrapper docstring in train.py
    modality_orders: Dict[str, List[str]] = {
        "T1CE": ["FLAIR", "T2", "T1"],
        "FLAIR": ["T1CE", "T1", "T2"],
        "T2": ["T1CE", "T1", "FLAIR"],
        "T1": ["FLAIR", "T1CE", "T2"],
    }
    if args.target_modality not in modality_orders:
        raise ValueError(f"Unsupported target modality: {args.target_modality}")
    inputs_needed = modality_orders[args.target_modality]

    provided = {
        "T1CE": args.input_t1ce,
        "T1":   args.input_t1,
        "T2":   args.input_t2,
        "FLAIR":args.input_flair,
    }
    for m in inputs_needed:
        if not provided.get(m):
            raise ValueError(f"Missing required input for {m}. Provide --input_{m.lower()}")

    # Load all inputs
    ref_shape = None
    ref_affine = None
    ref_header = None
    vols: Dict[str, Dict[str, object]] = {}
    for m in inputs_needed:
        slices, shp, aff, hdr, s0, s1 = load_and_preprocess_volume(provided[m], args.slice_half_range)
        vols[m] = dict(slices=slices, shape=shp, s0=s0, s1=s1)
        if ref_shape is None:
            ref_shape, ref_affine, ref_header = shp, aff, hdr
        elif shp != ref_shape:
            raise ValueError(f"All input volumes must share shape. Got {shp} vs {ref_shape} for {m}")

    n = len(vols[inputs_needed[0]]["slices"])  # type: ignore[call-arg]
    predicted_slices: List[np.ndarray] = []

    # Inference loop over slices
    for i in range(n):
        cond_tensors: List[torch.Tensor] = []
        for m in inputs_needed:
            sl = vols[m]["slices"][i]  # type: ignore[index]
            t = torch.from_numpy(sl.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            if t.shape[-2:] != (args.image_size, args.image_size):
                t = F.interpolate(t, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
            cond_tensors.append(t.to(device, non_blocking=True))

        cond1, cond2, cond3 = cond_tensors
        x_t = torch.randn_like(cond1)

        with torch.no_grad():
            fake = sample_from_model(pos_coeff, gen1, cond1, gen2, cond2, cond3,
                                     args.num_timesteps, x_t, T, args)
        # Map to [0,1] same as to_range_0_1 in train.py for reporting
        pred = ((fake + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy().squeeze()
        predicted_slices.append(pred)

        if (i + 1) % 10 == 0:
            print(f"[infer] processed {i+1}/{n} slices")

    # Rebuild volume in original shape
    s0 = int(vols[inputs_needed[0]]["s0"])  # type: ignore[index]
    s1 = int(vols[inputs_needed[0]]["s1"])  # type: ignore[index]
    vol_pred = reconstruct_volume_from_slices(predicted_slices, ref_shape, s0, s1)  # type: ignore[arg-type]

    # Save NIfTI
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"predicted_{args.target_modality.lower()}.nii.gz")
    nib.save(nib.Nifti1Image(vol_pred, ref_affine, ref_header), out_path)  # type: ignore[arg-type]
    print(f"[done] saved: {out_path} | shape={tuple(vol_pred.shape)} | slices={s0}..{s1}")

def build_argparser() -> argparse.Namespace:
    """
    Argument parser. Defaults chosen to match train.py where possible.
    """
    p = argparse.ArgumentParser("MU-Diff volume prediction")
    # Inputs
    p.add_argument('--input_t1ce', type=str, help='Path to T1CE NIfTI')
    p.add_argument('--input_t1',   type=str, help='Path to T1 NIfTI')
    p.add_argument('--input_t2',   type=str, help='Path to T2 NIfTI')
    p.add_argument('--input_flair',type=str, help='Path to FLAIR NIfTI')
    p.add_argument('--target_modality', type=str, required=True, choices=['T1CE','FLAIR','T2','T1'])
    p.add_argument('--output_dir', type=str, required=True)

    # Model/exp
    p.add_argument('--exp', type=str, required=True, help='Experiment directory name under --output_path')
    p.add_argument('--output_path', type=str, default='./results')

    # Preprocess
    p.add_argument('--slice_half_range', type=int, default=80)
    p.add_argument('--image_size', type=int, default=256)

    # Core arch params (must match training)
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--num_channels', type=int, default=1)
    p.add_argument('--num_channels_dae', type=int, default=128)
    p.add_argument('--n_mlp', type=int, default=3)
    p.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,4])
    p.add_argument('--num_res_blocks', type=int, default=2)
    p.add_argument('--attn_resolutions', nargs='+', type=int, default=[16])
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--resamp_with_conv', action='store_false', default=True)
    p.add_argument('--conditional', action='store_false', default=True)
    p.add_argument('--fir', action='store_false', default=True)
    p.add_argument('--fir_kernel', nargs='+', type=int, default=[1,3,3,1])
    p.add_argument('--skip_rescale', action='store_false', default=True)
    p.add_argument('--resblock_type', type=str, default='biggan')
    p.add_argument('--progressive', type=str, default='none')
    p.add_argument('--progressive_input', type=str, default='residual')
    p.add_argument('--progressive_combine', type=str, default='sum')

    # Diffusion params
    p.add_argument('--use_geometric', action='store_true', default=False)
    p.add_argument('--beta_min', type=float, default=0.1)
    p.add_argument('--beta_max', type=float, default=20.0)
    p.add_argument('--num_timesteps', type=int, default=4)

    # Embedding params
    p.add_argument('--embedding_type', type=str, default='positional')
    p.add_argument('--fourier_scale', type=float, default=16.0)
    p.add_argument('--not_use_tanh', action='store_true', default=False)
    p.add_argument('--z_emb_dim', type=int, default=256)
    p.add_argument('--t_emb_dim', type=int, default=256)
    p.add_argument('--nz', type=int, default=100)
    p.add_argument('--use_bf16', action='store_true', default=False)

    # Hardware
    p.add_argument('--gpu_chose', type=int, default=0)
    return p.parse_args()

if __name__ == '__main__':
    args = build_argparser()
    # Basic validation of modality inputs
    need = {
        "T1CE": ["--input_flair", "--input_t2", "--input_t1"],
        "FLAIR": ["--input_t1ce", "--input_t1", "--input_t2"],
        "T2": ["--input_t1ce", "--input_t1", "--input_flair"],
        "T1": ["--input_flair", "--input_t1ce", "--input_t2"],
    }[args.target_modality]
    missing = [flag for flag in need if getattr(args, flag.replace('--input_','input_')) is None]
    if missing:
        raise SystemExit(f"Missing inputs for {args.target_modality}: {', '.join(missing)}")

    predict_volume(args)