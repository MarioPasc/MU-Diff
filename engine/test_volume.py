import os
import argparse
import torch
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive

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
    return slices, start, end

def reconstruct_volume_from_slices(predicted_slices, original_shape, start_slice, end_slice):
    """
    Reconstruct a 3D volume from predicted slices, filling non-predicted slices with zeros.
    """
    volume = np.zeros(original_shape, dtype=np.float32)
    for i, slice_2d in enumerate(predicted_slices):
        slice_idx = start_slice + i
        if slice_idx <= end_slice and slice_idx < original_shape[2]:
            volume[:, :, slice_idx] = slice_2d
    return volume

# %% Diffusion coefficients - copied from test.py
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas

# %% Posterior sampling - copied from test.py
class Posterior_Coefficients():
    def __init__(self, args, device):
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
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior_combine(coefficients, x_0_1, x_0_2, x_t, t):
    def q_posterior(x_0_1, x_0_2, x_t, t):
        mean1 = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_1
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        mean2 = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_2
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        mean = (mean1 + mean2) / 2
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0_1, x_0_2, x_t, t):
        mean, _, log_var = q_posterior(x_0_1, x_0_2, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0_1, x_0_2, x_t, t)
    return sample_x_pos

def sample_from_model(coefficients, generator1, cond1, generator2, cond2, cond3, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0_1 = generator1(x, cond1, cond2, cond3, t_time, latent_z)
            x_0_2 = generator2(x, cond1, cond2, cond3, t_time, latent_z, x_0_1[:, [0], :])
            x_new = sample_posterior_combine(coefficients, x_0_1[:, [0], :], x_0_2[:, [0], :], x, t)
            x = x_new.detach()
    return x

def load_checkpoint(checkpoint_dir, netG, name_of_network, device='cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt, strict=False)
    netG.eval()

def load_and_preprocess_volume(file_path, slice_half_range=80):
    """
    Load a NIfTI volume and preprocess it (normalize + extract center slices).
    Returns the processed slices and metadata for reconstruction.
    """
    print(f"Loading and preprocessing {file_path}")
    img = nib.load(file_path)
    volume = img.get_fdata()
    
    # Normalize the volume
    volume_norm = normalize_volume(volume)
    
    # Extract center slices
    slices, start_slice, end_slice = extract_center_slices(volume_norm, slice_half_range)
    
    return slices, volume.shape, img.affine, img.header, start_slice, end_slice

def predict_volume(args):
    """
    Main function to predict a target modality from input modalities and save as NIfTI.
    """
    torch.manual_seed(42)
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))

    # Initialize and load models
    print("Loading models...")
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device)

    # Load model checkpoints
    exp_path = os.path.join(args.output_path, args.exp)
    checkpoint_file = exp_path + "/{}.pth"
    load_checkpoint(checkpoint_file, gen_diffusive_1, 'gen_diffusive_1', device=device)
    load_checkpoint(checkpoint_file, gen_diffusive_2, 'gen_diffusive_2', device=device)

    # Set up diffusion coefficients
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    # Load and preprocess input volumes
    print("Loading and preprocessing input volumes...")
    
    # Map target modality to input modalities (same as BratsDataset)
    modality_orders = {
        "T1CE": ["FLAIR", "T2", "T1"],   # conditions for T1CE target
        "FLAIR": ["T1CE", "T1", "T2"],  # conditions for FLAIR target
        "T2": ["T1CE", "T1", "FLAIR"],  # conditions for T2 target
        "T1": ["FLAIR", "T1CE", "T2"]   # conditions for T1 target
    }
    
    if args.target_modality not in modality_orders:
        raise ValueError(f"Unsupported target modality: {args.target_modality}")
    
    input_modalities = modality_orders[args.target_modality]
    
    # Load input volumes
    input_files = [args.input_t1ce, args.input_t1, args.input_t2, args.input_flair]
    input_names = ["T1CE", "T1", "T2", "FLAIR"]
    
    # Create mapping from modality name to file path
    modality_to_file = {name: file for name, file in zip(input_names, input_files) if file is not None}
    
    # Check that we have all required input modalities
    for modality in input_modalities:
        if modality not in modality_to_file:
            raise ValueError(f"Required input modality {modality} not provided")
    
    # Load and preprocess all input volumes
    input_volumes = {}
    reference_shape = None
    reference_affine = None
    reference_header = None
    
    for modality in input_modalities:
        file_path = modality_to_file[modality]
        slices, shape, affine, header, start_slice, end_slice = load_and_preprocess_volume(
            file_path, args.slice_half_range
        )
        input_volumes[modality] = {
            'slices': slices,
            'shape': shape,
            'start_slice': start_slice,
            'end_slice': end_slice
        }
        
        # Use first volume as reference for output
        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine
            reference_header = header
        
        # Verify all volumes have the same shape
        if shape != reference_shape:
            print(f"Warning: Volume {modality} has different shape {shape} vs reference {reference_shape}")

    print(f"Loaded {len(input_volumes)} input volumes")
    print(f"Processing {len(input_volumes[input_modalities[0]]['slices'])} slices")

    # Process slices through the model
    print("Running inference...")
    predicted_slices = []
    
    num_slices = len(input_volumes[input_modalities[0]]['slices'])
    
    for slice_idx in range(num_slices):
        # Prepare input for this slice
        cond_tensors = []
        for modality in input_modalities:
            slice_2d = input_volumes[modality]['slices'][slice_idx]
            # Convert to tensor and add batch and channel dimensions
            slice_tensor = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            cond_tensors.append(slice_tensor.to(device))
        
        # Stack the three condition modalities
        cond1, cond2, cond3 = cond_tensors
        
        # Generate random noise for diffusion
        x_t = torch.randn_like(cond1)
        
        # Run diffusion sampling
        with torch.no_grad():
            fake_sample = sample_from_model(
                pos_coeff, gen_diffusive_1, cond1, gen_diffusive_2, cond2, cond3,
                args.num_timesteps, x_t, T, args
            )
        
        # Convert back to numpy
        predicted_slice = fake_sample.cpu().numpy().squeeze()
        predicted_slices.append(predicted_slice)
        
        if (slice_idx + 1) % 10 == 0:
            print(f"Processed {slice_idx + 1}/{num_slices} slices")

    # Reconstruct full volume
    print("Reconstructing volume...")
    start_slice = input_volumes[input_modalities[0]]['start_slice']
    end_slice = input_volumes[input_modalities[0]]['end_slice']
    
    predicted_volume = reconstruct_volume_from_slices(
        predicted_slices, reference_shape, start_slice, end_slice
    )

    # Save as NIfTI
    print(f"Saving predicted {args.target_modality} volume...")
    output_img = nib.Nifti1Image(predicted_volume, reference_affine, reference_header)
    
    # Create output filename
    output_filename = f"predicted_{args.target_modality.lower()}.nii.gz"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the file
    nib.save(output_img, output_path)
    
    print(f"Successfully saved predicted volume to: {output_path}")
    print(f"Volume shape: {predicted_volume.shape}")
    print(f"Predicted slices range: {start_slice} to {end_slice}")
    print(f"Target modality: {args.target_modality}")
    print(f"Input modalities used: {input_modalities}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MU-Diff Volume Prediction')
    
    # Input files
    parser.add_argument('--input_t1ce', type=str, help='Path to T1CE NIfTI file')
    parser.add_argument('--input_t1', type=str, help='Path to T1 NIfTI file')
    parser.add_argument('--input_t2', type=str, help='Path to T2 NIfTI file')
    parser.add_argument('--input_flair', type=str, help='Path to FLAIR NIfTI file')
    
    # Target and output
    parser.add_argument('--target_modality', type=str, required=True,
                        choices=['T1CE', 'FLAIR', 'T2', 'T1'],
                        help='Which modality to synthesize')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the predicted volume')
    
    # Model and experiment settings
    parser.add_argument('--exp', type=str, required=True,
                        help='Experiment name (used to locate saved model weights)')
    parser.add_argument('--output_path', type=str, default='./results',
                        help='Base path where experiment results are stored')
    
    # Preprocessing settings
    parser.add_argument('--slice_half_range', type=int, default=80,
                        help='Number of slices to take on each side of the volume center')
    
    # Model architecture parameters (must match training)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels_dae', type=int, default=128)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none')
    parser.add_argument('--progressive_input', type=str, default='residual')
    parser.add_argument('--progressive_combine', type=str, default='sum')
    
    # Diffusion parameters (must match training)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.0)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    # Embedding parameters (must match training)
    parser.add_argument('--embedding_type', type=str, default='positional')
    parser.add_argument('--fourier_scale', type=float, default=16.0)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--nz', type=int, default=100)
    
    # Hardware settings
    parser.add_argument('--gpu_chose', type=int, default=0)
    
    args = parser.parse_args()
    
    # Validate that we have the required input files for the target modality
    modality_orders = {
        "T1CE": ["FLAIR", "T2", "T1"],
        "FLAIR": ["T1CE", "T1", "T2"],
        "T2": ["T1CE", "T1", "FLAIR"],
        "T1": ["FLAIR", "T1CE", "T2"]
    }
    
    required_modalities = modality_orders[args.target_modality]
    input_files = {
        "T1CE": args.input_t1ce,
        "T1": args.input_t1,
        "T2": args.input_t2,
        "FLAIR": args.input_flair
    }
    
    missing_files = []
    for modality in required_modalities:
        if input_files[modality] is None:
            missing_files.append(f"--input_{modality.lower()}")
    
    if missing_files:
        parser.error(f"Missing required input files for target {args.target_modality}: {', '.join(missing_files)}")
    
    # Check that input files exist
    for modality in required_modalities:
        file_path = input_files[modality]
        if not os.path.isfile(file_path):
            parser.error(f"Input file does not exist: {file_path}")
    
    predict_volume(args)
