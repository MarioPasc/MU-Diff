import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import numpy as np
import torchvision # type: ignore
from PIL import Image
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive
from dataset.dataset_brats import BratsDataset
from engine.train import _as_int_list
from torch.cuda.amp import autocast
# from torch.utils.checkpoint import checkpoint  # not typically used in eval

# Wrapper class to maintain compatibility with existing test loop
class BratsDatasetWrapper:
    """
    Wrapper for BratsDataset to match the expected format of the existing test loop.
    Converts (cond_stack, target_tensor) to (x1, x2, x3, x4) format.
    """
    def __init__(self, split="test", base_path="data/BRATS", target_modality="T1CE"):
        self.dataset = BratsDataset(split=split, base_path=base_path, target_modality=target_modality)
        self.target_modality = target_modality
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        cond_stack, target_tensor = self.dataset[idx]
        # cond_stack has shape [3, H, W] for 3 condition images
        # target_tensor has shape [1, H, W] for target image
        
        # Split condition stack into individual modalities
        x1 = cond_stack[0:1]  # First condition modality [1, H, W]
        x2 = cond_stack[1:2]  # Second condition modality [1, H, W]
        x3 = cond_stack[2:3]  # Third condition modality [1, H, W]
        x4 = target_tensor    # Target modality [1, H, W]
        
        return x1, x2, x3, x4

# %% Diffusion coefficients
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


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
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


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


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
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)  # .to(x.device)

            # autocast for inference
            with autocast('cuda'):
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


def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


# %%
def sample_and_test(args):
    torch.manual_seed(42)
    # device = 'cuda:0'
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))

    to_range_0_1 = lambda x: (x + 1.) / 2.

    # loading dataset
    phase = 'test'

    # Initializing and loading network
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device)

    exp = args.exp
    output_dir = args.output_path
    exp_path = os.path.join(output_dir, exp)

    checkpoint_file = exp_path + "/{}.pth"
    load_checkpoint(checkpoint_file, gen_diffusive_1, 'gen_diffusive_1', device=device)

    load_checkpoint(checkpoint_file, gen_diffusive_2, 'gen_diffusive_2', device=device)

    dataset = BratsDatasetWrapper(split='test', base_path=args.input_path, target_modality=args.target_modality)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4)

    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    save_dir = exp_path + "/generated_samples"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare output directories for predicted and ground truth images
    pred_dir = os.path.join(save_dir, "pred")
    gt_dir = os.path.join(save_dir, "gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Collect all predictions and ground truth for global scaling
    all_pred_slices = []
    all_gt_slices = []
    all_cond_data = []  # For debugging/visualization if needed

    print(f"Processing {len(data_loader)} test samples...")

    for iteration, (x1, x2, x3, y) in enumerate(data_loader):
        cond_data1 = x1.to(device, non_blocking=True)
        cond_data2 = x2.to(device, non_blocking=True)
        cond_data3 = x3.to(device, non_blocking=True)
        real_data = y.to(device, non_blocking=True)

        x1_t = torch.randn_like(real_data)

        # Generate fake sample using the two-generator approach
        fake_sample = sample_from_model(pos_coeff, gen_diffusive_1, cond_data1, gen_diffusive_2, cond_data2,
                                         cond_data3,
                                         args.num_timesteps, x1_t, T, args)

        # Original normalization for saving concatenated samples
        fake_sample_norm = to_range_0_1(fake_sample)
        fake_sample_norm = fake_sample_norm / fake_sample_norm.mean()
        real_data_norm = to_range_0_1(real_data)
        real_data_norm = real_data_norm / real_data_norm.mean()
        x1_norm = to_range_0_1(x1)
        x1_norm = x1_norm / x1_norm.mean()
        x2_norm = to_range_0_1(x2)
        x2_norm = x2_norm / x2_norm.mean()
        x3_norm = to_range_0_1(x3)
        x3_norm = x3_norm / x3_norm.mean()

        # Save original format (synthetic samples only) - preserving original functionality
        torchvision.utils.save_image(fake_sample_norm, '{}/{}_samples_{}.jpg'.format(save_dir, phase, iteration),
                                     normalize=True)

        # Convert to numpy for individual image saving
        pred_slice = fake_sample.cpu().numpy().squeeze()
        gt_slice = real_data.cpu().numpy().squeeze()
        
        # Store for global scaling
        all_pred_slices.append(pred_slice)
        all_gt_slices.append(gt_slice)
        all_cond_data.append({
            'x1': x1.cpu().numpy().squeeze(),
            'x2': x2.cpu().numpy().squeeze(), 
            'x3': x3.cpu().numpy().squeeze()
        })

        if iteration % 50 == 0:
            print(f"Processed {iteration}/{len(data_loader)} samples")

    # Determine global intensity range for scaling images (to avoid per-slice normalization)
    print("Computing global intensity range...")
    all_pred_array = np.concatenate([p.flatten() for p in all_pred_slices])
    all_gt_array = np.concatenate([g.flatten() for g in all_gt_slices])
    global_min = float(min(all_pred_array.min(), all_gt_array.min()))
    global_max = float(max(all_pred_array.max(), all_gt_array.max()))
    
    if global_max <= global_min:
        global_min, global_max = 0.0, 1.0  # default range if images are constant
    
    print(f"Global intensity range: [{global_min:.4f}, {global_max:.4f}]")

    # Save slices as PNG images with global scaling
    print("Saving individual PNG images...")
    for i, (pred_slice, gt_slice) in enumerate(zip(all_pred_slices, all_gt_slices)):
        # Scale to [0,255] using global min/max for consistency
        pred_img = np.clip((pred_slice - global_min) / (global_max - global_min) * 255.0, 0, 255).astype(np.uint8)
        gt_img = np.clip((gt_slice - global_min) / (global_max - global_min) * 255.0, 0, 255).astype(np.uint8)
        
        Image.fromarray(pred_img).save(os.path.join(pred_dir, f"pred_{i:05d}.png"))
        Image.fromarray(gt_img).save(os.path.join(gt_dir, f"gt_{i:05d}.png"))

    print(f"Successfully completed testing!")
    print(f"Saved {len(all_pred_slices)} predicted slices to '{pred_dir}'")
    print(f"Saved {len(all_gt_slices)} ground truth slices to '{gt_dir}'")
    print(f"Original format samples saved to '{save_dir}'")
    print(f"Target modality: {args.target_modality}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('mudiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int, default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', default='data/BRATS', help='path to input data (base path to dataset)')
    parser.add_argument('--output_path', default='./results', help='path to output saves')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')

    # optimizaer parameters
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--gpu_chose', type=int, default=0)

    parser.add_argument('--source', type=str, default='T2',
                        help='source contrast')
    parser.add_argument('--target_modality', type=str, default='T1CE',
                        help='Which modality to synthesize (T1, T2, FLAIR, or T1CE)')
    args = parser.parse_args()

    args.attn_resolutions = _as_int_list(args.attn_resolutions)
    args.fir_kernel       = _as_int_list(args.fir_kernel)
    sample_and_test(args)

