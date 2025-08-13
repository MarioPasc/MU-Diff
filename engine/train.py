import argparse
import torch
import numpy as np

import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")   # serialized kernels → accurate trace
os.environ.setdefault("NCCL_DEBUG", "WARN")          # INFO if you want more detail
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # better allocator behavior


from backbones.dense_layer import conv2d

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision  # type: ignore
from dataset.dataset_brats import BratsDataset

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr

from torch.amp import autocast, GradScaler 
from torch.utils.checkpoint import checkpoint

import sys

class BratsDatasetWrapper:
    """
    Wrapper for BratsDataset to match the expected format of the existing training loop.
    Converts (cond_stack, target_tensor) to (x1, x2, x3, x4) format.
    
    The BratsDataset orders modalities based on target_modality:
    - For T1CE target: conditions are [FLAIR, T2, T1], target is T1CE
    - For FLAIR target: conditions are [T1CE, T1, T2], target is FLAIR  
    - For T2 target: conditions are [T1CE, T1, FLAIR], target is T2
    - For T1 target: conditions are [FLAIR, T1CE, T2], target is T1
    """
    def __init__(self, split="train", base_path="data/BRATS", target_modality="T1CE"):
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

# Memory logging helper (rank 0 only)
RANK = int(os.environ.get("RANK", "0"))

def log_cuda(tag):
    if RANK == 0 and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        maxa = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[MEM] {tag}: alloc={alloc:.0f}MB reserved={reserv:.0f}MB max={maxa:.0f}MB", flush=True)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


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


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = torch.cumprod(self.a_s, dim=0).to(device)               
        self.sigmas_cum = torch.sqrt(1.0 - self.a_s_cum ** 2).to(device)        
        
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


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
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            with autocast('cuda', dtype=torch.float16 if not args.use_bf16 else torch.bfloat16):
                x_0_1 = generator1(x, cond1, cond2, cond3, t_time, latent_z)
                x_0_2 = generator2(x, cond1, cond2, cond3, t_time, latent_z, x_0_1[:, [0], :])
                x_new = sample_posterior_combine(coefficients, x_0_1[:, [0], :], x_0_2[:, [0], :], x, t)
            x = x_new.detach()
    return x


def uncer_loss(mean, var, label):
    loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
    loss2 = var
    loss = .5 * (loss1 + loss2)
    return loss.mean()


# %%
def train_mudiff(rank, gpu, args):
    from backbones.discriminator import Discriminator_large

    from backbones.ncsnpp_generator_adagn_feat import NCSNpp
    from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive

    from utils.EMA import EMA

    # rank = args.node_rank * args.num_process_per_node + gpu

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    torch.backends.cudnn.benchmark = True

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    dataset = BratsDatasetWrapper(split="train", base_path=args.input_path, target_modality=args.target_modality)
    dataset_val = BratsDatasetWrapper(split="val", base_path=args.input_path, target_modality=args.target_modality)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                  num_replicas=args.world_size,
                                                                  rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  sampler=val_sampler,
                                                  drop_last=True)

    val_l1_loss = np.zeros([2, args.num_epoch, len(data_loader_val)])
    val_psnr_values = np.zeros([2, args.num_epoch, len(data_loader_val)])
    print('train data size:' + str(len(data_loader)))
    print('val data size:' + str(len(data_loader_val)))
    print('target modality:' + str(args.target_modality))
    to_range_0_1 = lambda x: (x + 1.) / 2.
    critic_criterian = nn.BCEWithLogitsLoss(reduction='none')

    # networks performing reverse denoising
    gen_diffusive_1 = NCSNpp(args).to(device, memory_format=torch.channels_last)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device, memory_format=torch.channels_last)

    args.num_channels = 1
    att_conv = conv2d(64 * 8, 1, 1, padding=0).cuda()

    disc_diffusive_2 = Discriminator_large(nc=2, ngf=args.ngf,
                                           t_emb_dim=args.t_emb_dim,
                                           act=nn.LeakyReLU(0.2)).to(device, memory_format=torch.channels_last)

    broadcast_params(gen_diffusive_1.parameters())
    broadcast_params(gen_diffusive_2.parameters())

    broadcast_params(disc_diffusive_2.parameters())

    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    # AMP scalers
    scaler_d = GradScaler('cuda')
    scaler_g = GradScaler('cuda')

    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)

    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch,
                                                                           eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch,
                                                                           eta_min=1e-5)

    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch,
                                                                            eta_min=1e-5)

    # ddp
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])

    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])

    exp = args.exp

    output_path = args.output_path

    exp_path = output_path
    if rank == 0:
        # Restore provenance copy: save a snapshot of this script and the backbones/ dir
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            # Use absolute path for backbones directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            backbones_dir = os.path.join(os.path.dirname(script_dir), 'backbones')
            if os.path.exists(backbones_dir):
                dest_backbones = os.path.join(exp_path, 'backbones')
                if not os.path.exists(dest_backbones):
                    shutil.copytree(backbones_dir, dest_backbones)
            else:
                print(f"Warning: backbones directory not found at {backbones_dir}")
        else:
            os.makedirs(exp_path, exist_ok=True)
        
        
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])

        # load G

        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2'])

        # load D

        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # Helpers for optional checkpointing of generator forwards
    def run_g1(x, c1, c2, c3, t, z):
        return gen_diffusive_1(x, c1, c2, c3, t, z)

    def run_g2(x, c1, c2, c3, t, z, prev):
        return gen_diffusive_2(x, c1, c2, c3, t, z, prev)

    for epoch in range(init_epoch, args.num_epoch + 1):
        # train_sampler.set_epoch(epoch)

        for iteration, (x1, x2, x3, x4) in enumerate(data_loader):
            for p in disc_diffusive_2.parameters():
                p.requires_grad = True

            optimizer_disc_diffusive_2.zero_grad(set_to_none=True)

            # sample from p(x_0)
            cond_data1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data2 = x2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data3 = x3.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            real_data = x4.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            log_cuda('after data load (D step)')

            t2 = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data, t2)
            x2_t.requires_grad = True

            # train discriminator with real
            with autocast('cuda', dtype=torch.float16 if not args.use_bf16 else torch.bfloat16):
                D2_real, _ = disc_diffusive_2(x2_t, t2, x2_tp1.detach())
                errD2_real2 = F.softplus(-D2_real).mean()

            errD_real2 = errD2_real2
            scaler_d.scale(errD_real2).backward(retain_graph=True)

            if args.lazy_reg is None:

                grad2_real = torch.autograd.grad(
                    outputs=D2_real.sum(), inputs=x2_t, create_graph=True
                )[0]
                grad2_penalty = (
                        grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty2 = args.r1_gamma / 2 * grad2_penalty
                scaler_d.scale(grad_penalty2).backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad2_real = torch.autograd.grad(
                        outputs=D2_real.sum(), inputs=x2_t, create_graph=True
                    )[0]
                    grad2_penalty = (
                            grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty2 = args.r1_gamma / 2 * grad2_penalty
                    scaler_d.scale(grad_penalty2).backward()

            # train with fake (wrap generator forwards in no_grad to avoid building G graph)
            latent_z2 = torch.randn(batch_size, nz, device=device)
            # Fake generation for D under no_grad to avoid retaining G graph
            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16 if not args.use_bf16 else torch.bfloat16):
                    x_tp1_det = x2_tp1.detach()
                    x2_0_predict_diff_g1 = run_g1(x_tp1_det, cond_data1, cond_data2, cond_data3, t2, latent_z2)
                    x2_0_predict_diff_g2 = run_g2(x_tp1_det, cond_data1, cond_data2, cond_data3, t2, latent_z2,
                                                  x2_0_predict_diff_g1[:, [0], :])
                    x2_pos_sample_g1 = sample_posterior(pos_coeff, x2_0_predict_diff_g1[:, [0], :], x2_tp1, t2)
                    x2_pos_sample_g2 = sample_posterior(pos_coeff, x2_0_predict_diff_g2[:, [0], :], x2_tp1, t2)
                log_cuda('after fake generation (D step)')
            with autocast('cuda', dtype=torch.float16 if not args.use_bf16 else torch.bfloat16):
                output2_g1, _ = disc_diffusive_2(x2_pos_sample_g1.detach(), t2, x2_tp1.detach())
                output2_g2, _ = disc_diffusive_2(x2_pos_sample_g2.detach(), t2, x2_tp1.detach())
                errD2_fake2_g1 = (F.softplus(output2_g1)).mean()
                errD2_fake2_g2 = (F.softplus(output2_g2)).mean()
                errD_fake2 = errD2_fake2_g1 + errD2_fake2_g2

            scaler_d.scale(errD_fake2).backward()

            scaler_d.step(optimizer_disc_diffusive_2)
            scaler_d.update()

            for p in disc_diffusive_2.parameters():
                p.requires_grad = False

            cond_data1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data2 = x2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data3 = x3.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            real_data = x4.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            optimizer_gen_diffusive_1.zero_grad(set_to_none=True)
            optimizer_gen_diffusive_2.zero_grad(set_to_none=True)

            t2 = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            # sample x_t and x_tp1
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data, t2)

            latent_z2 = torch.randn(batch_size, nz, device=device)

            with autocast('cuda', dtype=torch.float16 if not args.use_bf16 else torch.bfloat16):
                x_tp1_det = x2_tp1.detach()
                if args.use_grad_checkpoint:
                    x_in = x_tp1_det.requires_grad_()
                    x2_0_predict_diff_g1 = checkpoint(run_g1, x_in, cond_data1, cond_data2, cond_data3, t2, latent_z2)
                else:
                    x2_0_predict_diff_g1 = run_g1(x_tp1_det, cond_data1, cond_data2, cond_data3, t2, latent_z2)

                if args.use_grad_checkpoint:
                    x2_0_predict_diff_g2 = checkpoint(
                        run_g2,
                        x_tp1_det,
                        cond_data1,
                        cond_data2,
                        cond_data3,
                        t2,
                        latent_z2,
                        x2_0_predict_diff_g1[:, [0], :]
                    )
                else:
                    x2_0_predict_diff_g2 = run_g2(x_tp1_det, cond_data1, cond_data2, cond_data3, t2, latent_z2,
                                                  x2_0_predict_diff_g1[:, [0], :])

                # sampling q(x_t | x_0_predict, x_t+1)
                x2_pos_sample_g1 = sample_posterior(pos_coeff, x2_0_predict_diff_g1[:, [0], :], x2_tp1, t2)
                x2_pos_sample_g2 = sample_posterior(pos_coeff, x2_0_predict_diff_g2[:, [0], :], x2_tp1, t2)

                # D output for fake sample x_pos_sample
                output2_g1, att_feat_g1 = disc_diffusive_2(x2_pos_sample_g1, t2, x2_tp1.detach())
                output2_g2, att_feat_g2 = disc_diffusive_2(x2_pos_sample_g2, t2, x2_tp1.detach())

                att_map_g1 = torch.sigmoid(att_conv(att_feat_g1))
                att_map_g1 = F.interpolate(att_map_g1, size=(256, 256), mode='bilinear', align_corners=False)

                att_map_g2 = torch.sigmoid(att_conv(att_feat_g2))
                att_map_g2 = F.interpolate(att_map_g2, size=(256, 256), mode='bilinear', align_corners=False)

                mask_loss_1 = (att_map_g2 * critic_criterian(x2_pos_sample_g1, torch.sigmoid(x2_pos_sample_g2))).mean()
                mask_loss_2 = (att_map_g1 * critic_criterian(x2_pos_sample_g2, torch.sigmoid(x2_pos_sample_g1))).mean()

                mask_loss = mask_loss_1 + mask_loss_2

                errG2 = F.softplus(-output2_g1).mean()
                errG4 = F.softplus(-output2_g2).mean()
                errG_adv = errG2 + errG4

                errG1_2_L1 = F.l1_loss(x2_0_predict_diff_g1[:, [0], :], real_data)
                errG2_2_L1 = F.l1_loss(x2_0_predict_diff_g2[:, [0], :], real_data)
                errG_L1 = errG1_2_L1 + errG2_2_L1

                errG = errG_adv + (args.lambda_l1_loss * errG_L1) + (args.lambda_mask_loss * mask_loss)

            scaler_g.scale(errG).backward()
            scaler_g.step(optimizer_gen_diffusive_1)
            scaler_g.step(optimizer_gen_diffusive_2)
            scaler_g.update()

            global_step += 1
            if iteration % 100 == 0 and rank == 0:
                log_cuda('after G update')
                print('epoch {} iteration{},  G-Adv: {}, G-Sum: {}'.format(epoch, iteration, errG_adv.item(), errG.item()))

        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()

            scheduler_disc_diffusive_2.step()

        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x2_pos_sample_g1,
                                             os.path.join(exp_path, 'xposg1_epoch_{}.png'.format(epoch)),
                                             normalize=True)
                torchvision.utils.save_image(x2_pos_sample_g2,
                                             os.path.join(exp_path, 'xposg2_epoch_{}.png'.format(epoch)),
                                             normalize=True)
            # concatenate noise and source contrast
            x2_t = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, gen_diffusive_1, cond_data1, gen_diffusive_2, cond_data2,
                                            cond_data3,
                                            args.num_timesteps, x2_t, T, args)

            fake_sample = torch.cat((real_data, fake_sample), axis=-1)

            torchvision.utils.save_image(fake_sample,
                                         os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)),
                                         normalize=True)

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'gen_diffusive_1_dict': gen_diffusive_1.state_dict(),
                               'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                               'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(),
                               'gen_diffusive_2_dict': gen_diffusive_2.state_dict(),
                               'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                               'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(),
                               'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                               'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(),
                               'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                               }

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)

                # e = epoch
                # path1 = os.path.join(exp_path, f'gen_diffusive_1_{e}.pth')
                # path2 = os.path.join(exp_path, f'gen_diffusive_2_{e}.pth')
                # torch.save(gen_diffusive_1.state_dict(), path1)
                # torch.save(gen_diffusive_2.state_dict(), path2)

                torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1.pth'))
                torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2.pth'))

                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)

        for iteration, (x1_val, x2_val, x3_val, x4_val) in enumerate(data_loader_val):
            cond_data1_val = x1_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data2_val = x2_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data3_val = x3_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            real_data_val = x4_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            x_t = torch.randn_like(real_data)

            fake_sample_val = sample_from_model(pos_coeff, gen_diffusive_1, cond_data1_val, gen_diffusive_2,
                                                cond_data2_val,
                                                cond_data3_val,
                                                args.num_timesteps, x_t, T, args)

            # diffusion steps
            fake_sample_val = to_range_0_1(fake_sample_val); fake_sample_val = fake_sample_val / fake_sample_val.mean()
            real_data_val = to_range_0_1(real_data_val); real_data_val = real_data_val / real_data_val.mean()

            fake_sample_val = fake_sample_val.cpu().numpy()
            real_data_val = real_data_val.cpu().numpy()
            val_l1_loss[0, epoch, iteration] = abs(fake_sample_val - real_data_val).mean()

            val_psnr_values[0, epoch, iteration] = psnr(real_data_val, fake_sample_val, data_range=real_data_val.max())

        print(np.nanmean(val_psnr_values[0, epoch, :]))
        np.save('{}/val_l1_loss.npy'.format(exp_path), val_l1_loss)
        np.save('{}/val_psnr_values.npy'.format(exp_path), val_psnr_values)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num

    # Map global rank -> local index (0..num_process_per_node-1)
    local_idx = rank % args.num_process_per_node

    # Robustly pin each child to exactly ONE CUDA device
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    print(f"[rank: {rank}] pre-pinning to GPU, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    if vis:
        devs = [d.strip() for d in vis.split(",") if d.strip() != ""]
        if local_idx < len(devs):
            # Restrict visibility to a single GPU for this child
            os.environ["CUDA_VISIBLE_DEVICES"] = devs[local_idx]
            gpu = 0  # in this process, the only visible device is index 0
        else:
            gpu = local_idx
    else:
        gpu = local_idx

    torch.cuda.set_device(gpu)

    print(f"[rank: {rank}] pinned to GPU {gpu}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)


    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()

def _as_int_list(v):
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    if isinstance(v, str):
        # accept "16" or "16,32" or "16 32"
        parts = v.replace(',', ' ').split()
        return [int(x) for x in parts]
    return [int(v)]

# %%
if __name__ == '__main__':
    
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser('mudiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
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
    parser.add_argument('--input_path', default='/data/BRATS/')
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=1, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5,
                        help='weightening of l1 loss part of diffusion ans cycle models')
    parser.add_argument('--lambda_mask_loss', type=float, default=0.1,
                        help='weightening of l1 loss part of diffusion ans cycle models')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                        help='weighting of adversarial loss for generators')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--target_modality', type=str, default='T1CE',
                        help='Which modality to synthesize (T1, T2, FLAIR, or T1CE)')
    parser.add_argument('--port_num', type=str, default='6021',
                        help='port selection for code')

    # New flag to enable gradient checkpointing for memory savings
    parser.add_argument('--use_grad_checkpoint', action='store_true', default=False,
                        help='Enable gradient checkpointing on generator forwards to save memory')
    parser.add_argument('--use_bf16', action='store_true', default=False,
                        help='Use bfloat16 autocast for reduced memory (default off)')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    args.attn_resolutions = _as_int_list(args.attn_resolutions)
    args.fir_kernel       = _as_int_list(args.fir_kernel)
    if size > 1:
        processes = []
        for rank in range(size):
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            p = Process(target=init_processes, args=(global_rank, global_size, train_mudiff, args), daemon=False)

            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        # Propaga fallo si algún hijo cayó
        bad = [p.exitcode for p in processes if p.exitcode not in (0, None)]
        if bad:
            sys.exit(1)
    else:
        init_processes(0, size, train_mudiff, args)
