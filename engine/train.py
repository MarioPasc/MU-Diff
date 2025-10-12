import argparse
import torch
import numpy as np
import time

import os

# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")   # serialized kernels → accurate trace; ONLY FOR DEBUGGING
os.environ.setdefault("NCCL_DEBUG", "WARN")          # INFO if you want more detail
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # better allocator behavior

# NCCL timeout configuration for distributed training
# Default: 30 minutes (1800 seconds) to handle long operations on supercomputing clusters
# Can be overridden via environment variable: export NCCL_TIMEOUT_MINUTES=60
nccl_timeout_minutes = int(os.environ.get("NCCL_TIMEOUT_MINUTES", "30"))
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")  # Enable async error handling for better debugging
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")  # Use blocking wait to catch errors immediately

# Set NCCL timeout for PyTorch distributed operations
import datetime
torch.distributed.distributed_c10d._DEFAULT_PG_TIMEOUT = datetime.timedelta(minutes=nccl_timeout_minutes)

torch.autograd.set_detect_anomaly(False)  # TEMPORARY

RETAIN_GRAPH: bool = False  # whether to retain graph in D step for debugging only; should be False for normal training

from backbones.dense_layer import conv2d

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision  # type: ignore
from utils.train_utils import epoch_visual_report
from dataset.dataset_brats import BratsDataset

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr

from torch.amp import autocast, GradScaler #type: ignore
from torch.utils.checkpoint import checkpoint

import sys


# at top-level
if os.environ.get("MUDIFF_DEBUG_SYNC", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

torch.backends.cudnn.benchmark = True

try:
    # Newer PyTorch
    torch.set_float32_matmul_precision("high") # type: ignore
except AttributeError:
    # Older PyTorch: approximate equivalent via TF32 flags
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True # type: ignore


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

# Pretty, structured logger for per-step and per-epoch summaries (rank 0)
def _mem_stats():
    if not torch.cuda.is_available():
        return dict(alloc_mb=0.0, reserved_mb=0.0, max_alloc_mb=0.0)
    return dict(
        alloc_mb=torch.cuda.memory_allocated() / 1024**2,
        reserved_mb=torch.cuda.memory_reserved() / 1024**2,
        max_alloc_mb=torch.cuda.max_memory_allocated() / 1024**2,
    )

def _lr_of(optim):
    try:
        for g in optim.param_groups:
            return float(g.get('lr', 0.0))
    except Exception:
        return 0.0
    return 0.0

def log_step(scope, epoch, iteration, global_step, losses, lrs, times, batch_size, world_size, scaler_g=None, scaler_d=None):
    """
    scope: 'train' | 'val'
    losses: dict of scalar floats
    lrs: dict of learning rates
    times: dict with keys like 'data', 'batch', optional 'iter'
    """
    if RANK != 0:
        return
    mem = _mem_stats()
    # Build a compact, aligned message
    parts = [
        f"[{scope.upper()}] E{epoch:03d} I{iteration:05d} GS{global_step:07d}",
        f"bs={batch_size}x{world_size}",
        f"time(b/d)={times.get('batch', 0):.3f}/{times.get('data', 0):.3f}s",
        f"mem(a/r/m)={mem['alloc_mb']:.0f}/{mem['reserved_mb']:.0f}/{mem['max_alloc_mb']:.0f}MB",
    ]
    try:
        bt = float(times.get('batch', 0.0))
        if bt > 0:
            ips = (batch_size * world_size) / bt
            parts.append(f"ips={ips:.1f}")
    except Exception:
        pass
    if lrs:
        lr_str = " ".join([f"{k}={v:.2e}" for k, v in lrs.items()])
        parts.append(f"lr: {lr_str}")
    if scaler_g is not None:
        try:
            parts.append(f"scale_g={float(scaler_g.get_scale()):.1f}")
        except Exception:
            pass
    if scaler_d is not None:
        try:
            parts.append(f"scale_d={float(scaler_d.get_scale()):.1f}")
        except Exception:
            pass
    if losses:
        loss_str = " ".join([f"{k}={v:.4f}" for k, v in losses.items()])
        parts.append(f"loss: {loss_str}")
    print(" | ".join(parts), flush=True)

def log_epoch_summary(epoch, global_step, epoch_avg_losses, val_metrics=None):
    if RANK != 0:
        return
    mem = _mem_stats()
    print("\n===== Epoch Summary =====", flush=True)
    print(f"Epoch {epoch} @ global_step {global_step}", flush=True)
    if epoch_avg_losses:
        loss_str = ", ".join([f"{k}={v:.4f}" for k, v in epoch_avg_losses.items()])
        print(f"Train avg: {loss_str}", flush=True)
    if val_metrics:
        vstr = ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
        print(f"Val: {vstr}", flush=True)
    print(f"GPU mem: alloc={mem['alloc_mb']:.0f}MB reserved={mem['reserved_mb']:.0f}MB peak={mem['max_alloc_mb']:.0f}MB", flush=True)
    print("========================\n", flush=True)

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
            with autocast('cuda', dtype=torch.float16 if not opt.use_bf16 else torch.bfloat16):
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
    
    # Data loader worker configuration
    # Reduced default workers from 8 to 4 to avoid file descriptor exhaustion on clusters
    tw = int(os.environ.get("MU_TRAIN_WORKERS", "4"))
    vw = int(os.environ.get("MU_VAL_WORKERS",   "2"))
    prefetch = int(os.environ.get("MU_PREFETCH", "2"))
    persistent = bool(int(os.environ.get("MU_PERSISTENT", "0"))) and tw > 0
    # Reduced timeout from 600s to 120s to catch hangs earlier and prevent NCCL timeouts
    timeout_s = int(os.environ.get("MU_DL_TIMEOUT", "120"))
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=tw,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch if tw > 0 else None,
        sampler=train_sampler,
        drop_last=True,
        timeout=timeout_s if tw > 0 else 0,
    )


    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                  num_replicas=args.world_size,
                                                                  rank=rank)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=vw,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,
        timeout=timeout_s if vw > 0 else 0,
    )

    val_l1_loss = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    val_psnr_values = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    print('train data size:' + str(len(data_loader)))
    print('val data size:' + str(len(data_loader_val)))
    print('target modality:' + str(args.target_modality))
    to_range_0_1 = lambda x: (x + 1.) / 2.
    critic_criterian = nn.BCEWithLogitsLoss(reduction='none')

    # networks performing reverse denoising
    gen_diffusive_1 = NCSNpp(args).to(device, memory_format=torch.channels_last)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device, memory_format=torch.channels_last)

    args.num_channels = 1
    att_conv = conv2d(64 * 8, 1, 1, padding=0).to(device, memory_format=torch.channels_last)

    disc_diffusive_2 = Discriminator_large(nc=2, ngf=args.ngf,
                                           t_emb_dim=args.t_emb_dim,
                                           act=nn.LeakyReLU(0.2)).to(device, memory_format=torch.channels_last)

    # Note: broadcast_params removed - DDP automatically synchronizes parameters during initialization
    # This avoids redundant communication and potential synchronization issues

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
    # gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    # gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])
    # disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])
    ddp_kwargs = dict(
        device_ids=[gpu],
        broadcast_buffers=False,           # evita sync de buffers (reduce memoria/overhead)
        gradient_as_bucket_view=True,      # grads como vistas → menos copias
        static_graph=True                  # grafo estable → menos metadatos
    )
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, **ddp_kwargs)
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, **ddp_kwargs)
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, **ddp_kwargs)

    # One-time model parameter summary
    if rank == 0:
        def _count_params(m):
            p = sum(p.numel() for p in m.parameters())
            t = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return p, t
        p1, t1 = _count_params(gen_diffusive_1)
        p2, t2 = _count_params(gen_diffusive_2)
        pd, td = _count_params(disc_diffusive_2)
        print(f"[MODEL] G1 params: {p1:,} (trainable {t1:,}); G2 params: {p2:,} (trainable {t2:,}); D params: {pd:,} (trainable {td:,})", flush=True)

        # Print distributed training configuration
        print(f"\n[CONFIG] Distributed Training Configuration:", flush=True)
        print(f"  - World size: {args.world_size}", flush=True)
        print(f"  - NCCL timeout: {nccl_timeout_minutes} minutes", flush=True)
        print(f"  - Data loader workers (train/val): {tw}/{vw}", flush=True)
        print(f"  - Data loader timeout: {timeout_s} seconds", flush=True)
        print(f"  - Prefetch factor: {prefetch}", flush=True)
        print(f"  - Persistent workers: {persistent}", flush=True)
        print(f"  - RETAIN_GRAPH: {RETAIN_GRAPH}\n", flush=True)


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
        if args.pretrained_dir:
            if rank == 0:
                print(f"[PRETRAIN] Loading generators from {args.pretrained_dir}")
            def _load_pre(model_ddp, filename):
                path = os.path.join(args.pretrained_dir, filename)
                if not os.path.isfile(path):
                    if rank == 0:
                        print(f"[PRETRAIN] File not found: {path}")
                    return
                try:
                    result = model_ddp.load_state_dict(torch.load(path, map_location=device), strict=False)
                    missing = getattr(result, 'missing_keys', None) or (result[0] if isinstance(result, (list, tuple)) else [])
                    unexpected = getattr(result, 'unexpected_keys', None) or (result[1] if isinstance(result, (list, tuple)) else [])
                    if rank == 0:
                        print(f"[PRETRAIN] {filename} loaded (missing={len(missing)} unexpected={len(unexpected)})")
                        if missing:
                            print(f"           Missing sample: {missing[:8]}{' ...' if len(missing)>8 else ''}")
                        if unexpected:
                            print(f"           Unexpected sample: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
                except Exception as e:
                    if rank == 0:
                        print(f"[PRETRAIN] Error loading {filename}: {e}")
            _load_pre(gen_diffusive_1, 'gen_diffusive_1.pth')
            _load_pre(gen_diffusive_2, 'gen_diffusive_2.pth')
            if rank == 0:
                print('[PRETRAIN] Pretrained initialization done.')

    # Helpers for optional checkpointing of generator forwards
    def run_g1(x, c1, c2, c3, t, z):
        return gen_diffusive_1(x, c1, c2, c3, t, z)

    def run_g2(x, c1, c2, c3, t, z, prev):
        return gen_diffusive_2(x, c1, c2, c3, t, z, prev)

    for epoch in range(init_epoch, args.num_epoch):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)

        # running loss accumulators for epoch summary
        ep_losses = {
            'D_total': 0.0, 'D_real': 0.0, 'D_fake': 0.0, 'R1': 0.0,
            'G_total': 0.0, 'G_adv': 0.0, 'G_L1': 0.0, 'G_mask': 0.0,
        }
        ep_count = 0
        iter_start_time = time.time()

        for iteration, (x1, x2, x3, x4) in enumerate(data_loader):
            try:
                data_time = time.time() - iter_start_time
            except Exception as e:
                if rank == 0:
                    print(f"[rank {rank}] Data loader error at iteration {iteration}: {e}", flush=True)
                raise RuntimeError(f"Data loader failed at iteration {iteration}: {e}")
            # ---- D step ----
            # Enable D grads, disable G grads (extra safety even though we use no_grad for G forwards)
            for p in disc_diffusive_2.parameters():
                p.requires_grad_(True)
            for p in gen_diffusive_1.parameters():
                p.requires_grad_(False)
            for p in gen_diffusive_2.parameters():
                p.requires_grad_(False)

            optimizer_disc_diffusive_2.zero_grad(set_to_none=True)

            # sample from p(x_0)
            cond_data1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data2 = x2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data3 = x3.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            real_data = x4.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            # log_cuda('after data load (D step)')

            t2 = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data, t2)
            x2_t.requires_grad = True

            # train discriminator with real
            with autocast('cuda', dtype=torch.float16 if not args.use_bf16 else torch.bfloat16):
                D2_real, _ = disc_diffusive_2(x2_t, t2, x2_tp1.detach())
                errD2_real2 = F.softplus(-D2_real).mean()
            errD_real2 = errD2_real2
            # scaler_d.scale(errD_real2).backward(retain_graph=RETAIN_GRAPH)

            grad_penalty2 = torch.zeros((), device=device, dtype=real_data.dtype)  # default 0 in case we skip
            if args.lazy_reg is None or (global_step % args.lazy_reg == 0):
                # re-run D at full precision to keep higher-order grads numerically stable
                with autocast('cuda', enabled=False):
                    D2_real_r1, _ = disc_diffusive_2(x2_t, t2, x2_tp1.detach())
                    # build graph for penalty so it contributes gradients to D (R1 needs create_graph=True)
                    grad2_real = torch.autograd.grad(
                        outputs=D2_real_r1.sum(), inputs=x2_t, create_graph=True, retain_graph=RETAIN_GRAPH
                    )[0]
                    grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty2 = (args.r1_gamma / 2) * grad2_penalty

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
                #log_cuda('after fake generation (D step)')
            with autocast('cuda', dtype=torch.float16 if not args.use_bf16 else torch.bfloat16):
                output2_g1, _ = disc_diffusive_2(x2_pos_sample_g1.detach(), t2, x2_tp1.detach())
                output2_g2, _ = disc_diffusive_2(x2_pos_sample_g2.detach(), t2, x2_tp1.detach())
                errD2_fake2_g1 = (F.softplus(output2_g1)).mean()
                errD2_fake2_g2 = (F.softplus(output2_g2)).mean()
                errD_fake2 = errD2_fake2_g1 + errD2_fake2_g2


            # Optional verbose tensor debug
            if args.debug_verbose and rank == 0:
                def _dbg(tag, t):
                    try:
                        print(f"[rank {rank}] {tag}: dev={t.device} req_grad={t.requires_grad} "
                              f"shape={tuple(t.shape)} is_leaf={getattr(t, 'is_leaf', 'NA')}", flush=True)
                    except Exception:
                        pass
                _dbg("fake1_D", x2_pos_sample_g1)
                _dbg("fake2_D", x2_pos_sample_g2)
                _dbg("target0", x2_t)

            # scaler_d.scale(errD_fake2).backward()
            d_total = errD_real2 + grad_penalty2 + errD_fake2
            if args.debug_verbose and rank == 0:
                print(f"[rank {rank}] d_total={d_total.item():.4f}", flush=True)
                
                
            if args.debug_verbose and rank == 0:
                def dbg(n, t):
                    print(f"[rank {rank}] {n}: dev={t.device} req_grad={t.requires_grad} "
                        f"is_leaf={getattr(t,'is_leaf','NA')} shape={tuple(t.shape)}", flush=True)
                dbg("D2_real", D2_real); dbg("D2_real_r1", D2_real_r1)
                dbg("grad2_real", grad2_real); print(f"[rank {rank}] d_total={float(d_total.detach())}", flush=True)


            scaler_d.scale(d_total).backward()
            scaler_d.step(optimizer_disc_diffusive_2)
            scaler_d.update()

            # ---- G step ----
            # Disable D grads, enable G grads
            for p in disc_diffusive_2.parameters():
                p.requires_grad_(False)
            for p in gen_diffusive_1.parameters():
                p.requires_grad_(True)
            for p in gen_diffusive_2.parameters():
                p.requires_grad_(True)

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
                H, W = x2_pos_sample_g1.shape[-2], x2_pos_sample_g1.shape[-1]
                att_map_g1 = F.interpolate(att_map_g1, size=(H, W), mode='bilinear', align_corners=False)
                
                att_map_g2 = torch.sigmoid(att_conv(att_feat_g2))
                att_map_g2 = F.interpolate(att_map_g2, size=(H, W), mode='bilinear', align_corners=False)

                if args.debug_verbose and rank == 0:
                    def _sh(name, t):
                        print(f"[rank {rank}] {name}: shape={tuple(t.shape)} "
                            f"min={t.min().item():.3g} max={t.max().item():.3g}", flush=True)

                    _sh("x2_pos_sample_g1", x2_pos_sample_g1)
                    _sh("x2_pos_sample_g2", x2_pos_sample_g2)
                    _sh("att_feat_g1", att_feat_g1)
                    _sh("att_feat_g2", att_feat_g2)
                    _sh("att_map_g1(resized)", att_map_g1)
                    _sh("att_map_g2(resized)", att_map_g2)

                # sanity checks before using them
                assert att_map_g1.shape == x2_pos_sample_g2.shape, \
                    f"att_map_g1 {att_map_g1.shape} vs x2_pos_sample_g2 {x2_pos_sample_g2.shape}"
                assert att_map_g2.shape == x2_pos_sample_g1.shape, \
                    f"att_map_g2 {att_map_g2.shape} vs x2_pos_sample_g1 {x2_pos_sample_g1.shape}"


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

            # Debug prints before G backward (rank 0 only)
            if args.debug_verbose and rank == 0:
                def _dbg2(tag, t):
                    try:
                        print(f"[rank {rank}] {tag}: dev={t.device} req_grad={t.requires_grad} "
                              f"shape={tuple(t.shape)} is_leaf={getattr(t, 'is_leaf', 'NA')}", flush=True)
                    except Exception:
                        pass
                _dbg2("out1_G", x2_pos_sample_g1)
                _dbg2("out2_G0", x2_pos_sample_g2)
                _dbg2("g_total", errG)
            scaler_g.scale(errG).backward()
            scaler_g.step(optimizer_gen_diffusive_1)
            scaler_g.step(optimizer_gen_diffusive_2)
            scaler_g.update()

            global_step += 1
            # accumulate epoch metrics
            ep_losses['D_total'] += float(d_total.detach().item())
            ep_losses['D_real']  += float(errD_real2.detach().item())
            ep_losses['D_fake']  += float(errD_fake2.detach().item())
            ep_losses['R1']      += float(grad_penalty2.detach().item()) if torch.is_tensor(grad_penalty2) else float(grad_penalty2)
            ep_losses['G_total'] += float(errG.detach().item())
            ep_losses['G_adv']   += float(errG_adv.detach().item())
            ep_losses['G_L1']    += float(errG_L1.detach().item())
            ep_losses['G_mask']  += float(mask_loss.detach().item())
            ep_count += 1

            # periodic step logging
            if args.log_every > 0 and (iteration % args.log_every == 0):
                batch_time = time.time() - (iter_start_time)
                log_step(
                    scope='train', epoch=epoch, iteration=iteration, global_step=global_step,
                    losses=dict(G=errG.item(), G_adv=errG_adv.item(), G_L1=errG_L1.item(), G_mask=mask_loss.item(),
                                D=d_total.item(), D_real=errD_real2.item(), D_fake=errD_fake2.item(), R1=float(grad_penalty2.item()) if torch.is_tensor(grad_penalty2) else float(grad_penalty2)),
                    lrs=dict(lr_g=_lr_of(optimizer_gen_diffusive_1), lr_d=_lr_of(optimizer_disc_diffusive_2)),
                    times=dict(batch=batch_time, data=data_time),
                    batch_size=batch_size, world_size=args.world_size,
                    scaler_g=scaler_g, scaler_d=scaler_d,
                )
                if rank == 0 and args.log_mem_after_update:
                    log_cuda('after G update')
                # reset timer for next iter
                iter_start_time = time.time()

            # Heartbeat logging every 50 iterations to detect hangs/desynchronization
            if iteration > 0 and iteration % 50 == 0:
                print(f"[rank {rank}] Heartbeat: epoch={epoch} iter={iteration} global_step={global_step}", flush=True)

        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()

            scheduler_disc_diffusive_2.step()

        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[MEM] epoch {epoch} peak_alloc={peak:.0f}MB", flush=True)
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

            # keep originals for reporting
            last_real_for_report = real_data.detach().clone()
            last_fake_for_report = fake_sample.detach().clone()

            preview = torch.cat((real_data, fake_sample), axis=-1)
            torchvision.utils.save_image(preview,
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

        # epoch summary logging (after stepping schedulers, before validation files saving)
        avg_losses = {}
        if ep_count > 0:
            avg_losses = {k: v / ep_count for k, v in ep_losses.items()}
            log_epoch_summary(epoch, global_step, avg_losses)

        for iteration, (x1_val, x2_val, x3_val, x4_val) in enumerate(data_loader_val):
            cond_data1_val = x1_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data2_val = x2_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            cond_data3_val = x3_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            real_data_val = x4_val.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            x_t = torch.randn_like(real_data_val)

            fake_sample_val = sample_from_model(pos_coeff, gen_diffusive_1, cond_data1_val, gen_diffusive_2,
                                                cond_data2_val,
                                                cond_data3_val,
                                                args.num_timesteps, x_t, T, args)

            # diffusion steps
            fake_sample_val = to_range_0_1(fake_sample_val); fake_sample_val = fake_sample_val / fake_sample_val.mean()
            real_data_val = to_range_0_1(real_data_val); real_data_val = real_data_val / real_data_val.mean()

            fake_sample_val = fake_sample_val.cpu().numpy()
            real_data_val = real_data_val.cpu().numpy()
            
            epoch_slot = epoch - init_epoch

            val_l1_loss[0, epoch_slot, iteration] = abs(fake_sample_val - real_data_val).mean()
            eps = 1e-8
            fake = to_range_0_1(fake_sample_val); fake = fake / (fake.mean() + eps)
            real = to_range_0_1(real_data_val);    real = real / (real.mean() + eps)
            val_psnr_values[0, epoch_slot, iteration] = psnr(real, fake, data_range=max(real.max(), eps))
        
        mean_psnr = float(np.nanmean(val_psnr_values[0, epoch_slot, :]))
        if rank == 0:
            log_step(
                scope='val', epoch=epoch, iteration=0, global_step=global_step,
                losses={}, lrs={}, times={'batch': 0.0, 'data': 0.0},
                batch_size=batch_size, world_size=args.world_size,
            )
            log_epoch_summary(epoch, global_step, epoch_avg_losses={'train_G': avg_losses.get('G_total', 0.0), 'train_D': avg_losses.get('D_total', 0.0)},
                              val_metrics={'val_psnr': mean_psnr, 'val_l1': float(np.nanmean(val_l1_loss[0, epoch_slot, :]))})
            # Enhanced epoch report
            try:
                epoch_time = time.time() - epoch_start_time
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                epoch_visual_report(
                    out_dir=exp_path,
                    epoch=epoch,
                    real_batch=last_real_for_report if 'last_real_for_report' in locals() else real_data,
                    fake_batch=last_fake_for_report if 'last_fake_for_report' in locals() else fake_sample,
                    avg_losses=avg_losses,
                    val_metrics={'val_psnr': mean_psnr, 'val_l1': float(np.nanmean(val_l1_loss[0, epoch_slot, :]))},
                    epoch_time_sec=epoch_time,
                    peak_mem_mb=peak_mem,
                    extra={'global_step': global_step}
                )
            except Exception as e:
                print(f"[REPORT] Epoch report failed: {e}")
            if rank == 0:
                print(mean_psnr)
                np.save(f'{exp_path}/val_l1_loss.npy', val_l1_loss)
                np.save(f'{exp_path}/val_psnr_values.npy', val_psnr_values)


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

    # Initialize process group with explicit timeout
    timeout = datetime.timedelta(minutes=nccl_timeout_minutes)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size, timeout=timeout)

    training_succeeded = False
    try:
        fn(rank, gpu, args)
        training_succeeded = True
        # Only barrier if training succeeded to avoid hanging on errors
        if rank == 0:
            print(f"[rank: {rank}] Training completed successfully, synchronizing ranks...", flush=True)
        dist.barrier()
    except Exception as e:
        print(f"[rank: {rank}] ERROR during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Don't barrier on error - let process exit immediately
        raise
    finally:
        if dist.is_initialized():
            try:
                if not training_succeeded:
                    # Give other ranks a moment to detect the error before destroying process group
                    time.sleep(2)
                dist.destroy_process_group()
                if rank == 0:
                    print(f"[rank: {rank}] Process group destroyed", flush=True)
            except Exception as e:
                print(f"[rank: {rank}] Error destroying process group: {e}", flush=True)

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
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='Directory with gen_diffusive_1.pth and gen_diffusive_2.pth to initialize generators (ignored if --resume).')

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

    # logging controls
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log every N iterations (per-rank=0). Set 0 to disable per-iter logs.')
    parser.add_argument('--log_mem_after_update', action='store_true', default=False,
                        help='Emit memory line after each logged update.')
    parser.add_argument('--debug_verbose', action='store_true', default=False,
                        help='Print verbose tensor shapes/stats for debugging.')

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
