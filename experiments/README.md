# MU-Diff Experiments

This directory contains the experiment management system for MU-Diff.

## Usage

### Running a single experiment

```bash
# Run a specific experiment (both training and testing)
python run.py -c cfg/local.yaml -e synthesize_T1CE

# Run only training
python run.py -c cfg/local.yaml -e synthesize_T1CE --train-only

# Run only testing
python run.py -c cfg/local.yaml -e synthesize_T1CE --test-only
```

### Available experiments in local.yaml

- `synthesize_T1CE` - Synthesize T1CE from [FLAIR, T2, T1]
- `synthesize_FLAIR` - Synthesize FLAIR from [T1CE, T1, T2]
- `synthesize_T2` - Synthesize T2 from [T1CE, T1, FLAIR]
- `synthesize_T1` - Synthesize T1 from [FLAIR, T1CE, T2]

### Configuration

Update `cfg/local.yaml` to:
1. Set correct `data_path` pointing to your BraTS dataset
2. Set `output_root` where results should be saved
3. Modify hyperparameters as needed for each experiment

### Key configuration parameters

The YAML file includes all tunable hyperparameters from train.py:

**Model Architecture:**
- `image_size`: Input image size (default: 256)
- `num_channels`: Number of input channels (default: 1)
- `ch_mult`: Channel multipliers for U-Net [1, 2, 4]
- `num_res_blocks`: Residual blocks per scale (default: 2)
- `num_channels_dae`: Initial channels in denoising model (default: 128)

**Training:**
- `batch_size`: Training batch size (default: 3)
- `num_epoch`: Number of training epochs (default: 30)
- `lr_g`: Generator learning rate (default: 1.6e-4)
- `lr_d`: Discriminator learning rate (default: 1.0e-4)

**Diffusion:**
- `num_timesteps`: Number of diffusion timesteps (default: 4)
- `beta_min`/`beta_max`: Diffusion noise schedule bounds

**Loss weights:**
- `lambda_l1_loss`: L1 reconstruction loss weight (default: 0.5)
- `lambda_mask_loss`: Mask loss weight (default: 0.1)
- `lambda_adv`: Adversarial loss weight (default: 1.0)

### Output structure

Results are saved to: `{output_root}/{experiment_name}/`
- Model checkpoints
- Training logs
- Generated samples

### Testing output structure

After testing, the following outputs are generated:

```
{output_root}/{experiment_name}/generated_samples/
├── test_samples_00000.jpg  # Original format (concatenated samples)
├── test_samples_00001.jpg
├── ...
├── pred/                   # Individual PNG predictions
│   ├── pred_00000.png
│   ├── pred_00001.png
│   └── ...
└── gt/                     # Individual PNG ground truth
    ├── gt_00000.png
    ├── gt_00001.png
    └── ...
```

**Key testing features:**
- **Global intensity scaling**: All images scaled using the same global min/max for consistency
- **Individual PNG files**: Each slice saved as a separate PNG for easy analysis
- **Preserved original format**: Original concatenated JPG samples still saved
- **Progress reporting**: Real-time progress updates during testing
- **Modality-specific**: Automatically uses correct modality ordering based on target

### Dataset structure

The system expects your BraTS dataset to be organized as:
```
{data_path}/
├── train/
│   ├── T1.npy
│   ├── T2.npy
│   ├── FLAIR.npy
│   └── T1CE.npy
├── val/
│   ├── T1.npy
│   ├── T2.npy
│   ├── FLAIR.npy
│   └── T1CE.npy
└── test/
    ├── T1.npy
    ├── T2.npy
    ├── FLAIR.npy
    └── T1CE.npy
```

Each `.npy` file should contain all slices for that modality with shape `(N, H, W)` where N is the number of slices.
