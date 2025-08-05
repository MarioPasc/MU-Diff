#!/usr/bin/env python3
"""
Volume Prediction Wrapper for MU-Diff

This script provides a convenient interface for predicting complete 3D volumes
using trained MU-Diff models. It handles preprocessing internally and saves
results as NIfTI files.
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path

def find_input_file(input_dir, patterns):
    """Find input file with multiple possible naming conventions."""
    for pattern in patterns:
        for ext in ['.nii.gz', '.nii']:
            file_path = os.path.join(input_dir, f"{pattern}{ext}")
            if os.path.isfile(file_path):
                return file_path
    return None

def get_experiment_config(config_file, experiment_name):
    """Load experiment configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    experiments = config.get("experiments", [])
    for exp in experiments:
        exp_name = exp.get("exp_name") or exp.get("name") or exp.get("exp")
        if exp_name == experiment_name:
            return exp, config
    
    return None, config

def main():
    parser = argparse.ArgumentParser(
        description="Predict 3D volumes using trained MU-Diff models",
        epilog="""
Examples:
  # Using experiment configuration
  python predict_volume_wrapper.py -c cfg/local.yaml -e synthesize_T1CE \\
                                   -i /path/to/input/volumes -o /path/to/output

  # Direct specification
  python predict_volume_wrapper.py --target T1CE -i /path/to/input/volumes \\
                                   -o /path/to/output --exp synthesize_T1CE
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration-based approach
    parser.add_argument("--config", "-c", type=str,
                        help="Path to YAML configuration file")
    parser.add_argument("--experiment", "-e", type=str,
                        help="Name of experiment in config file")
    
    # Direct specification approach
    parser.add_argument("--target", type=str, choices=['T1CE', 'FLAIR', 'T2', 'T1'],
                        help="Target modality to synthesize")
    parser.add_argument("--exp", type=str,
                        help="Experiment name (for model loading)")
    
    # Input/Output
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Directory containing input NIfTI files")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory to save predicted volume")
    
    # Optional overrides
    parser.add_argument("--results_path", type=str, default="./results",
                        help="Base path where experiment results are stored")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device to use")
    parser.add_argument("--slice_range", type=int, default=80,
                        help="Number of slices on each side of volume center")
    
    args = parser.parse_args()
    
    # Determine target modality and experiment name
    target_modality = None
    exp_name = None
    
    if args.config and args.experiment:
        # Load from configuration
        print(f"Loading experiment '{args.experiment}' from config: {args.config}")
        exp_config, global_config = get_experiment_config(args.config, args.experiment)
        
        if exp_config is None:
            print(f"Error: Experiment '{args.experiment}' not found in config file")
            return 1
        
        target_modality = exp_config.get("target") or exp_config.get("target_modality")
        exp_name = exp_config.get("exp_name") or exp_config.get("name") or exp_config.get("exp")
        
        if not target_modality:
            print("Error: No target modality specified in experiment config")
            return 1
            
    elif args.target and args.exp:
        # Direct specification
        target_modality = args.target
        exp_name = args.exp
        
    else:
        print("Error: Must specify either (--config and --experiment) or (--target and --exp)")
        return 1
    
    print(f"Target modality: {target_modality}")
    print(f"Experiment name: {exp_name}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Check input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Find input files with various naming conventions
    print("Searching for input files...")
    input_patterns = {
        'T1CE': ['t1ce', 't1c', 'T1CE', 'T1C'],
        'T1': ['t1', 't1n', 'T1', 'T1N'],
        'T2': ['t2', 't2w', 'T2', 'T2W'],
        'FLAIR': ['flair', 't2f', 'FLAIR', 'T2F']
    }
    
    input_files = {}
    for modality, patterns in input_patterns.items():
        file_path = find_input_file(args.input_dir, patterns)
        input_files[modality] = file_path
        status = file_path if file_path else "NOT FOUND"
        print(f"  {modality}: {status}")
    
    print()
    
    # Determine required input modalities based on target
    modality_orders = {
        "T1CE": ["FLAIR", "T2", "T1"],
        "FLAIR": ["T1CE", "T1", "T2"],
        "T2": ["T1CE", "T1", "FLAIR"],
        "T1": ["FLAIR", "T1CE", "T2"]
    }
    
    required_modalities = modality_orders[target_modality]
    print(f"Required input modalities for {target_modality}: {required_modalities}")
    
    # Check that all required files exist
    missing_files = []
    for modality in required_modalities:
        if not input_files[modality]:
            missing_files.append(modality)
    
    if missing_files:
        print(f"Error: Missing required input files: {missing_files}")
        return 1
    
    print("All required input files found!")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build command for test_volume.py
    script_dir = Path(__file__).parent
    engine_dir = script_dir.parent / "engine"
    test_volume_script = engine_dir / "test_volume.py"
    
    cmd = [
        sys.executable, str(test_volume_script),
        "--target_modality", target_modality,
        "--output_dir", args.output_dir,
        "--exp", exp_name,
        "--output_path", args.results_path,
        "--gpu_chose", str(args.gpu),
        "--slice_half_range", str(args.slice_range)
    ]
    
    # Add input file arguments
    if input_files['T1CE']:
        cmd.extend(["--input_t1ce", input_files['T1CE']])
    if input_files['T1']:
        cmd.extend(["--input_t1", input_files['T1']])
    if input_files['T2']:
        cmd.extend(["--input_t2", input_files['T2']])
    if input_files['FLAIR']:
        cmd.extend(["--input_flair", input_files['FLAIR']])
    
    # If using config, load model parameters
    if args.config and args.experiment:
        exp_config, global_config = get_experiment_config(args.config, args.experiment)
        test_args = exp_config.get("test_args", {})
        
        # Add model architecture parameters
        model_params = [
            'num_channels', 'image_size', 'num_channels_dae', 'n_mlp', 'ch_mult',
            'num_res_blocks', 'attn_resolutions', 'dropout', 'resamp_with_conv',
            'conditional', 'fir', 'fir_kernel', 'skip_rescale', 'resblock_type',
            'progressive', 'progressive_input', 'progressive_combine',
            'use_geometric', 'beta_min', 'beta_max', 'num_timesteps',
            'embedding_type', 'fourier_scale', 'not_use_tanh',
            'z_emb_dim', 't_emb_dim', 'nz', 'seed'
        ]
        
        for param in model_params:
            if param in test_args:
                value = test_args[param]
                if isinstance(value, list):
                    cmd.extend([f"--{param}"] + [str(v) for v in value])
                elif isinstance(value, bool):
                    if value:
                        cmd.append(f"--{param}")
                else:
                    cmd.extend([f"--{param}", str(value)])
    
    # Run the prediction
    print("Running volume prediction...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, cwd=str(engine_dir))
        print()
        print("Prediction completed successfully!")
        output_file = os.path.join(args.output_dir, f"predicted_{target_modality.lower()}.nii.gz")
        print(f"Output saved to: {output_file}")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Prediction failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
