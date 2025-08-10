#!/usr/bin/env python3
"""
MU-Diff Pipeline Consistency Checker

This script validates the entire MU-Diff pipeline for SLURM deployment,
checking for inconsistencies that could prevent successful training and testing.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - NOT FOUND")
        return False

def check_directory_structure():
    """Check the overall directory structure."""
    print("\n=== DIRECTORY STRUCTURE ===")
    
    required_dirs = [
        "experiments",
        "experiments/cfg", 
        "engine",
        "slurm_scripts",
        "dataset",
        "backbones",
        "utils",
        "data",  # Should contain preprocessed data
        "results"  # Will be created by experiments
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Directory: {dir_path} - NOT FOUND")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0

def check_configuration_files():
    """Check configuration files."""
    print("\n=== CONFIGURATION FILES ===")
    
    config_files = [
        "experiments/cfg/local.yaml",
        "experiments/run.py"
    ]
    
    all_exist = True
    for config_file in config_files:
        if not check_file_exists(config_file, "Config file"):
            all_exist = False
    
    return all_exist

def check_slurm_scripts():
    """Check SLURM scripts."""
    print("\n=== SLURM SCRIPTS ===")
    
    slurm_scripts = [
        "slurm_scripts/mudiff_t1ce.sh",
        "slurm_scripts/mudiff_flair.sh", 
        "slurm_scripts/mudiff_t2.sh",
        "slurm_scripts/mudiff_t1.sh",
        "submit_all_jobs.sh"
    ]
    
    all_exist = True
    for script in slurm_scripts:
        if check_file_exists(script, "SLURM script"):
            # Check if executable
            if os.access(script, os.X_OK):
                print(f"  ✓ Executable: {script}")
            else:
                print(f"  ✗ Not executable: {script}")
                all_exist = False
        else:
            all_exist = False
    
    return all_exist

def check_core_scripts():
    """Check core training and testing scripts."""
    print("\n=== CORE SCRIPTS ===")
    
    core_scripts = [
        "engine/train.py",
        "engine/test.py",
        "engine/test_volume.py"
    ]
    
    all_exist = True
    for script in core_scripts:
        if not check_file_exists(script, "Core script"):
            all_exist = False
    
    return all_exist

def check_dataset_integration():
    """Check dataset integration."""
    print("\n=== DATASET INTEGRATION ===")
    
    dataset_files = [
        "dataset/dataset_brats.py"
    ]
    
    all_exist = True
    for dataset_file in dataset_files:
        if not check_file_exists(dataset_file, "Dataset file"):
            all_exist = False
    
    # Check for BratsDataset class
    try:
        if os.path.exists("dataset/dataset_brats.py"):
            with open("dataset/dataset_brats.py", 'r') as f:
                content = f.read()
                if "class BratsDataset" in content:
                    print("✓ BratsDataset class found")
                else:
                    print("✗ BratsDataset class not found")
                    all_exist = False
        
        # Check train.py integration
        if os.path.exists("engine/train.py"):
            with open("engine/train.py", 'r') as f:
                content = f.read()
                if "from dataset.dataset_brats import BratsDataset" in content:
                    print("✓ BratsDataset properly imported in train.py")
                elif "BratsDatasetWrapper" in content:
                    print("✓ BratsDatasetWrapper found in train.py")
                else:
                    print("✗ Dataset integration missing in train.py")
                    all_exist = False
                    
    except Exception as e:
        print(f"✗ Error checking dataset integration: {e}")
        all_exist = False
    
    return all_exist

def check_yaml_experiments():
    """Check YAML experiment configurations."""
    print("\n=== YAML EXPERIMENT CONFIGURATION ===")
    
    config_path = "experiments/cfg/local.yaml"
    if not os.path.exists(config_path):
        print(f"✗ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required experiments
        required_experiments = ["synthesize_T1CE", "synthesize_FLAIR", "synthesize_T2", "synthesize_T1"]
        experiments = config.get("experiments", [])
        
        if not experiments:
            print("✗ No experiments found in YAML config")
            return False
        
        found_experiments = []
        for exp in experiments:
            exp_name = exp.get("exp_name") or exp.get("name") or exp.get("exp")
            if exp_name:
                found_experiments.append(exp_name)
        
        all_found = True
        for req_exp in required_experiments:
            if req_exp in found_experiments:
                print(f"✓ Experiment found: {req_exp}")
            else:
                print(f"✗ Experiment missing: {req_exp}")
                all_found = False
        
        # Check distributed training configuration
        print("\nChecking distributed training configuration...")
        for exp in experiments:
            exp_name = exp.get("exp_name", "unknown")
            train_args = exp.get("train_args", {})
            
            # Check GPU requirements
            num_process_per_node = train_args.get("num_process_per_node", 1)
            if num_process_per_node == 2:
                print(f"✓ {exp_name}: Correct GPU configuration (2 processes)")
            else:
                print(f"✗ {exp_name}: Incorrect GPU configuration ({num_process_per_node} processes)")
                all_found = False
            
            # Check unique ports
            port_num = train_args.get("port_num", "6021")
            print(f"  Port for {exp_name}: {port_num}")
        
        return all_found
        
    except Exception as e:
        print(f"✗ Error parsing YAML config: {e}")
        return False

def check_data_paths():
    """Check data paths and structure."""
    print("\n=== DATA PATHS ===")
    
    # Check if data directory exists (should be uploaded to supercomputer)
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"✓ Data directory exists: {data_dir}")
        
        # Check for expected subdirectories
        expected_subdirs = ["brats", "isles"]  # Based on the data structure
        for subdir in expected_subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"✓ Subdirectory exists: {subdir_path}")
                
                # Check for split files
                split_files = ["train.list", "val.list", "test.list"]
                for split_file in split_files:
                    split_path = os.path.join(subdir_path, split_file)
                    if os.path.exists(split_path):
                        print(f"  ✓ Split file: {split_path}")
                    else:
                        print(f"  ✗ Split file missing: {split_path}")
            else:
                print(f"✗ Subdirectory missing: {subdir_path}")
    else:
        print(f"⚠ Data directory not found: {data_dir}")
        print("  This is expected if data hasn't been uploaded to supercomputer yet.")
        print("  Make sure to upload preprocessed data before running SLURM jobs.")
    
    return True  # Don't fail on missing data for now

def check_dependencies():
    """Check Python dependencies."""
    print("\n=== PYTHON DEPENDENCIES ===")
    
    required_modules = [
        "torch",
        "torchvision", 
        "numpy",
        "PIL",
        "skimage",
        "yaml",
        "nibabel"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ Module available: {module}")
        except ImportError:
            print(f"✗ Module missing: {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing modules: {missing_modules}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Check MU-Diff pipeline consistency")
    parser.add_argument("--skip-deps", action="store_true", 
                        help="Skip dependency checking (useful for supercomputer validation)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MU-DIFF PIPELINE CONSISTENCY CHECKER")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_configuration_files),
        ("SLURM Scripts", check_slurm_scripts),
        ("Core Scripts", check_core_scripts),
        ("Dataset Integration", check_dataset_integration),
        ("YAML Experiments", check_yaml_experiments),
        ("Data Paths", check_data_paths)
    ]
    
    if not args.skip_deps:
        checks.append(("Python Dependencies", check_dependencies))
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"✗ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{check_name:<25}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Pipeline ready for SLURM deployment!")
        print("\nNext steps:")
        print("1. Upload code and preprocessed data to supercomputer")
        print("2. Set up Python environment on supercomputer") 
        print("3. Run: ./submit_all_jobs.sh")
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues before deployment")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
