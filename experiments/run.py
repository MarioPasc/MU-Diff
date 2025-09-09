import os
import sys
import subprocess
import argparse
import yaml
import socket
import platform
import datetime as _dt
import json
from shutil import which
from pathlib import Path

# ==============================================
# MU-Diff Enhanced Runner
# Adds rich session logging + env exports
# ==============================================

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = (
    f"{REPO_ROOT}:{os.environ.get('PYTHONPATH','')}"
    if os.environ.get("PYTHONPATH") else str(REPO_ROOT)
)

# 1) Locate the pip-installed CUDA 12.1 nvcc dir (works for any Python x.y)
import inspect, os
import nvidia.cuda_nvcc as m
NVCC_DIR=os.path.dirname(inspect.getfile(m))


# 2) Use that toolchain
os.environ["PYTHONPATH"] = (
    f"{REPO_ROOT}:{os.environ.get('PYTHONPATH','')}"
    if os.environ.get("PYTHONPATH") else str(REPO_ROOT)
)
os.environ["CUDA_HOME"] = NVCC_DIR
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ['PATH']}"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"   # RTX 4090
subprocess.run(["which", "nvcc"])
subprocess.run(["nvcc", "--version"])



# Add these sets to encode boolean flag semantics
STORE_TRUE_FLAGS = {
    # train.py
    "use_geometric", "use_ema", "not_use_tanh", "no_lr_decay", "save_content",
    "use_bf16",
    # test.py
    "compute_fid",
}
STORE_FALSE_FLAGS = {
    # train.py flags that flip True->False when present
    "centered", "resamp_with_conv", "conditional", "fir", "skip_rescale",
}

# Keys that argparse expects as a single argument (string), even if YAML provides a list
LIST_AS_COMMA_FLAGS = {"attn_resolutions", "fir_kernel"}


def append_args(cmd, args_dict):
    for key, val in args_dict.items():
        if val is None:
            continue

        # Booleans: include flag only if it changes the default
        if isinstance(val, bool):
            if (key in STORE_TRUE_FLAGS and val) or (key in STORE_FALSE_FLAGS and not val):
                cmd.append(f"--{key}")
            continue

        # Lists or tuples
        if isinstance(val, (list, tuple)):
            if key in LIST_AS_COMMA_FLAGS:
                # Join into a single comma-separated token: e.g. "1,3,3,1"
                cmd.append(f"--{key}")
                cmd.append(",".join(str(x) for x in val))
            else:
                # True var-args list (e.g., ch_mult)
                cmd.append(f"--{key}")
                cmd += [str(x) for x in val]
            continue

        # Everything else (numbers/strings)
        cmd.append(f"--{key}")
        cmd.append(str(val))
    return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a specific experiment as defined in a YAML config.",
        epilog="""
Examples:
  python run.py -c cfg/local.yaml -e synthesize_T1CE
  python run.py -c cfg/local.yaml -e synthesize_FLAIR --train-only
  python run.py -c cfg/local.yaml -e synthesize_T2 --test-only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", "-c", type=str, required=True, 
                        help="Path to the YAML configuration file.")
    parser.add_argument("--experiment", "-e", type=str, required=True, 
                        help="Name of the specific experiment to run (must match exp_name in YAML).")
    parser.add_argument("--train-only", action='store_true', 
                        help="Run only training (skip testing).")
    parser.add_argument("--test-only", action='store_true', 
                        help="Run only testing (skip training).")
    args = parser.parse_args()

    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------
    def _git_info():
        info = {}
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            info["commit"] = commit
            info["branch"] = branch
            info["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip())
        except Exception:
            pass
        return info

    def _nvidia_smi():
        if which("nvidia-smi") is None:
            return None
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap","--format=csv,noheader"], stderr=subprocess.DEVNULL).decode().strip().splitlines()
            return out
        except Exception:
            return None

    def _torch_info():
        try:
            import torch
            obj = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
            if torch.cuda.is_available():
                obj["devices"] = [
                    {
                        "idx": i,
                        "name": torch.cuda.get_device_name(i),
                        "cc": torch.cuda.get_device_capability(i),
                    } for i in range(torch.cuda.device_count())
                ]
            return obj
        except Exception as e:
            return {"torch_info_error": str(e)}

    def export_env_from_dict(prefix, d):
        exports = {}
        for k, v in d.items():
            if v is None:
                continue
            # Convert list / tuple to comma joined string for single var
            if isinstance(v, (list, tuple)):
                v_str = ",".join(str(x) for x in v)
            else:
                v_str = str(v)
            env_key = f"{prefix}{k.upper()}"
            os.environ[env_key] = v_str
            exports[env_key] = v_str
        return exports

    def _print_kv_block(title, mapping):
        print(f"\n--- {title} ---")
        if not mapping:
            print("(none)")
            return
        pad = max(len(k) for k in mapping.keys()) if mapping else 0
        for k in sorted(mapping.keys()):
            print(f"{k.ljust(pad)} : {mapping[k]}")

    # --------------------------------------------------
    # Session metadata (before doing any heavy work)
    # --------------------------------------------------
    START_TIME = _dt.datetime.utcnow()
    print("\n=====================================")
    print(" MU-Diff Experiment Orchestrator")
    print("=====================================")
    print(f"Start (UTC)     : {START_TIME.isoformat()}Z")
    print(f"Host            : {socket.gethostname()}")
    print(f"User            : {os.getenv('USER')}")
    print(f"Working Dir     : {os.getcwd()}")
    print(f"Python Exec     : {sys.executable}")
    print(f"Python Version  : {platform.python_version()}")
    print(f"PythonPATH      : {os.environ.get('PYTHONPATH','(not set)')}")
    print(f"Platform        : {platform.platform()}")
    print(f"Interpreter Args: {' '.join(sys.argv)}")
    git = _git_info()
    if git:
        print(f"Git Branch      : {git.get('branch')}")
        print(f"Git Commit      : {git.get('commit')}")
        print(f"Git Dirty       : {git.get('dirty')}")
    # SLURM environment (if running under a job)
    slurm_keys = [k for k in os.environ.keys() if k.startswith("SLURM_")]
    if slurm_keys:
        subset = {k: os.environ[k] for k in sorted(slurm_keys)}
        _print_kv_block("SLURM Environment", subset)
    else:
        print("No SLURM environment detected.")
    print("=====================================\n")

    # Load YAML configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    experiments = config.get("experiments", [])
    if not experiments:
        sys.exit("No experiments found in the configuration file.")

    # Find the specified experiment
    target_exp = None
    for exp in experiments:
        exp_name = exp.get("exp_name") or exp.get("name") or exp.get("exp")
        if exp_name == args.experiment:
            target_exp = exp
            break
    
    if target_exp is None:
        available_experiments = [exp.get("exp_name") or exp.get("name") or exp.get("exp") for exp in experiments]
        sys.exit(f"Experiment '{args.experiment}' not found. Available experiments: {available_experiments}")

    print(f"\n=== Running Experiment: {args.experiment} ===")
    
    # Optional global settings from YAML
    data_path = config.get("data_path", None)        # base dataset path if provided
    output_root = config.get("output_root", None)    # base output directory for results if provided

    exp_name = target_exp.get("exp_name") or target_exp.get("name") or target_exp.get("exp")
    train_args = target_exp.get("train_args", {})
    test_args = target_exp.get("test_args", {})

    # Insert global defaults if not set in experiment
    if data_path:
        train_args.setdefault("input_path", data_path)
        test_args.setdefault("input_path", data_path)
    if output_root:
        out_dir = os.path.join(output_root, exp_name)
        os.makedirs(out_dir, exist_ok=True)
        train_args.setdefault("output_path", out_dir)
        test_args.setdefault("output_path", out_dir)
    
    # Ensure experiment name and target modality are passed
    train_args.setdefault("exp", exp_name)
    test_args.setdefault("exp", exp_name)
    target_mod = target_exp.get("target") or target_exp.get("target_modality")
    if target_mod:
        train_args.setdefault("target_modality", target_mod)
        test_args.setdefault("target_modality", target_mod)

    # Run training if not test-only
    if not args.test_only:
        print("\nPreparing training phase...")
        train_cmd = [sys.executable, "../engine/train.py"]
        train_cmd = append_args(train_cmd, train_args)
        # Export env vars for train args
        train_exports = export_env_from_dict("MUDIFF_TRAIN_", train_args)
        os.environ["MUDIFF_PHASE"] = "train"
        os.environ["MUDIFF_EXP_NAME"] = exp_name
        _print_kv_block("Train Arguments", {k: train_args[k] for k in train_args})
        _print_kv_block("Train Env Exports", train_exports)
        print("Train command:", " ".join(train_cmd))
        result = subprocess.run(train_cmd, cwd=os.path.dirname(__file__))
        if result.returncode != 0:
            print(f"Training failed for {exp_name}")
            sys.exit(1)

    # Run testing if not train-only
    if not args.train_only:
        test_script_path = os.path.join(os.path.dirname(__file__), "../engine/test.py")
        if os.path.exists(test_script_path):
            print("\nPreparing testing phase...")
            test_cmd = [sys.executable, "../engine/test.py"]
            test_cmd = append_args(test_cmd, test_args)
            test_exports = export_env_from_dict("MUDIFF_TEST_", test_args)
            os.environ["MUDIFF_PHASE"] = "test"
            _print_kv_block("Test Arguments", {k: test_args[k] for k in test_args})
            _print_kv_block("Test Env Exports", test_exports)
            print("Test command:", " ".join(test_cmd))
            result = subprocess.run(test_cmd, cwd=os.path.dirname(__file__))
            if result.returncode != 0:
                print(f"Testing failed for {exp_name}")
                sys.exit(1)
        else:
            print("Test script not found, skipping testing phase.")

    END_TIME = _dt.datetime.utcnow()
    duration = (END_TIME - START_TIME).total_seconds()
    print(f"\nExperiment '{exp_name}' completed successfully.")
    print(f"End (UTC)      : {END_TIME.isoformat()}Z")
    print(f"Total Duration : {duration:.2f} seconds (~{duration/3600:.2f} h)")

    # Summarize resources (best-effort)
    torch_meta = _torch_info()
    if torch_meta:
        print("\n--- Torch / CUDA Summary ---")
        print(json.dumps(torch_meta, indent=2))
    smi = _nvidia_smi()
    if smi:
        print("\n--- nvidia-smi (compact) ---")
        for line in smi:
            print(line)

    print("\nAll done.")
