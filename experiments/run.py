import os
import sys
import subprocess
import argparse
import yaml

# Add these sets to encode boolean flag semantics
STORE_TRUE_FLAGS = {
    # train.py
    "use_geometric", "use_ema", "not_use_tanh", "no_lr_decay", "save_content",
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
        train_cmd = [sys.executable, "../engine/train.py"]
        train_cmd = append_args(train_cmd, train_args)
        print("Train command:", " ".join(train_cmd))
        result = subprocess.run(train_cmd, cwd=os.path.dirname(__file__))
        if result.returncode != 0:
            print(f"Training failed for {exp_name}")
            sys.exit(1)

    # Run testing if not train-only
    if not args.train_only:
        test_script_path = os.path.join(os.path.dirname(__file__), "../engine/test.py")
        if os.path.exists(test_script_path):
            test_cmd = [sys.executable, "../engine/test.py"]
            test_cmd = append_args(test_cmd, test_args)
            print("Test command:", " ".join(test_cmd))
            result = subprocess.run(test_cmd, cwd=os.path.dirname(__file__))
            if result.returncode != 0:
                print(f"Testing failed for {exp_name}")
                sys.exit(1)
        else:
            print("Test script not found, skipping testing phase.")

    print(f"\nExperiment '{exp_name}' completed successfully.")
