#!/usr/bin/env python3
"""
Pre-build CUDA extensions for MU-Diff to avoid JIT compilation issues
in multi-process distributed training environments.

This script should be run ONCE before starting training to compile
all CUDA extensions (fused_bias_act and upfirdn2d) ahead of time.

Usage:
    python build_extensions.py
"""

import os
import sys
import torch
from pathlib import Path

def build_extension(name, sources, build_dir=None):
    """Build a single CUDA extension"""
    print(f"\n{'='*60}")
    print(f"Building {name} extension...")
    print(f"{'='*60}")

    try:
        from torch.utils.cpp_extension import load

        ext = load(
            name=name,
            sources=sources,
            verbose=True,
            build_directory=build_dir,
        )
        print(f"[SUCCESS] {name} extension built successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to build {name} extension: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Build all CUDA extensions for MU-Diff"""
    print("="*60)
    print("MU-Diff CUDA Extensions Builder")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    print()

    # Set up paths
    repo_root = Path(__file__).parent.resolve()
    utils_op_dir = repo_root / "utils" / "op"

    # Set cache directory
    torch_ext_dir = os.environ.get('TORCH_EXTENSIONS_DIR')
    if torch_ext_dir:
        print(f"Using TORCH_EXTENSIONS_DIR: {torch_ext_dir}")
    else:
        print("TORCH_EXTENSIONS_DIR not set, using default")

    success_count = 0
    total_count = 2

    # Build fused_bias_act extension
    fused_sources = [
        str(utils_op_dir / "fused_bias_act.cpp"),
        str(utils_op_dir / "fused_bias_act_kernel.cu"),
    ]

    if build_extension("fused", fused_sources, torch_ext_dir):
        success_count += 1

    # Build upfirdn2d extension
    upfirdn2d_sources = [
        str(utils_op_dir / "upfirdn2d.cpp"),
        str(utils_op_dir / "upfirdn2d_kernel.cu"),
    ]

    if build_extension("upfirdn2d", upfirdn2d_sources, torch_ext_dir):
        success_count += 1

    # Summary
    print("\n" + "="*60)
    print("Build Summary")
    print("="*60)
    print(f"Successfully built: {success_count}/{total_count} extensions")

    if success_count == total_count:
        print("\n✓ All extensions built successfully!")
        print("You can now run training without JIT compilation issues.")
        return 0
    else:
        print(f"\n✗ {total_count - success_count} extension(s) failed to build.")
        print("Please check the error messages above and fix any issues.")
        print("\nCommon issues:")
        print("  1. Missing libxcrypt: conda install -c conda-forge libxcrypt")
        print("  2. CUDA compiler not found: Check CUDA_HOME and nvcc")
        print("  3. Incompatible compiler: Check C++ compiler version")
        return 1

if __name__ == "__main__":
    sys.exit(main())
