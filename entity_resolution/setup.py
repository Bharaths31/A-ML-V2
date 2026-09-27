#!/usr/bin/env python3
"""
Windows (and cross-platform) setup automation for the V2 Business Entity Resolution project.

Run from the `entity_resolution/` folder:
    python setup.py

What it does:
    1. Checks Python version (3.11+ recommended).
    2. Detects NVIDIA GPU / CUDA and saves `cuda_status.json`.
    3. Creates a virtual environment at `.venv`.
    4. Upgrades pip and installs `requirements.txt`.
    5. If a GPU is detected, attempts to ensure GPU-enabled LightGBM (conda first, then prints manual fallback).
    6. Creates the unified `data/` folder.
    7. Locates `student_resource/` and copies train/test/utils/Documentation_template.md into `data/`.
    8. Prints the activation command for the developer.

Flags:
    --cpu            Force CPU-only mode.
    --gpu            Force GPU mode (will fall back to CPU if GPU unavailable).
    --cuda-tag TAG   Override CUDA tag for logging (e.g. cu124).
    --skip-venv      Skip venv creation and only copy data.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from cuda_utils import detect_nvidia_gpu, test_lightgbm_gpu, save_status, load_status

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
DATA_DIR = PROJECT_ROOT / "data"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

DEFAULT_STUDENT_RESOURCE_SEARCH = [
    PROJECT_ROOT / "student_resource",
    PROJECT_ROOT.parent / "student_resource",
    PROJECT_ROOT.parent.parent / "student_resource",
]


def log(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def find_student_resource(user_path: str | None) -> Path | None:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Provided student_resource path not found: {p}")

    for p in DEFAULT_STUDENT_RESOURCE_SEARCH:
        if p.exists() and p.is_dir():
            return p.resolve()
    return None


def copy_student_resource(src: Path, dst: Path) -> None:
    """Copy only the needed parts of student_resource into data/.

    The dataset/ subfolder is flattened: its train/ and test/ contents go
    directly into data/train/ and data/test/.
    """
    needed_files = ["utils", "Documentation_template.md", "README.md"]
    for name in needed_files:
        src_item = src / name
        if not src_item.exists():
            log(f"WARNING: {src_item} not found, skipping")
            continue
        dst_item = dst / name
        if dst_item.exists():
            shutil.rmtree(dst_item, ignore_errors=True) if dst_item.is_dir() else dst_item.unlink()
        if src_item.is_dir():
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)
        log(f"copied {name}")

    # Flatten dataset/ into data/
    dataset_src = src / "dataset"
    if dataset_src.exists() and dataset_src.is_dir():
        for sub in ["train", "test"]:
            src_sub = dataset_src / sub
            dst_sub = dst / sub
            if not src_sub.exists():
                log(f"WARNING: {src_sub} not found, skipping")
                continue
            if dst_sub.exists():
                shutil.rmtree(dst_sub, ignore_errors=True)
            shutil.copytree(src_sub, dst_sub)
            log(f"copied dataset/{sub}")
    else:
        log("WARNING: dataset/ folder not found in student_resource")


def create_venv() -> Path:
    python_exe = sys.executable
    log(f"using interpreter: {python_exe}")
    if VENV_DIR.exists():
        log("removing existing .venv")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    log("creating virtual environment at .venv")
    subprocess.check_call([python_exe, "-m", "venv", str(VENV_DIR)])
    return VENV_DIR


def get_venv_python(venv: Path) -> Path:
    if platform.system().lower() == "windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def install_requirements(venv_python: Path) -> None:
    log("upgrading pip")
    subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    if REQUIREMENTS.exists():
        log(f"installing {REQUIREMENTS}")
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    else:
        log("WARNING: requirements.txt not found")


def ensure_gpu_lightgbm(venv_python: Path, gpu_info: dict | None, force_gpu: bool) -> bool:
    """Attempt to install/enable GPU LightGBM. Return True if GPU training works."""
    if not gpu_info and not force_gpu:
        return False

    log("GPU detected; probing / enabling GPU LightGBM")
    # First test if the pip-installed LightGBM already has GPU support.
    if test_lightgbm_gpu(str(venv_python)):
        log("LightGBM GPU already works")
        return True

    # Try conda-forge if conda is available (most reliable Windows GPU path).
    conda = shutil.which("conda")
    if conda:
        log("Attempting conda-forge LightGBM GPU install...")
        try:
            subprocess.check_call(
                [conda, "install", "-y", "-p", str(VENV_DIR), "-c", "conda-forge", "lightgbm"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if test_lightgbm_gpu():
                log("LightGBM GPU enabled via conda-forge")
                return True
        except Exception as e:
            log(f"conda-forge install failed: {e}")

    # Fallback: try source build with CUDA. This often fails on Windows without MSVC/CMake,
    # so we catch and report rather than crash.
    log("Attempting pip source build of LightGBM with CUDA (this may take minutes)...")
    try:
        subprocess.check_call(
            [str(venv_python), "-m", "pip", "install", "--force-reinstall",
             "--no-binary", ":all:", "lightgbm"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if test_lightgbm_gpu():
            log("LightGBM GPU enabled via source build")
            return True
    except Exception as e:
        log(f"Source build failed: {e}")

    log("WARNING: GPU detected but GPU-enabled LightGBM could not be installed automatically.")
    log("  The pipeline will fall back to CPU. To enable GPU manually, run one of:")
    log("    conda install -c conda-forge lightgbm")
    log("    or build LightGBM from source with USE_CUDA=ON")
    return False


def print_activation(venv: Path) -> None:
    if platform.system().lower() == "windows":
        activate = venv / "Scripts" / "activate.bat"
        pwsh = venv / "Scripts" / "Activate.ps1"
        log("\n=== Activate your environment ===")
        log(f"  CMD:     {activate}")
        log(f"  PowerShell:  {pwsh}")
    else:
        activate = venv / "bin" / "activate"
        log(f"\n=== Activate your environment ===")
        log(f"  source {activate}")
    log("\nThen run:")
    log("  python master_profiler.py --stage all")
    log("  python master_trainer.py --stage all")


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup V2 Business Entity Resolution environment")
    parser.add_argument("--student-resource", "-s", default=None,
                        help="Path to student_resource folder (auto-detected if omitted)")
    parser.add_argument("--skip-venv", action="store_true",
                        help="Skip venv creation and only copy data")
    parser.add_argument("--cpu", "--force-cpu", dest="force_cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--gpu", "--force-gpu", dest="force_gpu", action="store_true",
                        help="Force GPU mode")
    parser.add_argument("--cuda-tag", type=str, default=None,
                        help="Override CUDA tag (e.g. cu124)")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        log("WARNING: Python 3.11+ is recommended")

    DATA_DIR.mkdir(exist_ok=True)

    # 1. GPU detection (before venv so the user sees it early)
    gpu_info = None
    if args.force_cpu:
        log("Forcing CPU mode as requested")
    else:
        if args.cuda_tag:
            log(f"Using explicit CUDA tag: {args.cuda_tag}")
        gpu_info = detect_nvidia_gpu()
        if gpu_info and args.cuda_tag:
            gpu_info["tag"] = args.cuda_tag

    src_res = find_student_resource(args.student_resource)
    if src_res:
        log(f"found student_resource at {src_res}")
        copy_student_resource(src_res, DATA_DIR)
    else:
        log("WARNING: student_resource not found. Provide it with --student-resource PATH")
        log("searched:")
        for p in DEFAULT_STUDENT_RESOURCE_SEARCH:
            log(f"  {p}")

    if not args.skip_venv:
        venv = create_venv()
        venv_python = get_venv_python(venv)
        install_requirements(venv_python)

        # Attempt GPU LightGBM enablement
        gpu_works = False
        if gpu_info or args.force_gpu:
            gpu_works = ensure_gpu_lightgbm(venv_python, gpu_info, args.force_gpu)

        # Re-probe in case base install already had GPU
        if not gpu_works and not args.force_cpu:
            gpu_works = test_lightgbm_gpu(str(venv_python))

        status = {
            "gpu_detected": gpu_info is not None,
            "gpu_info": gpu_info,
            "force_cpu": args.force_cpu,
            "force_gpu": args.force_gpu,
            "lightgbm_gpu_works": gpu_works,
            "platform": platform.system(),
        }
    else:
        venv = VENV_DIR
        status = {
            "gpu_detected": gpu_info is not None,
            "gpu_info": gpu_info,
            "force_cpu": args.force_cpu,
            "force_gpu": args.force_gpu,
            "lightgbm_gpu_works": False,
            "skipped_venv": True,
        }

    save_status(str(PROJECT_ROOT), status)
    log(f"CUDA/GPU status saved to cuda_status.json: {status}")

    print_activation(venv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
