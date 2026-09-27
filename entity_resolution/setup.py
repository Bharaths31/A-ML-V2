#!/usr/bin/env python3
"""
Windows (and cross-platform) setup automation for the V2 Business Entity Resolution project.

Run from the `entity_resolution/` folder:
    python setup.py

What it does:
    1. Checks Python version (3.11+ recommended).
    2. Creates a virtual environment at `.venv`.
    3. Upgrades pip and installs `requirements.txt`.
    4. Creates the unified `data/` folder.
    5. Locates `student_resource/` and copies train/test/utils/Documentation_template.md into `data/`.
    6. Prints the activation command for the developer.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

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
    print(f"[setup] {msg}")


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
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        log("WARNING: Python 3.11+ is recommended")

    DATA_DIR.mkdir(exist_ok=True)

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
    else:
        venv = VENV_DIR

    print_activation(venv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
