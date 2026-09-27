#!/usr/bin/env python3
"""
CUDA / GPU detection and LightGBM GPU capability utilities.

Adapted from the reference setup_venv.py CUDA detector to target LightGBM GPU
training instead of PyTorch.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PYTORCH_CUDA_INDEXES = [
    # (min_driver_cuda, index_tag)
    (12, 4, "cu124"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
]


def log(msg: str) -> None:
    print(f"[cuda] {msg}", flush=True)


def detect_nvidia_gpu() -> dict[str, Any] | None:
    """Detect NVIDIA GPU hardware and CUDA driver version via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        log("No NVIDIA GPU detected (nvidia-smi not found)")
        return None

    try:
        r = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            log("nvidia-smi execution failed")
            return None

        lines = r.stdout.strip().split("\n")
        if not lines or not lines[0].strip():
            log("No GPU listed by nvidia-smi")
            return None

        parts = lines[0].split(",")
        gpu_name = parts[0].strip()

        # Parse CUDA Version from full nvidia-smi output
        r2 = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=10)
        cuda_ver = None
        for out_line in r2.stdout.split("\n"):
            if "CUDA Version" in out_line:
                m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out_line)
                if m:
                    cuda_ver = (int(m.group(1)), int(m.group(2)))
                break

        cuda_ver = cuda_ver or (12, 4)
        cuda_major, cuda_minor = cuda_ver

        chosen_tag = None
        for req_major, req_minor, tag in PYTORCH_CUDA_INDEXES:
            if (cuda_major, cuda_minor) >= (req_major, req_minor):
                chosen_tag = tag
                break

        if not chosen_tag:
            log(f"NVIDIA GPU detected ({gpu_name}), but CUDA {cuda_major}.{cuda_minor} is older than supported versions.")
            return None

        log(f"NVIDIA GPU detected: {gpu_name} (CUDA {cuda_major}.{cuda_minor} -> {chosen_tag})")
        return {"gpu": gpu_name, "cuda": cuda_ver, "tag": chosen_tag}

    except Exception as e:
        log(f"GPU detection failed ({e})")
        return None


def test_lightgbm_gpu(python_exe: str | None = None) -> bool:
    """Probe whether the installed LightGBM can actually train on GPU.

    If python_exe is provided, the probe runs in that interpreter (e.g. the venv
    python). Otherwise it probes the current interpreter.
    """
    code = (
        "import lightgbm as lgb, numpy as np; "
        "X=np.random.rand(100,4).astype(np.float32); y=np.random.randint(0,2,size=100); "
        "ds=lgb.Dataset(X,label=y); "
        "lgb.train({\"objective\":\"binary\",\"device_type\":\"gpu\",\"gpu_platform_id\":0,\"gpu_device_id\":0,\"verbosity\":-1,\"num_leaves\":7}, ds, num_boost_round=2); "
        "print('LIGHTGBM_GPU_OK')"
    )
    exe = python_exe or sys.executable
    try:
        r = subprocess.run(
            [exe, "-c", code],
            capture_output=True, text=True, timeout=60
        )
        ok = r.returncode == 0 and "LIGHTGBM_GPU_OK" in r.stdout
        if not ok:
            log(f"LightGBM GPU probe failed: {r.stderr.strip() or r.stdout.strip()}")
        return ok
    except Exception as e:
        log(f"LightGBM GPU probe failed: {e}")
        return False


def get_recommended_device(force_cpu: bool = False, force_gpu: bool = False) -> str:
    """Return 'gpu' or 'cpu' based on detection and CLI flags."""
    if force_cpu:
        return "cpu"
    if force_gpu:
        if detect_nvidia_gpu() or test_lightgbm_gpu():
            return "gpu"
        log("GPU forced but no working GPU LightGBM detected; falling back to CPU")
        return "cpu"
    # auto
    gpu_info = detect_nvidia_gpu()
    if gpu_info and test_lightgbm_gpu():
        return "gpu"
    return "cpu"


def save_status(project_root: str, status: dict[str, Any]) -> None:
    path = Path(project_root) / "cuda_status.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)


def load_status(project_root: str) -> dict[str, Any] | None:
    path = Path(project_root) / "cuda_status.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
