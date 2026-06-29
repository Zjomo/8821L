"""
Installation script for the packages needed to run the (standard) SmartTrap
"""

#!/usr/bin/env python3
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys

BASE_REQ = "requirements-base.txt"

PT_INDEXES = {
    "cpu":   "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu129": "https://download.pytorch.org/whl/cu129",
    "rocm63": "https://download.pytorch.org/whl/rocm6.3",
}

def run(cmd, check=True, capture=False):
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["text"] = True
    print(f">> {cmd}")
    proc = subprocess.run(cmd, shell=True, **kwargs)
    if check and proc.returncode != 0:
        print(proc.stdout if capture else "", file=sys.stderr)
        sys.exit(proc.returncode)
    return proc.stdout if capture else ""

def parse_cuda_from_nvidia_smi(out: str) -> str | None:
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}"

def parse_cuda_from_nvcc(out: str) -> str | None:
    m = re.search(r"release\s+([0-9]+)\.([0-9]+)", out) or re.search(r"V([0-9]+)\.([0-9]+)", out)
    return f"{m.group(1)}.{m.group(2)}" if m else None

def detect_cuda_version() -> str | None:
    if shutil.which("nvidia-smi"):
        out = run("nvidia-smi", capture=True, check=False)
        ver = parse_cuda_from_nvidia_smi(out or "")
        if ver:
            return ver
    if shutil.which("nvcc"):
        out = run("nvcc --version", capture=True, check=False)
        ver = parse_cuda_from_nvcc(out or "")
        if ver:
            return ver
    return None

def detect_rocm() -> bool:
    if platform.system() != "Linux":
        return False
    if shutil.which("rocminfo"):
        return subprocess.run("rocminfo >/dev/null 2>&1", shell=True).returncode == 0
    return False

def pick_torch_channel(cuda_ver: str | None, force: str | None, prefer_cpu: bool) -> str:
    """
    Returns a key from PT_INDEXES (cpu | cu118 | cu126 | cu128 | cu129 | rocm63)
    """
    if force:
        f = force.lower()
        if f == "cpu": return "cpu"
        if f in ("cuda11_8", "cu118"): return "cu118"
        if f in ("cuda12_6", "cu126"): return "cu126"
        if f in ("cuda12_8", "cu128"): return "cu128"
        if f in ("cuda12_9", "cu129"): return "cu129"
        if f in ("rocm6_3", "rocm63", "rocm"): return "rocm63"
        print(f"Unknown --force option: {force}", file=sys.stderr); sys.exit(2)

    if detect_rocm():
        return "rocm63"
    if prefer_cpu or not cuda_ver:
        return "cpu"

    try:
        major, minor = map(int, cuda_ver.split("."))
    except Exception:
        return "cpu"

    if major == 11 and minor >= 8:
        return "cu118"
    if major == 12:
        if minor >= 9:
            return "cu129"
        if minor >= 8:
            return "cu128"
        if minor >= 6:
            return "cu126"
        if minor >=5:
            return "cpu"# safest for older 12.x
        return "cu126"  

    return "cpu"

def main():
    ap = argparse.ArgumentParser(
        description="Auto-install base requirements and the right torch/vision based on CUDA/ROCm (official PyTorch indexes)."
    )
    ap.add_argument("--base", default=BASE_REQ, help="Path to requirements-base.txt")
    ap.add_argument("--prefer-cpu", action="store_true", help="Prefer CPU if unsure")
    ap.add_argument("--force", default=None,
                    help="Override autodetect: cpu | cuda11_8 | cuda12_6 | cuda12_8 | cuda12_9 | rocm6_3")
    args = ap.parse_args()

    # 1) upgrade pip
    run(f"{sys.executable} -m pip install --upgrade pip")

    # 2) install base requirements first
    if not os.path.exists(args.base):
        print(f"Base requirements file not found: {args.base}", file=sys.stderr)
        sys.exit(1)
    run(f"{sys.executable} -m pip install -r {args.base}")

    # 3) detect CUDA/ROCm and choose channel
    cuda_ver = detect_cuda_version()
    if cuda_ver:
        print(f"Detected CUDA: {cuda_ver}")
    elif detect_rocm():
        print("Detected ROCm runtime.")
    else:
        print("No CUDA/ROCm detected.")

    pt_key = pick_torch_channel(cuda_ver, args.force, args.prefer_cpu)
    pt_index = PT_INDEXES[pt_key]
    print(f"Selected PyTorch channel: {pt_key} -> {pt_index}")

    # 4) install torch + torchvision using official index-url style
    run(f"{sys.executable} -m pip install torch torchvision --index-url {pt_index}")

    print("\n✅ Installation complete.")
    print("   If you need the Thorlabs TSI SDK package, run:")
    print('     pip install "path/to/thorlabs_tsi_camera_python_sdk_package.zip"')

if __name__ == "__main__":
    main()
