#!/usr/bin/env python3
"""Install NVIDIA's CUDA 12.4 NVCC component into the current virtual environment."""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

# NVIDIA's redistrib_12.4.1.json manifest pins this component and its checksum.
URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/"
    "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.4.131-archive.tar.xz"
)
SHA256 = "7ffba1ada0e4b8c17e451ac7a60d386aa2642ecd08d71202a0b100c98bd74681"


def main() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("The NVIDIA installation profile targets Linux x86_64.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this helper with the project's virtual-environment Python.")
    prefix = Path(sys.prefix)
    marker = prefix / ".affordancewam-nvcc-sha256"
    nvcc = prefix / "bin" / "nvcc"
    if marker.is_file() and marker.read_text().strip() == SHA256 and nvcc.is_file():
        subprocess.run([str(nvcc), "--version"], check=True)
        return
    with tempfile.TemporaryDirectory(prefix="affordancewam-nvcc-") as temp:
        temp_path = Path(temp)
        archive = temp_path / "nvcc.tar.xz"
        print("Downloading NVIDIA CUDA NVCC 12.4.131 (51 MB)...", flush=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as source, archive.open("wb") as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
                digest.update(block)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("NVCC archive checksum does not match NVIDIA's release manifest.")
        with tarfile.open(archive, "r:xz") as package:
            for member in package.getmembers():
                if not (temp_path / member.name).resolve().is_relative_to(temp_path):
                    raise RuntimeError("Invalid path in NVCC archive.")
            package.extractall(temp_path)
        unpacked = temp_path / "cuda_nvcc-linux-x86_64-12.4.131-archive"
        shutil.copytree(unpacked, prefix, dirs_exist_ok=True)
    marker.write_text(SHA256 + "\n")
    subprocess.run([str(nvcc), "--version"], check=True)


if __name__ == "__main__":
    main()
