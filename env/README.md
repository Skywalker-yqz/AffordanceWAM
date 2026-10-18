# Environment

AffordanceWAM uses a Python environment shared by training and inference. The
installation below starts from a standard Linux host and downloads packages from
PyPI and the official PyTorch wheel index.

## Environment Overview

| Component | Version |
|-----------|---------|
| Operating system | Linux x86_64; Ubuntu 22.04 or 24.04 |
| Python | 3.10 or 3.11 |
| PyTorch / torchvision | 2.6.0 / 0.21.0 |
| CUDA runtime | 12.4, included with the PyTorch wheels |
| NVIDIA driver | 550.54.15 or newer |
| CUDA NVCC | 12.4.131, installed into the virtual environment |
| DeepSpeed | 0.16.9 |
| Transformers | 4.49.0 |
| NumPy | 1.26.4 |
| Attention | PyTorch scaled dot-product attention (SDPA) |
| Optimizer | PyTorch AdamW with DeepSpeed ZeRO-2 |

Use an NVIDIA GPU with BF16 support, including A100, H100, or H800. Available GPU
memory and the number of workers determine the batch size and parallelism. The
installer adds the CUDA NVCC component to the virtual environment for DeepSpeed
compatibility checks. Attention uses the kernels provided by PyTorch.

## Install from a Fresh Host

Install the NVIDIA driver using your operating system or cloud provider's driver
package, reboot if requested, and confirm that `nvidia-smi` lists the GPU. Driver
installation is a host operation; Python environments use the installed driver.
The driver requirement above follows the
[CUDA 12.4 update 1 release notes](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html).

On Ubuntu 22.04, install Python and the basic command-line tools:

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv git curl ca-certificates
```

On systems with another default Python, install `git`, `curl`, and
`ca-certificates` through the system package manager, then install Python 3.10 or
3.11 with [Miniforge](https://github.com/conda-forge/miniforge):

```bash
curl -L -o /tmp/Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -n affordancewam-python python=3.10 -y
conda activate affordancewam-python
```

Clone the repository and run the installer:

```bash
git clone https://github.com/Skywalker-yqz/AffordanceWAM.git
cd AffordanceWAM
bash env/install.sh nvidia
source .venv/bin/activate
export CUDA_HOME="$VIRTUAL_ENV"
python env/check.py --device cuda --training
```

The installer creates `.venv`, installs the pinned dependencies and NVIDIA NVCC,
installs the project in editable mode, runs `pip check`, and checks imports, CPU
attention, and video I/O. The NVCC archive is verified against the SHA-256 in
[NVIDIA's CUDA 12.4.1 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.1.json).
Set `WAM_PYTHON` to select a Python interpreter, or `WAM_ENV_DIR` to place the
environment elsewhere:

```bash
WAM_PYTHON=python3.11 WAM_ENV_DIR="$HOME/venvs/affordancewam" \
  bash env/install.sh nvidia
source "$HOME/venvs/affordancewam/bin/activate"
export CUDA_HOME="$VIRTUAL_ENV"
```

The CUDA and torchvision versions follow the
[official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v260).
Keep the PyTorch and torchvision pair together when creating an environment.

## Manual Installation

The equivalent commands, from the repository root, are:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python env/install_nvcc.py
export CUDA_HOME="$VIRTUAL_ENV"
python -m pip install --index-url https://pypi.org/simple \
  pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --index-url https://pypi.org/simple -r env/requirements.txt
DS_BUILD_OPS=0 python -m pip install --index-url https://pypi.org/simple \
  --no-build-isolation -c env/requirements.txt deepspeed==0.16.9
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python env/check.py --device cuda --training
```

DeepSpeed is installed after PyTorch, with ahead-of-time extension builds disabled.
Its CUDA compatibility checks use the environment's `nvcc`. The training
configuration selects PyTorch AdamW, so it does not compile the DeepSpeed
FusedAdam extension. See the
[DeepSpeed installation guide](https://www.deepspeed.ai/tutorials/advanced-install/)
for extension build options.

## Verify the Runtime

The checker runs model and tokenizer imports, masked attention forward/backward,
and FFmpeg/PyAV video I/O. CUDA mode also checks BF16 support. It uses synthetic
inputs and does not download model weights.

```bash
python env/check.py --device cuda --training
```

Check communication and a BF16 ZeRO-2 AdamW update on one or more GPUs:

```bash
torchrun --standalone --nproc_per_node=1 env/check.py --device cuda --distributed
# Set the worker count to the number of visible GPUs on this host.
torchrun --standalone --nproc_per_node=8 env/check.py --device cuda --distributed
```

For development on a CPU host:

```bash
WAM_ENV_DIR=.venv-cpu bash env/install.sh cpu
source .venv-cpu/bin/activate
python env/check.py --device cpu --training
python -m pip install -r env/requirements-dev.txt
python -m pytest -q
```

Training and model inference use CUDA. The CPU profile provides dependency,
module, and algorithm checks.

## Troubleshooting

- **`torch.cuda.is_available()` is false:** check `nvidia-smi`, GPU visibility, and
  `python -c 'import torch; print(torch.__version__, torch.version.cuda)'`. The
  NVIDIA profile should report a `+cu124` PyTorch build. Run the NVIDIA installer
  in a fresh environment if a CPU wheel is installed.
- **`No module named venv` or `ensurepip` is unavailable:** install the matching
  `python3.10-venv` package, or use the Miniforge interpreter above.
- **An existing CUDA toolkit appears in an error:** run
  `export CUDA_HOME="$VIRTUAL_ENV"` after activating the environment and confirm
  that `"$CUDA_HOME/bin/nvcc" --version` reports CUDA 12.4. The runtime libraries
  come with PyTorch; remove unrelated toolkit directories from `LD_LIBRARY_PATH`
  for this shell and rerun the runtime check.
- **A distributed launch cannot initialize:** use one process per visible GPU,
  run the single-worker check first, and verify that all workers use the same
  environment. For multiple hosts, configure the network interface and rendezvous
  address for your cluster.
