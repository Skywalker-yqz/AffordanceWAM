#!/usr/bin/env bash
# Create a virtual environment and install AffordanceWAM from public packages.
set -euo pipefail

profile="${1:-nvidia}"
case "$profile" in
  nvidia|cpu) ;;
  *) printf 'Usage: bash env/install.sh [nvidia|cpu]\n' >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${WAM_ENV_DIR:-$repo_root/.venv}"
bootstrap_python="${WAM_PYTHON:-python3}"
export PYTHONNOUSERSITE=1

"$bootstrap_python" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    raise SystemExit("Use Python 3.10 or 3.11, for example WAM_PYTHON=python3.10 bash env/install.sh nvidia.")
PY

venv_dir="$("$bootstrap_python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$venv_dir")"
if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
  "$bootstrap_python" -m venv "$venv_dir"
fi
wam_python="$venv_dir/bin/python"
"$wam_python" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    raise SystemExit("The existing virtual environment must use Python 3.10 or 3.11.")
PY

if [[ "$profile" == nvidia ]]; then
  "$wam_python" "$repo_root/env/install_nvcc.py"
  export CUDA_HOME="$venv_dir"
  export PATH="$venv_dir/bin:$PATH"
fi

"$wam_python" -m pip install --index-url https://pypi.org/simple \
  pip==25.0.1 setuptools==75.8.0 wheel==0.45.1

torch_build=cu124
[[ "$profile" != cpu ]] || torch_build=cpu
torch_index="https://download.pytorch.org/whl/$torch_build"
"$wam_python" -m pip install --index-url "$torch_index" \
  "torch==2.6.0+$torch_build" "torchvision==0.21.0+$torch_build"

"$wam_python" -m pip install --index-url https://pypi.org/simple \
  --requirement "$repo_root/env/requirements.txt"

DS_BUILD_OPS=0 "$wam_python" -m pip install --index-url https://pypi.org/simple \
  --no-build-isolation --constraint "$repo_root/env/requirements.txt" deepspeed==0.16.9

"$wam_python" -m pip install --no-build-isolation --no-deps --editable "$repo_root"
"$wam_python" -m pip check
"$wam_python" "$repo_root/env/check.py" --device cpu --training
printf '\nInstallation complete. Activate with: source %q/bin/activate\n' "$venv_dir"
if [[ "$profile" == nvidia ]]; then
  printf 'Set the CUDA compiler prefix: export CUDA_HOME="$VIRTUAL_ENV"\n'
  printf 'Check the GPU with: python env/check.py --device cuda --training\n'
fi
