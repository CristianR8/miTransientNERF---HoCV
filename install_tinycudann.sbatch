#!/usr/bin/env bash
#SBATCH --job-name=tcnn-build
#SBATCH --account=crisr
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=tcnn-build-%j.log

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CUDA_ENV="${CUDA_ENV:-$HOME/.conda/envs/mitra-build}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"

[[ -x "$PYTHON_BIN" ]] || {
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
}
[[ -x "$CUDA_ENV/bin/nvcc" ]] || {
    echo "CUDA compiler not found: $CUDA_ENV/bin/nvcc" >&2
    exit 1
}

export CUDA_HOME="$CUDA_ENV"
export PATH="$CUDA_HOME/bin:$PATH"
export TCNN_CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-2}"

echo "Project: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
nvcc --version

"$PYTHON_BIN" -m pip install \
    "setuptools>=68,<81" wheel ninja packaging

"$PYTHON_BIN" -m pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

"$PYTHON_BIN" - <<'PY'
import torch
import tinycudann

print("tinycudann import: OK")
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
PY
