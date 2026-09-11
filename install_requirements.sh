#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc was not found. Load a CUDA 12.8 or CUDA 13.0 toolkit module first." >&2
    echo "Example: module load cuda/12.8" >&2
    exit 1
fi

NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
case "$NVCC_RELEASE" in
    12.8*) CUDA_CHANNEL="cu128" ;;
    13.0*) CUDA_CHANNEL="cu130" ;;
    *)
        echo "Unsupported nvcc release: ${NVCC_RELEASE:-unknown}. Use CUDA 12.8 or 13.0 for sm_120." >&2
        exit 1
        ;;
esac

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    "python${PYTHON_VERSION}" -m venv "$VENV_PATH"
fi
PYTHON_BIN="$VENV_PATH/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# PyTorch 2.10 is available for both CUDA channels and supports Python
# 3.10-3.14. torchvision 0.25 is its matching domain-library release.
"$PYTHON_BIN" -m pip install \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url "https://download.pytorch.org/whl/$CUDA_CHANNEL"

"$PYTHON_BIN" -m pip install -r requirements.txt

# Build only for the workstation's RTX PRO 6000 Blackwell (sm_120). Disabling
# build isolation ensures tiny-cuda-nn can import the torch installed above.
export TCNN_CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST=12.0
"$PYTHON_BIN" -m pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

"$PYTHON_BIN" - <<'PY'
import h5py
import nerfacc
import torch
import torchvision
import tinycudann

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
print("nerfacc:", nerfacc.__version__)
print("tiny-cuda-nn import: OK")
PY

echo "Environment ready: $VENV_PATH"
