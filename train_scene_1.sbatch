#!/usr/bin/env bash
#SBATCH --job-name=tnrf-scene1
#SBATCH --account=crisr
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR" results

# Create .venv once on a GPU/login node with ./install_requirements.sh.
# Override when submitting if needed:
# sbatch --export=ALL,PYTHON_BIN=/path/to/python,MAX_STEPS=250000 train_scene_1.sbatch
if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

CONFIG="$PROJECT_ROOT/configs/train/simulated/scene_1_5views.ini"
DATASET="$PROJECT_ROOT/data/scene_1.h5"
MAX_STEPS="${MAX_STEPS:-10000}"
SUMMARY_EVERY="${SUMMARY_EVERY:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
RUN_NAME="${RUN_NAME:-scene_1_5views_${SLURM_JOB_ID:-local}}"

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 1; }

echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Host:       $(hostname)"
echo "Python:     $PYTHON_BIN"
echo "Dataset:    $DATASET"
echo "Run name:   $RUN_NAME"
echo "Max steps:  $MAX_STEPS"
echo "Checkpoint: every $CHECKPOINT_EVERY steps"
nvidia-smi

# Fail early if the environment, GPU, or copied HDF5 file is not usable.
"$PYTHON_BIN" - <<'PY'
import h5py
import torch
import nerfacc
import tinycudann
from radiance_fields.ngp import NGPRadianceField

assert torch.cuda.is_available(), "PyTorch cannot see the Slurm GPU"
with h5py.File("data/scene_1.h5", "r") as scene:
    expected = (25, 256, 256, 1034, 3)
    assert scene["transients"].shape == expected, scene["transients"].shape
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Dataset metadata and training imports: OK")
PY

srun "$PYTHON_BIN" train.py \
    -c "$CONFIG" \
    --device cuda:0 \
    --max_steps "$MAX_STEPS" \
    --steps_til_checkpoint "$CHECKPOINT_EVERY" \
    --summary_every "$SUMMARY_EVERY" \
    --exp_name "$RUN_NAME" \
    --final True
