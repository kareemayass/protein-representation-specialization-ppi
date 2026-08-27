#!/bin/bash
#SBATCH --job-name=joint64_mlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/joint64_mlm_%j.out
#SBATCH --error=logs/joint64_mlm_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="${INTERPRO_PROJECT_ROOT:-$REPO_ROOT}"
TUNA_ROOT="${TUNA_ROOT:-$REPO_ROOT/external/TUnA-R}"
VENV="${VENV:-$TUNA_ROOT/.venv}"

export INTERPRO_PROJECT_ROOT="$PROJECT"
export TUNA_ROOT
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SPLIT_DIR="$PROJECT/data/processed/top100_domain_pipeline_v1/splits_top64_length_filtered_v1"
OUTDIR="$PROJECT/results/joint/top64_mlm_preserving_production_seed47"
PLAN_PATH="$PROJECT/results/joint/top64_mlm_preserving_production_seed47_plan.json"

MAX_EPOCHS=15
PATIENCE=3
MIN_DELTA=0.0001

MLM_MASK_FRACTION=0.15
MLM_TARGET_GRADIENT_RATIO=0.20
MLM_CALIBRATION_BATCHES=8
MLM_STAGE2_BURN_IN_TASK_STEPS=100

cd "$REPO_ROOT"

module load python/3.13.2
source "$VENV/bin/activate"

export PYTHONPATH="$PROJECT/src:$PROJECT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29953

export SPLIT_DIR
export OUTDIR
export PLAN_PATH

export MLM_MASK_FRACTION
export MLM_TARGET_GRADIENT_RATIO
export MLM_CALIBRATION_BATCHES
export MLM_STAGE2_BURN_IN_TASK_STEPS

mkdir -p logs
mkdir -p "$(dirname "$OUTDIR")"

test -s "$SPLIT_DIR/SUCCESS"
test "$(wc -l < "$SPLIT_DIR/selected_domains.txt")" -eq 64

python -m py_compile \
  src/interpro_joint/data.py \
  src/interpro_joint/model.py \
  src/interpro_joint/curriculum.py \
  src/interpro_joint/engine.py \
  src/interpro_joint/runtime.py \
  src/interpro_joint/mlm.py \
  src/interpro_joint/mlm_engine.py \
  scripts/interpro_training/train_joint_curriculum_mlm.py \
  scripts/interpro_training/plan_joint_epoch_steps.py

python scripts/interpro_training/plan_joint_epoch_steps.py \
  --split-dir "$SPLIT_DIR" \
  --epochs "$MAX_EPOCHS" \
  --token-budget 4096 \
  --gradient-accumulation-steps 4 \
  --world-size 4 \
  --seed 47 \
  --format json \
  > "$PLAN_PATH"

eval "$(
  python scripts/interpro_training/plan_joint_epoch_steps.py \
    --split-dir "$SPLIT_DIR" \
    --epochs "$MAX_EPOCHS" \
    --token-budget 4096 \
    --gradient-accumulation-steps 4 \
    --world-size 4 \
    --seed 47 \
    --format shell
)"

export TOTAL_OPTIMIZER_STEPS
export VALIDATE_EVERY

if [[ -s "$OUTDIR/latest/trainer_state.json" ]]
then
    RESUME_MODE=auto
    OVERWRITE_FLAG=""
    echo "Resuming existing MLM-preserving production checkpoint."
else
    RESUME_MODE=none
    OVERWRITE_FLAG=--overwrite
    echo "Starting a fresh MLM-preserving production run."
fi

export RESUME_MODE
export OVERWRITE_FLAG

echo
echo "Started: $(date -Is)"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Dataset: $SPLIT_DIR"
echo "Output: $OUTDIR"
echo "Maximum epochs: $MAX_EPOCHS"
echo "Patience: $PATIENCE"
echo "Minimum improvement: $MIN_DELTA"
echo "Total optimizer steps: $TOTAL_OPTIMIZER_STEPS"
echo "Validate every: $VALIDATE_EVERY"
echo
echo "MLM mask fraction: $MLM_MASK_FRACTION"
echo "MLM target gradient ratio: $MLM_TARGET_GRADIENT_RATIO"
echo "MLM calibration batches/task: $MLM_CALIBRATION_BATCHES"
echo "Stage 1 calibration: after LR warmup"
echo "Stage 2 task-only burn-in: $MLM_STAGE2_BURN_IN_TASK_STEPS updates"
echo
cat "$PLAN_PATH"
echo

srun bash -c '
set -euo pipefail

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

ARGS=(
  --split-dir "$SPLIT_DIR"
  --output-dir "$OUTDIR"
  --total-optimizer-steps "$TOTAL_OPTIMIZER_STEPS"
  --validate-every "$VALIDATE_EVERY"
  --max-epochs 15
  --patience 3
  --min-delta 0.0001
  --log-every 100
  --gradient-accumulation-steps 4
  --train-token-budget 4096
  --validation-token-budget 4096
  --classifier-lr 1e-3
  --lora-lr 3e-4
  --weight-decay 0.01
  --warmup-fraction 0.05
  --max-grad-norm 1.0
  --stage1-head mlp
  --classifier-dropout 0.1
  --seed 47
  --num-workers 0
  --expected-world-size 4
  --resume "$RESUME_MODE"
  --max-runtime-minutes 940
  --runtime-buffer-minutes 20

  --mlm-mask-fraction "$MLM_MASK_FRACTION"
  --mlm-target-gradient-ratio "$MLM_TARGET_GRADIENT_RATIO"
  --mlm-calibration-batches "$MLM_CALIBRATION_BATCHES"
  --mlm-stage2-burn-in-task-steps "$MLM_STAGE2_BURN_IN_TASK_STEPS"
)

if [[ -n "$OVERWRITE_FLAG" ]]
then
    ARGS+=("$OVERWRITE_FLAG")
fi

echo \
  "rank=$SLURM_PROCID local_rank=$SLURM_LOCALID world_size=$SLURM_NTASKS"

python -u \
  scripts/interpro_training/train_joint_curriculum_mlm.py \
  "${ARGS[@]}"
'

python - <<'PY'
import json
import os
from pathlib import Path

output = Path(
    os.environ["OUTDIR"]
)

summary_path = (
    output / "run_summary.json"
)

if not summary_path.is_file():
    raise FileNotFoundError(
        summary_path
    )

summary = json.loads(
    summary_path.read_text(
        encoding="utf-8"
    )
)

allowed = {
    "COMPLETE",
    "EARLY_STOPPED",
    "ALLOCATION_STOP",
}

assert summary["status"] in allowed
assert summary["maximum_epochs"] == 15
assert summary["patience"] == 3

print(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
)

calibration_path = (
    output
    / "mlm_gradient_calibration.json"
)

if calibration_path.is_file():
    print(
        "\n===== MLM CALIBRATION STATE ====="
    )

    print(
        calibration_path.read_text(
            encoding="utf-8"
        )
    )

if summary["status"] == "ALLOCATION_STOP":
    print(
        "\nThe allocation ended safely. "
        "Resubmit this same Slurm script "
        "to resume from latest with the "
        "already calibrated MLM lambdas."
    )
PY

echo
echo "JOINT TOP64 MLM-PRESERVING PRODUCTION JOB FINISHED"
echo "Finished: $(date -Is)"
