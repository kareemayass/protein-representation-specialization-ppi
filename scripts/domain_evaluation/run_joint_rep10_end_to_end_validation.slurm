#!/bin/bash
#SBATCH --job-name=rep10_e2e_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/rep10_e2e_val_%j.out
#SBATCH --error=logs/rep10_e2e_val_%j.err


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="${INTERPRO_PROJECT_ROOT:-$REPO_ROOT}"
TUNA_ROOT="${TUNA_ROOT:-$REPO_ROOT/external/TUnA-R}"
VENV="${VENV:-$TUNA_ROOT/.venv}"

export INTERPRO_PROJECT_ROOT="$PROJECT"
export TUNA_ROOT
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"

module load python/3.13.2
source "$VENV/bin/activate"

export PYTHONPATH="$PROJECT/src:$PROJECT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p logs

CHECKPOINT="$PROJECT/results/joint/rep_preserve_alpha10_seed47/best_by_joint_score"
AUDIT="$PROJECT/audits/end_to_end_resolvable_instances"
OUTPUT="$PROJECT/results/joint/rep_preserve_alpha10_seed47/evaluations/end_to_end"

test -s "$CHECKPOINT/trainer_state.json"
test -s "$CHECKPOINT/shared_lora_adapter.pt"
test -s "$CHECKPOINT/stage1_head.pt"
test -s "$CHECKPOINT/stage2_domain_head.pt"

test -s "$AUDIT/validation/gold_regions.tsv.gz"
test -s "$AUDIT/validation/ambiguous_ignore_spans.tsv.gz"

python -m py_compile \
  scripts/domain_evaluation/evaluate_joint_end_to_end_rep10.py

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

echo "Started: $(date -Is)"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"

python -u \
  scripts/domain_evaluation/evaluate_joint_end_to_end_rep10.py \
  --mode validation \
  --token-budget 4096 \
  --progress-every 100

EVAL_STATUS=$?
if [ "$EVAL_STATUS" -ne 0 ]; then
    echo "ERROR: evaluator failed with status $EVAL_STATUS"
    exit "$EVAL_STATUS"
fi

SELECTION="$OUTPUT/selected_decoding.json"
GRID="$OUTPUT/validation_decoding_grid.tsv"

test -s "$SELECTION"
test -s "$GRID"

python - "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text(encoding="utf-8"))

assert result["status"] == "COMPLETE"
assert result["selection_split"] == "validation"
assert result["objective"] == "localization_span_f1_at_iou_0.5"
assert result["checkpoint_state"]["completed_epochs"] == 4
assert result["checkpoint_state"]["global_optimizer_step"] == 33632
assert result["selected"]["minimum_span_length"] == 10
assert result["selected"]["evaluable_gold_regions"] == 39635

print(
    json.dumps(
        result["selected"],
        indent=2,
        sort_keys=True,
    )
)

print("END_TO_END_VALIDATION_PREFLIGHT_PASSED")
PY

PREFLIGHT_STATUS=$?
if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
    echo "ERROR: result preflight failed with status $PREFLIGHT_STATUS"
    exit "$PREFLIGHT_STATUS"
fi

echo "Finished: $(date -Is)"
echo "JOINT64_END_TO_END_VALIDATION_JOB_COMPLETE"
