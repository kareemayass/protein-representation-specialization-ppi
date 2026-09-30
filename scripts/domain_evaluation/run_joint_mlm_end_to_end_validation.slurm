#!/bin/bash
#SBATCH --job-name=joint64_mlm_e2e_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/joint64_mlm_e2e_val_%j.out
#SBATCH --error=logs/joint64_mlm_e2e_val_%j.err


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

# Keep repository code importable when the artifact directory is elsewhere.
export PYTHONPATH="$REPO_ROOT/src:$PROJECT/src:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p logs

CHECKPOINT="$PROJECT/results/joint/top64_mlm_preserving_production_seed47/best_by_joint_score"
AUDIT="$PROJECT/audits/end_to_end_resolvable_instances"
OUTPUT="$PROJECT/results/joint/top64_mlm_preserving_production_seed47/evaluations/end_to_end"

test -s "$CHECKPOINT/trainer_state.json"
test -s "$CHECKPOINT/shared_lora_adapter.pt"
test -s "$CHECKPOINT/stage1_head.pt"
test -s "$CHECKPOINT/stage2_domain_head.pt"

test -s "$AUDIT/validation/gold_regions.tsv.gz"
test -s "$AUDIT/validation/ambiguous_ignore_spans.tsv.gz"

python -m py_compile \
  scripts/domain_evaluation/evaluate_joint_end_to_end_mlm.py

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

echo "Started: $(date -Is)"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"

python -u \
  scripts/domain_evaluation/evaluate_joint_end_to_end_mlm.py \
  --mode validation \
  --token-budget 4096 \
  --progress-every 100

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
assert result["checkpoint_state"]["completed_epochs"] == 5
assert result["checkpoint_state"]["global_optimizer_step"] == 42040
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

echo "Finished: $(date -Is)"
echo "JOINT64_END_TO_END_VALIDATION_JOB_COMPLETE"
