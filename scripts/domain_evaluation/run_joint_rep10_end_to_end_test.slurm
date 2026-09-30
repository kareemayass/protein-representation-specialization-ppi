#!/bin/bash
#SBATCH --job-name=rep10_e2e_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/rep10_e2e_test_%j.out
#SBATCH --error=logs/rep10_e2e_test_%j.err


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

AUDIT="$PROJECT/audits/end_to_end_resolvable_instances"
OUTPUT="$PROJECT/results/joint/rep_preserve_alpha10_seed47/evaluations/end_to_end"

test -s "$OUTPUT/selected_decoding.json"
test -s "$AUDIT/test/gold_regions.tsv.gz"
test -s "$AUDIT/test/ambiguous_ignore_spans.tsv.gz"

python -m py_compile \
  scripts/domain_evaluation/evaluate_joint_end_to_end_rep10.py

echo "Started: $(date -Is)"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"

python -u \
  scripts/domain_evaluation/evaluate_joint_end_to_end_rep10.py \
  --mode test \
  --token-budget 4096 \
  --progress-every 100

EVAL_STATUS=$?
if [ "$EVAL_STATUS" -ne 0 ]; then
    echo "ERROR: evaluator failed with status $EVAL_STATUS"
    exit "$EVAL_STATUS"
fi

SUMMARY="$OUTPUT/test_summary.json"

test -s "$SUMMARY"
test -s "$OUTPUT/test_predictions.tsv.gz"
test -s "$OUTPUT/test_gold_regions.tsv.gz"

python - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text(encoding="utf-8"))

assert result["status"] == "COMPLETE"
assert result["checkpoint_state"]["completed_epochs"] == 4
assert result["checkpoint_state"]["global_optimizer_step"] == 33632
assert result["test_metrics"]["evaluable_gold_regions"] == 34852

required = {
    "localization_precision_iou_0_5",
    "localization_recall_iou_0_5",
    "localization_f1_iou_0_5",
    "conditional_id_accuracy_among_matched",
    "end_to_end_precision",
    "end_to_end_recall",
    "end_to_end_f1",
    "oracle_gold_span_id_accuracy",
}

missing = required - set(result["test_metrics"])

assert not missing, missing

print(
    json.dumps(
        result["test_metrics"],
        indent=2,
        sort_keys=True,
    )
)

print("END_TO_END_TEST_PREFLIGHT_PASSED")
PY

PREFLIGHT_STATUS=$?
if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
    echo "ERROR: result preflight failed with status $PREFLIGHT_STATUS"
    exit "$PREFLIGHT_STATUS"
fi

echo "Summary: $SUMMARY"
echo "Finished: $(date -Is)"
echo "REP10_END_TO_END_TEST_JOB_COMPLETE"
