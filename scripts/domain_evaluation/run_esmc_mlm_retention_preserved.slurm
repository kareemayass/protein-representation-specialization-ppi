#!/bin/bash
#SBATCH --job-name=esmc_mlm_preserved
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/esmc_mlm_preserved_%j.out
#SBATCH --error=logs/esmc_mlm_preserved_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="${INTERPRO_PROJECT_ROOT:-$REPO_ROOT}"
TUNA_ROOT="${TUNA_ROOT:-$REPO_ROOT/external/TUnA-R}"
VENV="${VENV:-$TUNA_ROOT/.venv}"

export INTERPRO_PROJECT_ROOT="$PROJECT"
export TUNA_ROOT
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

FASTA="$PROJECT/data/processed/top100_domain_pipeline_v1/mlm_diagnostic_v1/heldout_20k_len30_1024.fasta.gz"

CHECKPOINT="$PROJECT/results/joint/top64_mlm_preserving_production_seed47/best_by_joint_score"

OUTPUT="$PROJECT/results/joint/top64_mlm_preserving_production_seed47/evaluations/mlm_retention"

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
mkdir -p "$OUTPUT"

test -s "$FASTA"
test -s "$CHECKPOINT/shared_lora_adapter.pt"
test -s "$CHECKPOINT/trainer_state.json"
test -s "$CHECKPOINT/stage1_head.pt"
test -s "$CHECKPOINT/stage2_domain_head.pt"

python -m py_compile \
  scripts/domain_evaluation/evaluate_esmc_mlm_retention.py

echo "Started: $(date -Is)"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"

python scripts/domain_evaluation/evaluate_esmc_mlm_retention.py \
  --fasta "$FASTA" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --mask-fraction 0.15 \
  --mask-seed 47 \
  --token-budget 8192 \
  --progress-every 50

echo "ESMC_MLM_RETENTION_JOB_COMPLETE"
echo "Finished: $(date -Is)"
