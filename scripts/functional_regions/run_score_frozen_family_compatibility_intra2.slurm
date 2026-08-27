#!/bin/bash
#SBATCH --job-name=fam_i2_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fam_i2_score_%j.out
#SBATCH --error=logs/fam_i2_score_%j.err

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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u \
  scripts/functional_regions/score_frozen_family_compatibility_intra2.py

STATUS=$?

OK=1

for SEED in 47 1047 2047
do
    [ -s results/family_compatibility_intra2/learned_seed${SEED}_intra2_pair_scores.tsv.gz ] || OK=0
done

if [ "$STATUS" -eq 0 ] \
   && [ "$OK" -eq 1 ] \
   && [ -s results/family_compatibility_intra2/frozen_family_intra2_summary.tsv ]
then
    echo "FAMILY_INTRA2_SCORE_COMPLETE"
else
    echo "FAMILY_INTRA2_SCORE_FAILED status=$STATUS outputs_ok=$OK"
    /bin/false
fi
