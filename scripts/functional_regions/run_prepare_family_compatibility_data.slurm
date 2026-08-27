#!/bin/bash
#SBATCH --job-name=fam_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fam_prep_%j.out
#SBATCH --error=logs/fam_prep_%j.err

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
  scripts/functional_regions/prepare_family_compatibility_data.py

STATUS=$?

if [ "$STATUS" -eq 0 ] \
   && [ -s results/family_compatibility_overnight/base_family_data.pt ] \
   && [ -s results/family_compatibility_overnight/prep_summary.tsv ]
then
    echo "FAMILY_PREP_JOB_COMPLETE"
else
    echo "FAMILY_PREP_JOB_FAILED status=$STATUS"
    /bin/false
fi
