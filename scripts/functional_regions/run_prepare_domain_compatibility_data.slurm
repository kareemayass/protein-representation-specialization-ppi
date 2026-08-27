#!/bin/bash
#SBATCH --job-name=dom_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dom_prep_%j.out
#SBATCH --error=logs/dom_prep_%j.err

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

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python -u \
  scripts/functional_regions/prepare_domain_compatibility_data.py

STATUS=$?

echo
echo "status=$STATUS"
echo "finished=$(date -Is)"

if [ "$STATUS" -ne 0 ]; then
    echo "DOMAIN_COMPATIBILITY_PREP_FAILED"
    /bin/false
fi
