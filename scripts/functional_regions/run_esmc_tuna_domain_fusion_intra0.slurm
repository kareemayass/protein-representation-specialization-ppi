#!/bin/bash
#SBATCH --job-name=esmc_dom_fuse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/esmc_dom_fuse_%j.out
#SBATCH --error=logs/esmc_dom_fuse_%j.err

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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python -u \
  scripts/functional_regions/evaluate_esmc_tuna_domain_fusion_intra0.py

STATUS=$?

echo
echo "status=$STATUS"
echo "finished=$(date -Is)"

if [ "$STATUS" -ne 0 ]; then
    echo "ESMC_TUNA_DOMAIN_FUSION_FAILED"
    /bin/false
fi
