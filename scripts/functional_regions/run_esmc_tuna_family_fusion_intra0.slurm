#!/bin/bash
#SBATCH --job-name=fam_i0_fuse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fam_i0_fuse_%j.out
#SBATCH --error=logs/fam_i0_fuse_%j.err

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

python -u \
  scripts/functional_regions/evaluate_esmc_tuna_family_fusion_intra0.py

STATUS=$?

OUT="results/family_compatibility_overnight/tuna_fusion_intra0"

if [ "$STATUS" -eq 0 ] \
   && [ -s "$OUT/final_intra0_fusion_model.pkl" ] \
   && [ -s "$OUT/fusion_summary.tsv" ]
then
    echo "FAMILY_INTRA0_FUSION_COMPLETE"
else
    echo "FAMILY_INTRA0_FUSION_FAILED status=$STATUS"
    /bin/false
fi
