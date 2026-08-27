#!/bin/bash
#SBATCH --job-name=bernett_spans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bernett_spans_%j.out
#SBATCH --error=logs/bernett_spans_%j.err

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
  scripts/functional_regions/extract_bernett_gold_domain_spans.py

STATUS=$?

echo
echo "status=$STATUS"
echo "finished=$(date -Is)"

if [ "$STATUS" -ne 0 ]; then
    echo "BERNETT_GOLD_DOMAIN_SPANS_FAILED"
    /bin/false
fi
