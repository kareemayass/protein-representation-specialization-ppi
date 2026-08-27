#!/bin/bash
#SBATCH --job-name=fam_spans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fam_spans_%j.out
#SBATCH --error=logs/fam_spans_%j.err

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
  scripts/functional_regions/extract_bernett_gold_family_spans.py

STATUS=$?

if [ "$STATUS" -eq 0 ] \
   && [ -s results/bernett_gold_family_spans/intra1_gold_family_spans.tsv.gz ] \
   && [ -s results/bernett_gold_family_spans/intra0_gold_family_spans.tsv.gz ] \
   && [ -s results/bernett_gold_family_spans/intra2_gold_family_spans.tsv.gz ]
then
    echo "FAMILY_SPAN_JOB_COMPLETE"
else
    echo "FAMILY_SPAN_JOB_FAILED status=$STATUS"
    /bin/false
fi
