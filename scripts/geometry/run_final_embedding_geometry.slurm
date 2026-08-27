#!/bin/bash
#SBATCH --job-name=rep_geometry_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rep_geometry_all_%j.out
#SBATCH --error=logs/rep_geometry_all_%j.err

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="${INTERPRO_PROJECT_ROOT:-$REPO_ROOT}"
TUNA_ROOT="${TUNA_ROOT:-$REPO_ROOT/external/TUnA-R}"
VENV="${VENV:-$TUNA_ROOT/.venv}"

export INTERPRO_PROJECT_ROOT="$PROJECT"
export TUNA_ROOT
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p logs

module load python/3.13.2
source "$VENV/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1

echo "Started: $(date -Is)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

echo
echo "========================================"
echo "STEP 1: GLOBAL REPRESENTATION GEOMETRY"
echo "========================================"

python -u scripts/geometry/audit_final_embedding_geometry.py
GLOBAL_STATUS=$?

if [ "$GLOBAL_STATUS" -ne 0 ]; then
    echo "ERROR: global geometry failed: $GLOBAL_STATUS"
    exit "$GLOBAL_STATUS"
fi

GLOBAL="results/representation_geometry/standard_mlm_rep10/geometry_summary.tsv"

if [ ! -s "$GLOBAL" ]; then
    echo "ERROR: global geometry summary missing"
    exit 1
fi

echo
echo "========================================"
echo "STEP 2: ACTUAL BERNETT PPI-PAIR GEOMETRY"
echo "========================================"

python -u scripts/geometry/audit_ppi_pair_geometry.py
PPI_STATUS=$?

if [ "$PPI_STATUS" -ne 0 ]; then
    echo "ERROR: PPI-pair geometry failed: $PPI_STATUS"
    exit "$PPI_STATUS"
fi

FINAL="results/representation_geometry/standard_mlm_rep10/geometry_summary_with_ppi.tsv"

if [ ! -s "$FINAL" ]; then
    echo "ERROR: final combined summary missing"
    exit 1
fi

echo
echo "========================================"
echo "FINAL COMBINED GEOMETRY RESULTS"
echo "========================================"

column -t -s $'\t' "$FINAL"

echo
echo "FINAL_GEOMETRY_PIPELINE_COMPLETE"
echo "Finished: $(date -Is)"
