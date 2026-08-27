# Protein Representation Specialization and PPI Prediction

Code and experiment configurations for studying how task-specific specialization of protein language model representations affects downstream protein–protein interaction (PPI) prediction.

This repository contains the training, evaluation, and analysis code retained for the associated research project.

## Experiments

The repository covers:

- PPI-specific LoRA adaptation
- InterPro Domain-supervised specialization of ESMC
- masked-language-model preservation during specialization
- direct representation-preservation regularization
- downstream transfer to TUnA PPI prediction
- representation-space geometry analysis
- Domain- and Family-level functional-region compatibility and fusion

## Repository structure

- `configs/` — frozen experiment configurations
- `scripts/interpro_training/` — specialization training
- `scripts/domain_evaluation/` — Domain and MLM evaluation
- `scripts/geometry/` — representation-space analyses
- `scripts/functional_regions/` — Domain/Family compatibility experiments
- `scripts/lora/` — LoRA paper analyses
- `src/interpro_*` — InterPro specialization implementation
- `src/tuna/` — project-specific TUnA modifications

## Installation

The package versions used for the final experiments are recorded in `requirements.txt`.

Create an environment and install the dependencies with:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt


## TUnA-R dependency

The downstream PPI experiments build on TUnA-R.

Project-specific TUnA modifications retained here are:

- `src/tuna/train.py`
- `src/tuna/pl_modules/lit_ppi.py`
- `configs/dataset/gold-standard.yaml`

## External artifacts

Large datasets, checkpoints, protein embeddings, and experiment outputs are not included in this repository.

Scripts that require the InterPro data/results tree use:

    export INTERPRO_PROJECT_ROOT=/path/to/interpro-data-and-results

The final specialization experiments expect the processed split at:

    data/processed/top100_domain_pipeline_v1/splits_top64_length_filtered_v1/

relative to `INTERPRO_PROJECT_ROOT`.

The repository does not include the data-preparation pipeline.

## Running experiments

The `.slurm` files preserve the production arguments and compute settings used for the final experiments.

Filesystem paths have been made portable. Cluster-specific Slurm resource directives may still need adjustment on another system.

For example:

    export INTERPRO_PROJECT_ROOT=/path/to/interpro
    export TUNA_ROOT=/path/to/TUnA-R
    sbatch scripts/interpro_training/run_joint_rep_preserve_alpha10_production.slurm

The provided Slurm wrappers add the repository `src/` directory to `PYTHONPATH`.

## Frozen downstream configurations

Hydra snapshots for the final downstream TUnA experiments are provided under:

- `configs/tuna/base/`
- `configs/tuna/standard/`
- `configs/tuna/mlm_preserved/`
- `configs/tuna/rep_preserved_alpha10/`

The final LoRA-r8 configuration is provided under `configs/lora/`.

## Reproducibility

See `REPRODUCIBILITY.md` for a mapping between the main experiments and their corresponding scripts and configurations.

## Reproducibility note

The scientific result and checkpoint lineage of the original Standard InterPro specialization experiment were preserved.

Some shared source files were subsequently revised during development before this public repository was assembled. Where exact historical source bytes were unavailable, this repository contains reviewed implementations corresponding to the reported method rather than claiming byte-for-byte recovery of every early source file.
