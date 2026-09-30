# Reproducibility guide

This document maps the principal experiment branches to the code and configurations retained in this repository.

## Reproduction scope

| Level | What runs | Requirements | What it establishes |
|---|---|---|---|
| Report visualization | Commands in [reports/README.md](reports/README.md) | Matplotlib | Regenerates both README figures from transcribed aggregate results |
| Research experiments | Training, evaluation, geometry, and fusion scripts below | Full research environment, GPUs where applicable, external data/weights/predictions, compatible TUnA-R | Requires additional artifacts; not rerun during this documentation update |

Run commands from the repository root unless stated otherwise. The scientific design is in [docs/methods.md](docs/methods.md).

## Research environment

`requirements.txt` retains the package versions recorded for the experiments. It is a package list, not a complete lockfile with Python/CUDA/transitive dependency hashes. The Slurm recipes load Python 3.13.2. A clean installation of the research stack has not been validated as part of this refresh.

On a compatible GPU host, create a dedicated environment and install the recorded requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

ESMC access/model-cache requirements and CUDA compatibility must be satisfied on that host. The production scripts set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; populate the model cache before submitting jobs. Installing packages does not supply the datasets or trained checkpoints.

## Artifact layout and HPC launch

Set `INTERPRO_PROJECT_ROOT` to the external InterPro data/results tree, `TUNA_ROOT` to a compatible TUnA-R checkout, and `VENV` to your research environment. The wrappers default `VENV` to `$TUNA_ROOT/.venv`; override it when using the repository's `.venv`.

The representation-preservation trainer expects the split directory
`$INTERPRO_PROJECT_ROOT/data/processed/top100_domain_pipeline_v1/splits_top64_length_filtered_v1/` containing:

| Artifact | Purpose |
|---|---|
| `SUCCESS`, `selected_domains.txt` | Preparation marker and the ordered list of 64 Domain IDs |
| `train.fasta`, `validation.fasta` | Protein sequences |
| `train_annotations.tsv.gz`, `validation_annotations.tsv.gz` | Localization annotations |
| `train_balanced_domain_targets.tsv.gz`, `validation_balanced_domain_targets.tsv.gz` | Domain-identification targets |
| `train_balanced_domain_target_annotations.tsv.gz`, `validation_balanced_domain_target_annotations.tsv.gz` | Target-region annotations |

This is the trainer's input contract, not a supplied dataset. Annotations use `protein_accession`, `interpro_id`, `start`, and `end`; residue coordinates are **1-based inclusive**. See [data.py](src/interpro_joint/data.py) for validation and Stage 2 target schemas. End-to-end test evaluation additionally requires the test FASTA, resolvable-region audit files, trained checkpoints, and validation-selected decoding settings referenced in its wrapper.

Before submission, adjust cluster module names, GPU/resource directives, and environment paths. The representation-preservation recipe requests **four H100 GPUs**; this is the retained production configuration, not a measured minimum requirement. Create the log directory **before** calling `sbatch`, since Slurm opens log files before the script runs:

```bash
export INTERPRO_PROJECT_ROOT=/absolute/path/to/interpro-artifacts
export TUNA_ROOT=/absolute/path/to/TUnA-R
export VENV="$PWD/.venv"
mkdir -p logs
sbatch scripts/interpro_training/run_joint_rep_preserve_alpha10_production.slurm
```

The wrappers preserve repository imports even when artifacts reside elsewhere. They remain site-specific launch recipes, not a portable scheduler abstraction. The representation-preservation wrapper now exits on failed shell commands so a failed prerequisite or distributed job does not continue into its success-summary stage.

### TUnA integration boundary

Only two project-specific TUnA Python files are bundled. The imports in `src/tuna/train.py` require external modules including `tuna.datamodule.ppi_module` and `tuna.inference.export`, and the frozen Hydra configurations require `tuna.models._transformer`. Merely adding this repository to `PYTHONPATH` does not provide those modules or guarantee that its partial `tuna` directory overrides an installed upstream package.

Use an isolated compatible upstream checkout, review and apply the retained project modifications there, and resolve config/data paths against that checkout. The exact upstream URL/revision was not recorded here, so this guide does not invent a clone command or claim a verified end-to-end invocation. Record the chosen upstream revision and environment before attempting reproduction.

### Historical fidelity and results provenance

The original Standard-specialization result and checkpoint lineage were preserved, but some shared source files were revised before repository assembly. Retained implementations correspond to the reported method where exact historical bytes were unavailable. Do not describe this release as exact recovery of every original run.

The [report tables](reports/README.md) are transcribed from the supplied project report. They support inspection of the research, but cannot replace raw predictions for checking metrics, confidence intervals, or pair-level analyses. No GPU experiments were rerun during this repository refresh.

## 1. PPI-specific LoRA adaptation

Analysis code:

- `scripts/lora/build_lora_r8_transitions.py`
- `scripts/lora/analyse_endpoint_structure.py`
- `scripts/lora/analyse_lora_r8_embedding_transfer.py`
- `scripts/lora/analyse_lora_r8_same_anchor.py`
- `scripts/lora/analyse_lora_r8_endpoint_stratification.py`

Configuration:

- `configs/lora/lora_r8_run_config.json`
- `configs/lora/lora_r8_best_val_accuracy_config.json`
- `configs/lora/lora_r8_adapter_config.json`

## 2. Standard InterPro Domain specialization

Training:

- `scripts/interpro_training/train_joint_curriculum.py`
- `scripts/interpro_training/run_joint_corrected_production.slurm`

Shared implementation:

- `src/interpro_joint/`
- `src/interpro_stage1/`
- `src/interpro_stage2/`

The specialization task combines residue-level Domain localization with masked-region Domain identification.

## 3. MLM-preserved specialization

Training:

- `scripts/interpro_training/train_joint_curriculum_mlm.py`
- `scripts/interpro_training/run_joint_mlm_production.slurm`

Additional implementation:

- `src/interpro_joint/mlm.py`
- `src/interpro_joint/mlm_engine.py`

Evaluation:

- `scripts/domain_evaluation/evaluate_esmc_mlm_retention.py`
- `scripts/domain_evaluation/evaluate_joint_end_to_end_mlm.py`

## 4. Representation-preserved specialization

Training:

- `scripts/interpro_training/train_joint_curriculum_rep_preserve.py`
- `scripts/interpro_training/run_joint_rep_preserve_alpha10_production.slurm`

Implementation:

- `src/interpro_joint/representation_regularization.py`
- `src/interpro_joint/model_rep_preserve.py`
- `src/interpro_joint/engine_rep_preserve.py`

The final reported representation-preservation weight is `alpha = 10`.

Evaluation:

- `scripts/domain_evaluation/evaluate_joint_end_to_end_rep10.py`

## 5. Downstream PPI transfer

Project-specific TUnA code:

- `src/tuna/train.py`
- `src/tuna/pl_modules/lit_ppi.py`

Frozen configurations:

- `configs/tuna/base/`
- `configs/tuna/standard/`
- `configs/tuna/mlm_preserved/`
- `configs/tuna/rep_preserved_alpha10/`

For each representation condition, checkpoint selection is performed on Intra0 validation before evaluation on the protein-disjoint Intra2 test split.

## 6. Representation geometry

Code:

- `scripts/geometry/audit_final_embedding_geometry.py`
- `scripts/geometry/audit_ppi_pair_geometry.py`
- `scripts/geometry/run_final_embedding_geometry.slurm`

These analyses compare Base ESMC with Standard, MLM-preserved, and representation-preserved encoders using nearest-neighbour preservation, pairwise cosine geometry, linear CKA, and PPI-pair geometry.

## 7. Functional-region compatibility

Domain and Family experiments are contained in:

    scripts/functional_regions/

They extract gold InterPro regions, construct Base ESMC region representations, train compatibility models, and evaluate frozen fusion with TUnA predictions.

The joint Domain + Family analysis is:

    scripts/functional_regions/evaluate_joint_domain_family_fusion.py

These experiments are separate from the restricted 64-Domain specialization task.

## External artifacts

The following are not stored in Git:

- processed InterPro datasets
- Bernett PPI datasets
- ESMC embedding exports
- model checkpoints
- LoRA adapter weights
- intermediate prediction files
- analysis outputs

Many training/evaluation scripts resolve artifacts through `INTERPRO_PROJECT_ROOT` and `TUNA_ROOT`. The historical LoRA analyses also retain relative paths such as `results/`, `internal/`, and `paper_results/`; inspect their module-level constants and provide the matching artifact layout before running them. Do not assume the two environment variables configure every script.

To make a future experimental release fully auditable, retain the preprocessing scripts and dataset manifests/hashes, original LoRA training entry point, upstream TUnA-R revision, environment lock, checkpoint hashes, and pair-level predictions. Exported results should identify their split, selection criterion, threshold convention, and annotation-eligible population.
