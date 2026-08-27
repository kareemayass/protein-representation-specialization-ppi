# Reproducibility guide

This document maps the principal experiment branches to the code and configurations retained in this repository.

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

External locations are configured through `INTERPRO_PROJECT_ROOT` and `TUNA_ROOT`.
