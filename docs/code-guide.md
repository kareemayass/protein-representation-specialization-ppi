# A guided tour of the code

Start with the research question in the [README](../README.md), then follow the part you want to inspect. These are code entry points, not a claim that every experiment runs without external artifacts.

| What to inspect | Entry point | What it demonstrates |
|---|---|---|
| Distributed task scheduling | [JointCurriculum](../src/interpro_joint/curriculum.py) | Deterministic 15% / 20% / 65% phases; exact task ratios within blocks |
| Variable-length protein batching | [batching.py](../src/interpro_stage1/batching.py) | Padded-token budgeting and distributed batch handling |
| Annotation handling | [data.py](../src/interpro_joint/data.py) | FASTA/TSV parsing, coordinate checks, Domain localization and identification datasets |
| Feature preservation | [representation_regularization.py](../src/interpro_joint/representation_regularization.py) | Adapter-disabled teacher pass, selected hidden states, masked normalized feature loss |
| Joint distributed optimization | [engine_rep_preserve.py](../src/interpro_joint/engine_rep_preserve.py) | Task losses, gradient accumulation, and reduction of loss denominators |
| Production training | [train_joint_curriculum_rep_preserve.py](../scripts/interpro_training/train_joint_curriculum_rep_preserve.py) | Training orchestration; paired with a retained Slurm recipe |
| Endpoint-effect analysis | [analyse_endpoint_structure.py](../scripts/lora/analyse_endpoint_structure.py) | Sparse endpoint design, nested ridge selection, held-out score-change analysis |
| Within-anchor discrimination | [analyse_lora_r8_same_anchor.py](../scripts/lora/analyse_lora_r8_same_anchor.py) | Ranking positive and negative partners for the same protein |
| Representation-space analysis | [audit_final_embedding_geometry.py](../scripts/geometry/audit_final_embedding_geometry.py) | Neighbour preservation, sampled cosine geometry, and linear CKA |
| Partner-level biological features | [train_learned_domain_compatibility.py](../scripts/functional_regions/train_learned_domain_compatibility.py) | Symmetric region comparison and pair-level compatibility |

The research scripts retain their original organization to preserve readable links to experiment history. See [reproducibility](../REPRODUCIBILITY.md) for the dependencies and artifacts required to run them.
