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
| Lightweight release checks | [ppi_audit](../src/ppi_audit/predictions.py), [tests](../tests/) | Strict unordered-pair alignment and a CPU-only example; companion tooling added for this repository |

The research scripts retain their original organization to preserve readable links to experiment history. The lightweight audit is a separate companion; it does not silently replace the historical analysis functions or their threshold conventions.

## Reading the tests

CPU checks exercise scientific invariants: pairs are unordered, comparisons must contain identical labels and pair sets, overlapping endpoint identities invalidate a protein-disjoint claim, and every simulated rank must choose the same curriculum task at the same optimizer step. Slurm checks ensure the external artifact directory does not hide repository modules.

These checks do not run ESMC, distributed GPU optimization, TUnA inference, or the complete historical analyses. See [reproducibility](../REPRODUCIBILITY.md) for those dependencies.
