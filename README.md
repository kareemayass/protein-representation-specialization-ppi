# Protein representation adaptation: what improves, and what transfers?

**Protein representation learning · Transfer learning · Leakage-reduced evaluation**

Research by **Kareem Ayass**, BSc Honours Computer Science and Biology, McGill University. Conducted in the **COMBINE Lab**, supervised by **Prof. Amin Emad**.

**Does adapting a protein language model improve discrimination between interaction partners, or mainly change protein-level interaction tendencies?** This project investigates that distinction through PPI-directed LoRA adaptation, endpoint-effect analysis, and biologically supervised representation learning.

The lead finding is that **protein-disjoint generalization does not, by itself, establish partner-specific learning**. Although joint LoRA-r8/TUnA adaptation slightly increased accuracy, it reduced AUPRC and within-protein partner ranking. Its protein-level score effects were predictable for unseen proteins from frozen pretrained **ESM-2** representations: a PCA–ridge model fitted on Intra0 transferred to **3,022 protein-disjoint Intra2 proteins** with **R² = 0.370** and **Spearman ρ = 0.667**.

This identifies a useful evaluation distinction: a model can generalize protein-level tendencies to new proteins without becoming better at deciding *which partners those proteins interact with*. It does not establish identity leakage or show that all protein-level signal is biologically invalid.

[Results & interpretation](docs/results.md) · [Methods](docs/methods.md) · [Code guide](docs/code-guide.md) · [Reproduce & run](REPRODUCIBILITY.md)

![PPI adaptation diagnostics: endpoint effects transfer from frozen ESM-2 embeddings to unseen proteins, while accuracy gains coexist with reduced AUPRC and within-anchor partner ranking.](docs/assets/lora-diagnostics.svg)

*Source: Table 1 and Figure 1 of the project report. The figure uses [reported metrics](reports/lora_results.csv) and [diagnostic summaries](reports/lora_diagnostics.csv). Cross-split transfer uses frozen ESM-2 features and PCA–ridge. The endpoint regression within Intra2 and the transfer to unseen proteins are separate analyses. See [provenance](reports/README.md).*

### Why this matters for evaluation

Checking that test proteins are absent from training is necessary for this benchmark's intended generalization setting, but it does not tell us which transferable features drive predictions. The report therefore combines pair-aligned changes, endpoint decomposition, embedding-based transfer, and within-anchor ranking. Together, these analyses separate **prediction changes associated with a protein** from **improved discrimination among its candidate partners**.

A complementary branch asks whether explicit Domain supervision makes representations more useful for PPI. It reveals a second constraint: accurate biological annotation can coexist with disrupted downstream transfer, while preserving intermediate representations recovers most of that loss.

## The investigation

| Question | Experiment | Finding |
|---|---|---|
| Does direct PPI supervision improve partner discrimination? | ESM-2 + LoRA-r8 inside TUnA; endpoint and same-anchor audits | Accuracy changed from **0.647 to 0.649**, but AUPRC fell from **0.682 to 0.669**. Score changes were strongly structured by protein identity. |
| Does explicit biological supervision improve transfer? | ESMC-300M specialized for Domain localization and identification | End-to-end Domain F1 reached **0.7579**, while PPI AUPRC fell from **0.6954 to 0.6422**. |
| Can transferable structure be retained? | Preserve intermediate pretrained representations during specialization | PPI AUPRC recovered to **0.6919**, with Domain F1 **0.7340**. Preserving masked-token prediction alone did not recover PPI transfer. |

These are distinct ESM-2 and ESMC experiment families; their baselines should not be conflated. Representation preservation **approached the ESMC baseline**, rather than surpassing it. [Full results, subset comparisons, and limitations →](docs/results.md)

## Complementary result: biological enrichment and transfer

![Reported Intra2 PPI AUPRC and held-out Domain F1 across ESMC encoder conditions. Representation preservation recovers PPI transfer while retaining most Domain performance.](docs/assets/transfer-summary.svg)

*Reported ESMC results from the project report, regenerated from [the transfer summary](reports/transfer_results.csv). All PPI conditions use the same 52,048-pair Intra2 partition; Domain F1 uses a separate InterPro test set. No experiments were rerun.*

## Research and implementation contributions

- **Generalization analysis:** pair-aligned score changes, endpoint-effect decomposition, transfer to unseen proteins, and ranking among alternative partners of the same anchor. [Analysis code](scripts/lora/)
- **Representation adaptation:** joint Domain localization and masked-region identification, with curriculum training and LoRA; compare MLM and intermediate-feature preservation. [Training code](scripts/interpro_training/) · [Preservation loss](src/interpro_joint/representation_regularization.py)
- **Large-scale training:** token-budget batching, distributed task scheduling, gradient accumulation, and resumable HPC runs over a processed InterPro dataset of **907,958 proteins**. [Batching](src/interpro_stage1/batching.py) · [Curriculum](src/interpro_joint/curriculum.py) · [Runtime](src/interpro_joint/runtime.py)
- **Mechanistic follow-up:** nearest-neighbour preservation, cosine geometry, linear CKA, and explicit Domain/Family compatibility with validation-fitted late fusion. [Geometry](scripts/geometry/) · [Functional regions](scripts/functional_regions/)

The project builds on **TUnA/TUnA-R, ESM-2, ESMC, PEFT, InterPro, and the Bernett benchmark**. Those models, resources, and benchmark design are upstream work. This repository retains the project's adaptations and analyses; [attribution and source lineage](docs/attribution.md) distinguish them.

## Navigate the repository

| Area | Purpose |
|---|---|
| [`src/interpro_joint/`](src/interpro_joint/) | Joint training, preservation objectives, curriculum, and runtime |
| [`src/interpro_stage1/`](src/interpro_stage1/), [`src/interpro_stage2/`](src/interpro_stage2/) | Localization, token batching, and Domain identification |
| [`src/tuna/`](src/tuna/) | Project-specific modifications; requires the external TUnA-R implementation |
| [`scripts/`](scripts/) | Training, evaluation, geometry, LoRA audits, and functional-region analyses |
| [`configs/`](configs/) | Retained LoRA settings and downstream Hydra snapshots |
| [`reports/`](reports/), [`docs/`](docs/) | Reported metrics, interpretation, methods, and source map |

## Reproducibility status

**Available now:** research implementations, retained configurations and Slurm launch arguments, report-derived summary tables, and regenerable figures.

**Needed for full experiments:** processed datasets, embedding exports, trained checkpoints, original prediction files, and a compatible TUnA-R checkout. These artifacts and the data-preparation pipeline are **not included**. The original LoRA training entry point and an exact upstream TUnA-R revision are also not retained here. Some shared Standard-specialization source files were revised before this repository was assembled; the retained method implementation is not a byte-for-byte historical snapshot.

The [reproducibility guide](REPRODUCIBILITY.md) documents the environment, input artifacts, and retained experiment entry points. Full training and evaluation require the external artifacts listed above.

## Scientific scope

The study concerns representation adaptation under a specific human PPI benchmark and downstream architecture. It motivates preserving transferable features and testing partner-specific information; it does not establish a universal PPI ceiling, a general causal explanation of transfer, or a virtual-cell model. [Limitations and next questions →](docs/results.md#limitations-and-next-questions)

For citation and reuse, see [CITATION.cff](CITATION.cff) and [attribution](docs/attribution.md).
