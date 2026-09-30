# Attribution and source lineage

## Project work

Kareem Ayass conducted this undergraduate research project in McGill University's COMBINE Lab under Prof. Amin Emad. The associated report is titled *Protein representation adaptation reveals constraints on partner-specific PPI prediction*.

The project investigates PPI-directed adaptation, InterPro-supervised specialization, preservation objectives, downstream transfer, endpoint-associated effects, representation geometry, and functional-region compatibility. Research modules and scripts retained here support those investigations. The repository's small `ppi_audit` utility, synthetic example, result-plotting script, and CPU tests are companion tooling for inspecting this release, not evidence that those tools produced the original results.

## Upstream work

| Dependency or resource | Role | Attribution |
|---|---|---|
| TUnA / TUnA-R | Downstream interaction architecture and external training/inference code | Ko et al., *TUnA: an uncertainty-aware transformer model for sequence-based protein-protein interaction prediction*, Briefings in Bioinformatics, 2024, bbae359 |
| ESM-2 | Pretrained protein encoder for direct PPI adaptation | Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model*, Science, 2023 |
| ESMC | Pretrained encoder for Domain specialization and transfer | EvolutionaryScale ESMC, accessed through the `esm` package |
| LoRA / PEFT | Parameter-efficient encoder adaptation | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021; Hugging Face PEFT implementation |
| Bernett benchmark | Leakage-reduced human PPI partitions | Bernett et al., *Cracking the black box of deep sequence-based protein–protein interaction prediction*, Briefings in Bioinformatics, 2024, bbae076 |
| InterPro | Domain and Family annotations | Paysan-Lafosse et al., *InterPro in 2022*, Nucleic Acids Research, 2023 |

The project-specific TUnA modifications are retained in `src/tuna/train.py`, `src/tuna/pl_modules/lit_ppi.py`, and `configs/dataset/gold-standard.yaml`. This is not a complete TUnA distribution. Its MIT notice is preserved in [THIRD_PARTY_LICENSES/TUnA-R-LICENSE](../THIRD_PARTY_LICENSES/TUnA-R-LICENSE). The exact upstream revision was not recorded in this release; compatibility must be established before reproduction.

## Historical fidelity and reuse

The original Standard-specialization result and checkpoint lineage were preserved, but some shared source files were revised before repository assembly. Where historical bytes were unavailable, the repository contains reviewed implementations of the reported method. This limitation predates the documentation refresh and is retained explicitly.

No repository-wide software license is declared. The bundled third-party MIT notice applies to the relevant upstream material; it should not be interpreted as licensing all project code, datasets, pretrained weights, or the report. Contact the author before reuse beyond permissions already granted by applicable upstream terms.
