# Reported results and provenance

These CSVs are **manual transcriptions of reported aggregate results**, not raw predictions or outputs of experiments rerun from this checkout.

Source: Kareem Ayass, *Protein representation adaptation reveals constraints on partner-specific PPI prediction*, supplied COMP 402 project report (10 pages). No journal publication, DOI, or peer-review status is asserted. Page numbers below are the report's printed page numbers. The full report is not bundled in this repository.

| File | Source | Evaluation population |
|---|---|---|
| `lora_results.csv` | Table 1, p. 5 | Full Intra2: 52,048 pairs; ESM-2/TUnA |
| `lora_diagnostics.csv` | Figure 1A/C/D/E, p. 6 and Methods, p. 4 | Pair transitions and within-Intra2 diagnostics; separate embedding-transfer evaluation on 3,022 unseen Intra2 proteins |
| `transfer_results.csv` | Results, p. 7 and opening continuation on p. 9; Figure 2, p. 8 | PPI: full Intra2; Domain metrics: held-out InterPro test partition; geometry: the report's held-out representation audit |
| `region_results.csv` | Table 2, p. 9 | Annotation-eligible Intra2 subsets; each base/fusion comparison uses the same subset |

Blank cells mean **not transcribed/not reported in these source passages**, never zero. The base geometry values of 1 are self-comparison references. PPI and Domain metrics have different evaluation populations and tasks; do not compare them as a single score.

The PPI transfer table uses four decimal places as reported in the prose. LoRA values use the rounded precision of Table 1. No uncertainty intervals or multi-seed statistics have been inferred. Do not reconstruct precision–recall curves from these aggregate values.

To regenerate both README figures, install Matplotlib in a separate plotting environment and run:

```bash
python scripts/reporting/plot_lora_diagnostics.py
python scripts/reporting/plot_reported_results.py
```

The figures visualize these transcriptions. The LoRA transition totals are derived by adding reported confusion transitions: corrected = 2,694 + 588 = 3,282; newly wrong = 2,629 + 558 = 3,187; net = 95. The cross-split endpoint-effect regression uses ESM-2 and PCA–ridge, not ESMC or nearest-neighbour prediction. No score-density curves or observed/predicted scatter points are reconstructed without raw data. Recomputing metrics requires the original labels, scores, split definitions, thresholds, and checkpoint lineage described in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
