# Reported results and provenance

These CSVs are **manual transcriptions of reported aggregate results**, not raw predictions or outputs of experiments rerun from this checkout.

Source: Kareem Ayass, *Protein representation adaptation reveals constraints on partner-specific PPI prediction*, supplied COMP 402 project report (10 pages). No journal publication, DOI, or peer-review status is asserted. Page numbers below are the report's printed page numbers. The full report is not bundled in this repository.

| File | Source | Evaluation population |
|---|---|---|
| `lora_results.csv` | Table 1, p. 5 | Full Intra2: 52,048 pairs; ESM-2/TUnA |
| `transfer_results.csv` | Results, p. 7 and opening continuation on p. 9; Figure 2, p. 8 | PPI: full Intra2; Domain metrics: held-out InterPro test partition; geometry: the report's held-out representation audit |
| `region_results.csv` | Table 2, p. 9 | Annotation-eligible Intra2 subsets; each base/fusion comparison uses the same subset |

Blank cells mean **not transcribed/not reported in these source passages**, never zero. The base geometry values of 1 are self-comparison references. PPI and Domain metrics have different evaluation populations and tasks; do not compare them as a single score.

The PPI transfer table uses four decimal places as reported in the prose. LoRA values use the rounded precision of Table 1. No uncertainty intervals or multi-seed statistics have been inferred. Do not reconstruct precision–recall curves from these aggregate values.

To regenerate the README figure, install Matplotlib in a separate plotting environment and run:

```bash
python scripts/reporting/plot_reported_results.py
```

The figure is a visualization of this transcription. Recomputing metrics requires the original labels, scores, split definitions, thresholds, and checkpoint lineage described in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
