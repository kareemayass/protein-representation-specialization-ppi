# CPU example and prediction audit

```bash
python examples/demo.py
```

This uses only Python's standard library (Python 3.11+). No weights, external datasets, network connection, or GPU are required. The example invokes the existing research curriculum and a companion prediction audit.

## What the example shows

The two TSVs contain **invented proteins and scores**. The adapted file reverses protein order and shuffles rows deliberately. Correct alignment produces:

- Baseline: TP=2, TN=3, FP=0, FN=1.
- Adapted: TP=3, TN=2, FP=1, FN=0.
- One corrected prediction, one broken prediction, and zero net additional correct predictions.
- Accuracy remains 5/6; recall rises from 2/3 to 1; specificity falls from 1 to 2/3.

The 40-step curriculum example produces 6 localization-only steps, 8 steps in a 3:1 mixture, and 26 steps in an equal mixture: 25 localization steps and 15 identification steps overall.

## Audit your own prediction files

From the repository root:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m ppi_audit compare examples/toy_baseline.tsv examples/toy_adapted.tsv
```

Required TSV columns: `protein_A`, `protein_B`, `label`, `score`. Labels are exactly `0` or `1`; scores are finite probabilities in `[0,1]`. Protein IDs are strings. Self-interactions are allowed. Duplicate unordered pairs, inconsistent labels, invalid scores, and unmatched pair sets are rejected.

The decision rule is **score ≥ threshold**. You can supply separate `--baseline-threshold` and `--adapted-threshold` values; defaults are 0.5. Choose thresholds on validation data before using this tool on a test set. This utility never fits thresholds and does not infer the historical paper's conventions. Undefined rates and MCC are JSON `null`.

To check exact identity separation between prediction files for different splits:

```bash
python -m ppi_audit splits /path/to/train_predictions.tsv /path/to/test_predictions.tsv
```

Exit status is 0 for protein-disjoint files, 1 if proteins overlap, and 2 for invalid input. The input schema still includes labels and scores. This checks **exact protein identifiers and unordered pairs only**; it does not detect sequence homology, aliases, or degree bias. It is not a substitute for the Bernett or InterPro splitting procedures.

The demo does not compute AUPRC or perform inference. Research summary metrics live separately in [reports/](../reports/).
