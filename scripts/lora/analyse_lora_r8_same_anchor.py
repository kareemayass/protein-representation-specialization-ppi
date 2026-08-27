#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


INPUTS = {
    "Intra0": Path(
        "paper_results/lora_r8/transitions/"
        "pretrained_vs_lora_r8_pair_aligned_intra0.tsv"
    ),
    "Intra2": Path(
        "paper_results/lora_r8/transitions/"
        "pretrained_vs_lora_r8_pair_aligned_intra2.tsv"
    ),
}

OUT = Path("paper_results/lora_r8/same_anchor")
N_BOOTSTRAPS = 2000
SEED = 47


def expand_anchors(df):
    required = {
        "protein_A",
        "protein_B",
        "label",
        "baseline_score",
        "lora_r8_score",
        "score_delta",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Self-pairs cannot provide two distinct anchor-partner orientations.
    df = df[df["protein_A"] != df["protein_B"]].copy()

    left = df[
        [
            "protein_A",
            "protein_B",
            "label",
            "baseline_score",
            "lora_r8_score",
            "score_delta",
        ]
    ].rename(
        columns={
            "protein_A": "anchor",
            "protein_B": "partner",
        }
    )

    right = df[
        [
            "protein_B",
            "protein_A",
            "label",
            "baseline_score",
            "lora_r8_score",
            "score_delta",
        ]
    ].rename(
        columns={
            "protein_B": "anchor",
            "protein_A": "partner",
        }
    )

    return pd.concat([left, right], ignore_index=True)


def build_anchor_table(expanded):
    rows = []

    for anchor, group in expanded.groupby("anchor", sort=True):
        positive = group[group["label"] == 1]
        negative = group[group["label"] == 0]

        # Same-anchor discrimination requires both classes.
        if len(positive) == 0 or len(negative) == 0:
            continue

        base_positive = positive["baseline_score"].mean()
        base_negative = negative["baseline_score"].mean()

        lora_positive = positive["lora_r8_score"].mean()
        lora_negative = negative["lora_r8_score"].mean()

        delta_positive = positive["score_delta"].mean()
        delta_negative = negative["score_delta"].mean()

        base_gap = base_positive - base_negative
        lora_gap = lora_positive - lora_negative
        delta_delta = delta_positive - delta_negative

        # These should be algebraically identical.
        if not np.isclose(
            delta_delta,
            lora_gap - base_gap,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Contrast identity failed for anchor {anchor}"
            )

        labels = group["label"].to_numpy(dtype=int)

        rows.append(
            {
                "anchor": anchor,
                "n_pairs": len(group),
                "n_positive_partners": len(positive),
                "n_negative_partners": len(negative),
                "mean_anchor_delta": group["score_delta"].mean(),
                "within_anchor_delta_variance": group[
                    "score_delta"
                ].var(ddof=1),
                "mean_positive_delta": delta_positive,
                "mean_negative_delta": delta_negative,
                "delta_delta": delta_delta,
                "base_positive_negative_gap": base_gap,
                "lora_positive_negative_gap": lora_gap,
                "gap_change": lora_gap - base_gap,
                "baseline_within_anchor_auc": roc_auc_score(
                    labels,
                    group["baseline_score"],
                ),
                "lora_within_anchor_auc": roc_auc_score(
                    labels,
                    group["lora_r8_score"],
                ),
                "delta_within_anchor_auc": roc_auc_score(
                    labels,
                    group["score_delta"],
                ),
            }
        )

    result = pd.DataFrame(rows)

    result["lora_minus_base_auc"] = (
        result["lora_within_anchor_auc"]
        - result["baseline_within_anchor_auc"]
    )

    return result


def calculate_metrics(anchor_table):
    delta_delta = anchor_table["delta_delta"]
    nonzero = delta_delta != 0

    positive_count = int((delta_delta[nonzero] > 0).sum())
    nonzero_count = int(nonzero.sum())

    between_variance = anchor_table[
        "mean_anchor_delta"
    ].var(ddof=1)

    mean_within_variance = anchor_table[
        "within_anchor_delta_variance"
    ].mean()

    variance_share = (
        between_variance
        / (between_variance + mean_within_variance)
        if between_variance + mean_within_variance > 0
        else np.nan
    )

    return {
        "n_eligible_anchors": len(anchor_table),
        "median_delta_delta": delta_delta.median(),
        "mean_delta_delta": delta_delta.mean(),
        "proportion_delta_delta_positive": (
            (delta_delta > 0).mean()
        ),
        "sign_test_p": (
            binomtest(
                positive_count,
                nonzero_count,
                p=0.5,
                alternative="two-sided",
            ).pvalue
            if nonzero_count
            else np.nan
        ),
        "mean_baseline_within_anchor_auc": anchor_table[
            "baseline_within_anchor_auc"
        ].mean(),
        "mean_lora_within_anchor_auc": anchor_table[
            "lora_within_anchor_auc"
        ].mean(),
        "mean_delta_within_anchor_auc": anchor_table[
            "delta_within_anchor_auc"
        ].mean(),
        "mean_lora_minus_base_auc": anchor_table[
            "lora_minus_base_auc"
        ].mean(),
        "median_baseline_gap": anchor_table[
            "base_positive_negative_gap"
        ].median(),
        "median_lora_gap": anchor_table[
            "lora_positive_negative_gap"
        ].median(),
        "between_anchor_variance_of_mean_delta": between_variance,
        "mean_within_anchor_delta_variance": mean_within_variance,
        "between_variance_share": variance_share,
    }


def bootstrap(anchor_table):
    rng = np.random.default_rng(SEED)
    n = len(anchor_table)
    rows = []

    for replicate in range(N_BOOTSTRAPS):
        sampled = anchor_table.iloc[
            rng.integers(0, n, size=n)
        ]

        between_variance = sampled[
            "mean_anchor_delta"
        ].var(ddof=1)

        within_variance = sampled[
            "within_anchor_delta_variance"
        ].mean()

        rows.append(
            {
                "bootstrap": replicate,
                "median_delta_delta": sampled[
                    "delta_delta"
                ].median(),
                "proportion_delta_delta_positive": (
                    sampled["delta_delta"] > 0
                ).mean(),
                "mean_baseline_within_anchor_auc": sampled[
                    "baseline_within_anchor_auc"
                ].mean(),
                "mean_lora_within_anchor_auc": sampled[
                    "lora_within_anchor_auc"
                ].mean(),
                "mean_delta_within_anchor_auc": sampled[
                    "delta_within_anchor_auc"
                ].mean(),
                "mean_lora_minus_base_auc": sampled[
                    "lora_minus_base_auc"
                ].mean(),
                "between_variance_share": (
                    between_variance
                    / (between_variance + within_variance)
                    if between_variance + within_variance > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    ci_rows = []

    for split, path in INPUTS.items():
        pair_data = pd.read_csv(path, sep="\t")
        expanded = expand_anchors(pair_data)
        anchor_table = build_anchor_table(expanded)

        if anchor_table.empty:
            raise RuntimeError(
                f"No eligible same-anchor contrasts for {split}"
            )

        metrics = calculate_metrics(anchor_table)
        summary_rows.append({"split": split, **metrics})

        bootstrap_table = bootstrap(anchor_table)

        for metric in bootstrap_table.columns:
            if metric == "bootstrap":
                continue

            ci_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "estimate": metrics[metric],
                    "ci_lower": bootstrap_table[
                        metric
                    ].quantile(0.025),
                    "ci_upper": bootstrap_table[
                        metric
                    ].quantile(0.975),
                }
            )

        anchor_table.insert(0, "split", split)

        anchor_table.to_csv(
            OUT / f"same_anchor_results_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

        bootstrap_table.insert(0, "split", split)
        bootstrap_table.to_csv(
            OUT / f"same_anchor_bootstrap_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

    summary = pd.DataFrame(summary_rows)
    confidence_intervals = pd.DataFrame(ci_rows)

    summary.to_csv(
        OUT / "same_anchor_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    confidence_intervals.to_csv(
        OUT / "same_anchor_bootstrap_confidence_intervals.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    display = [
        "split",
        "n_eligible_anchors",
        "median_delta_delta",
        "proportion_delta_delta_positive",
        "mean_baseline_within_anchor_auc",
        "mean_lora_within_anchor_auc",
        "mean_delta_within_anchor_auc",
        "mean_lora_minus_base_auc",
        "between_variance_share",
    ]

    print("\nSame-anchor summary:")
    print(summary[display].to_string(index=False))

    print("\nBootstrap confidence intervals:")
    print(confidence_intervals.to_string(index=False))

    print(f"\nSaved under: {OUT}")


if __name__ == "__main__":
    main()
