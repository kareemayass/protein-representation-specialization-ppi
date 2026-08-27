#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd


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

OUT = Path("paper_results/lora_r8/endpoint_stratification")

LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75
N_SCORE_BINS = 10
N_BOOTSTRAPS = 2000
SEED = 47


def add_endpoint_statistics(df: pd.DataFrame):
    proteins = sorted(
        set(df["protein_A"].astype(str))
        | set(df["protein_B"].astype(str))
    )

    all_endpoints = pd.concat(
        [
            df["protein_A"].astype(str),
            df["protein_B"].astype(str),
        ],
        ignore_index=True,
    )

    positive_pairs = df[df["label"] == 1]

    positive_endpoints = pd.concat(
        [
            positive_pairs["protein_A"].astype(str),
            positive_pairs["protein_B"].astype(str),
        ],
        ignore_index=True,
    )

    total_degree = all_endpoints.value_counts()
    positive_degree = positive_endpoints.value_counts()

    protein_table = pd.DataFrame({"protein": proteins})
    protein_table["total_degree"] = (
        protein_table["protein"]
        .map(total_degree)
        .fillna(0)
        .astype(int)
    )
    protein_table["positive_degree"] = (
        protein_table["protein"]
        .map(positive_degree)
        .fillna(0)
        .astype(int)
    )
    protein_table["negative_degree"] = (
        protein_table["total_degree"]
        - protein_table["positive_degree"]
    )
    protein_table["positive_rate"] = (
        protein_table["positive_degree"]
        / protein_table["total_degree"].replace(0, np.nan)
    )

    low_cutoff = float(
        protein_table["positive_degree"].quantile(
            LOW_QUANTILE,
            interpolation="lower",
        )
    )
    high_cutoff = float(
        protein_table["positive_degree"].quantile(
            HIGH_QUANTILE,
            interpolation="higher",
        )
    )

    if low_cutoff >= high_cutoff:
        raise RuntimeError(
            f"Low/high degree cutoffs overlap: "
            f"{low_cutoff}, {high_cutoff}"
        )

    protein_table["endpoint_group"] = "middle"
    protein_table.loc[
        protein_table["positive_degree"] <= low_cutoff,
        "endpoint_group",
    ] = "low"
    protein_table.loc[
        protein_table["positive_degree"] >= high_cutoff,
        "endpoint_group",
    ] = "high"

    lookup = protein_table.set_index("protein")

    result = df.copy()

    for side in ["A", "B"]:
        protein_column = f"protein_{side}"

        result[f"positive_degree_{side}"] = (
            result[protein_column]
            .astype(str)
            .map(lookup["positive_degree"])
            .astype(int)
        )

        result[f"total_degree_{side}"] = (
            result[protein_column]
            .astype(str)
            .map(lookup["total_degree"])
            .astype(int)
        )

        result[f"endpoint_group_{side}"] = (
            result[protein_column]
            .astype(str)
            .map(lookup["endpoint_group"])
        )

    result["pair_positive_degree_sum"] = (
        result["positive_degree_A"]
        + result["positive_degree_B"]
    )

    # Broad full-distribution stratification.
    propensity_percentile = (
        result["pair_positive_degree_sum"]
        .rank(method="average", pct=True)
    )

    result["endpoint_propensity_quintile"] = (
        np.ceil(propensity_percentile * 5)
        .clip(1, 5)
        .astype(int)
        .map(lambda value: f"Q{value}")
    )

    group_a = result["endpoint_group_A"]
    group_b = result["endpoint_group_B"]

    result["extreme_pair_regime"] = "other"
    result.loc[
        (group_a == "low") & (group_b == "low"),
        "extreme_pair_regime",
    ] = "low-low"
    result.loc[
        (group_a == "high") & (group_b == "high"),
        "extreme_pair_regime",
    ] = "high-high"
    result.loc[
        (
            ((group_a == "low") & (group_b == "high"))
            | ((group_a == "high") & (group_b == "low"))
        ),
        "extreme_pair_regime",
    ] = "low-high"

    cutoffs = {
        "low_positive_degree_cutoff": low_cutoff,
        "high_positive_degree_cutoff": high_cutoff,
        "n_proteins": int(len(protein_table)),
        "n_low_proteins": int(
            (protein_table["endpoint_group"] == "low").sum()
        ),
        "n_middle_proteins": int(
            (protein_table["endpoint_group"] == "middle").sum()
        ),
        "n_high_proteins": int(
            (protein_table["endpoint_group"] == "high").sum()
        ),
    }

    return result, protein_table, cutoffs


def summarize_subset(group: pd.DataFrame):
    positive = group[group["label"] == 1]
    negative = group[group["label"] == 0]

    base_fn = group[group["baseline_outcome"] == "FN"]
    base_tn = group[group["baseline_outcome"] == "TN"]

    fn_recovered = int(
        (base_fn["transition"] == "base_fn_recovered").sum()
    )
    tn_broken = int(
        (base_tn["transition"] == "base_tn_broken").sum()
    )

    baseline_correct = (
        group["baseline_prediction"] == group["label"]
    )
    lora_correct = (
        group["lora_r8_prediction"] == group["label"]
    )

    return {
        "n": int(len(group)),
        "n_positive": int(len(positive)),
        "n_negative": int(len(negative)),
        "positive_rate": float(group["label"].mean()),

        "mean_baseline_score": float(
            group["baseline_score"].mean()
        ),
        "mean_lora_score": float(
            group["lora_r8_score"].mean()
        ),

        "mean_score_delta": float(group["score_delta"].mean()),
        "median_score_delta": float(
            group["score_delta"].median()
        ),

        "mean_delta_positive_pairs": float(
            positive["score_delta"].mean()
        ) if len(positive) else np.nan,

        "mean_delta_negative_pairs": float(
            negative["score_delta"].mean()
        ) if len(negative) else np.nan,

        "baseline_accuracy": float(baseline_correct.mean()),
        "lora_accuracy": float(lora_correct.mean()),
        "accuracy_change": float(
            lora_correct.mean() - baseline_correct.mean()
        ),

        "negative_to_positive_flips": int(
            (
                group["prediction_flip"]
                == "negative_to_positive"
            ).sum()
        ),
        "negative_to_positive_flip_rate": float(
            (
                group["prediction_flip"]
                == "negative_to_positive"
            ).mean()
        ),

        "positive_to_negative_flips": int(
            (
                group["prediction_flip"]
                == "positive_to_negative"
            ).sum()
        ),
        "positive_to_negative_flip_rate": float(
            (
                group["prediction_flip"]
                == "positive_to_negative"
            ).mean()
        ),

        "base_fn_count": int(len(base_fn)),
        "fn_recovered": fn_recovered,
        "fn_recovery_rate": (
            fn_recovered / len(base_fn)
            if len(base_fn)
            else np.nan
        ),

        "base_tn_count": int(len(base_tn)),
        "tn_broken": tn_broken,
        "tn_break_rate": (
            tn_broken / len(base_tn)
            if len(base_tn)
            else np.nan
        ),
    }


def grouped_summary(df, column, split):
    rows = []

    for value, group in df.groupby(column, sort=True):
        rows.append(
            {
                "split": split,
                "stratification": column,
                "stratum": value,
                **summarize_subset(group),
            }
        )

    return pd.DataFrame(rows)


def create_matched_extremes(df):
    extreme = df[
        df["extreme_pair_regime"].isin(
            ["low-low", "high-high"]
        )
    ].copy()

    # Baseline-score deciles are calculated across the two extreme
    # groups. Labels are matched exactly as a separate dimension.
    score_rank = extreme["baseline_score"].rank(
        method="first",
        pct=True,
    )

    extreme["baseline_score_decile"] = (
        np.ceil(score_rank * N_SCORE_BINS)
        .clip(1, N_SCORE_BINS)
        .astype(int)
    )

    extreme["match_cell"] = (
        extreme["label"].astype(str)
        + "_"
        + extreme["baseline_score_decile"].astype(str)
    )

    rng = np.random.default_rng(SEED)
    matched_parts = []

    for cell, cell_df in extreme.groupby("match_cell"):
        low = cell_df[
            cell_df["extreme_pair_regime"] == "low-low"
        ]
        high = cell_df[
            cell_df["extreme_pair_regime"] == "high-high"
        ]

        n_match = min(len(low), len(high))

        if n_match == 0:
            continue

        low_indices = rng.choice(
            low.index.to_numpy(),
            size=n_match,
            replace=False,
        )
        high_indices = rng.choice(
            high.index.to_numpy(),
            size=n_match,
            replace=False,
        )

        matched_parts.append(extreme.loc[low_indices])
        matched_parts.append(extreme.loc[high_indices])

    if not matched_parts:
        raise RuntimeError("No matched low-low/high-high cells")

    return pd.concat(matched_parts, ignore_index=True)


def matched_difference(matched):
    low = summarize_subset(
        matched[
            matched["extreme_pair_regime"] == "low-low"
        ]
    )
    high = summarize_subset(
        matched[
            matched["extreme_pair_regime"] == "high-high"
        ]
    )

    keys = [
        "mean_score_delta",
        "median_score_delta",
        "mean_delta_positive_pairs",
        "mean_delta_negative_pairs",
        "negative_to_positive_flip_rate",
        "positive_to_negative_flip_rate",
        "fn_recovery_rate",
        "tn_break_rate",
        "accuracy_change",
    ]

    differences = {
        f"high_high_minus_low_low__{key}": high[key] - low[key]
        for key in keys
    }

    return low, high, differences


def bootstrap_matched(matched):
    rng = np.random.default_rng(SEED)
    rows = []

    grouped = {
        (cell, regime): group.reset_index(drop=True)
        for (cell, regime), group in matched.groupby(
            ["match_cell", "extreme_pair_regime"]
        )
    }

    cells = sorted(matched["match_cell"].unique())

    for replicate in range(N_BOOTSTRAPS):
        sampled_parts = []

        for cell in cells:
            for regime in ["low-low", "high-high"]:
                group = grouped[(cell, regime)]
                indices = rng.integers(
                    0,
                    len(group),
                    size=len(group),
                )
                sampled_parts.append(group.iloc[indices])

        sampled = pd.concat(sampled_parts, ignore_index=True)
        _, _, differences = matched_difference(sampled)

        rows.append(
            {
                "bootstrap": replicate,
                **differences,
            }
        )

    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    quintile_summaries = []
    regime_summaries = []
    matched_summaries = []
    matched_difference_rows = []
    confidence_interval_rows = []
    cutoff_metadata = {}

    for split, path in INPUTS.items():
        df = pd.read_csv(path, sep="\t")

        stratified, protein_table, cutoffs = (
            add_endpoint_statistics(df)
        )

        cutoff_metadata[split] = cutoffs

        stratified.to_csv(
            OUT / f"endpoint_stratified_pairs_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

        protein_table.insert(0, "split", split)
        protein_table.to_csv(
            OUT / f"endpoint_protein_table_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

        quintile_summaries.append(
            grouped_summary(
                stratified,
                "endpoint_propensity_quintile",
                split,
            )
        )

        regime_summaries.append(
            grouped_summary(
                stratified,
                "extreme_pair_regime",
                split,
            )
        )

        matched = create_matched_extremes(stratified)

        matched["split"] = split
        matched.to_csv(
            OUT
            / f"matched_low_low_high_high_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

        low, high, differences = matched_difference(matched)

        matched_summaries.extend(
            [
                {
                    "split": split,
                    "regime": "low-low",
                    **low,
                },
                {
                    "split": split,
                    "regime": "high-high",
                    **high,
                },
            ]
        )

        matched_difference_rows.append(
            {
                "split": split,
                **differences,
            }
        )

        bootstrap = bootstrap_matched(matched)

        bootstrap.to_csv(
            OUT
            / f"matched_extreme_bootstrap_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

        for metric in bootstrap.columns:
            if metric == "bootstrap":
                continue

            confidence_interval_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "estimate": differences[metric],
                    "ci_lower": float(
                        bootstrap[metric].quantile(0.025)
                    ),
                    "ci_upper": float(
                        bootstrap[metric].quantile(0.975)
                    ),
                }
            )

    quintile_summary = pd.concat(
        quintile_summaries,
        ignore_index=True,
    )
    regime_summary = pd.concat(
        regime_summaries,
        ignore_index=True,
    )
    matched_summary = pd.DataFrame(matched_summaries)
    matched_differences = pd.DataFrame(
        matched_difference_rows
    )
    confidence_intervals = pd.DataFrame(
        confidence_interval_rows
    )

    quintile_summary.to_csv(
        OUT / "endpoint_propensity_quintile_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    regime_summary.to_csv(
        OUT / "endpoint_pair_regime_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    matched_summary.to_csv(
        OUT / "matched_low_low_high_high_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    matched_differences.to_csv(
        OUT / "matched_high_high_minus_low_low.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    confidence_intervals.to_csv(
        OUT / "matched_extreme_confidence_intervals.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    config = {
        "diagnostic_only": True,
        "degree_definition": (
            "Positive endpoint degree calculated from labels "
            "within each evaluated partition"
        ),
        "low_definition": (
            f"positive endpoint degree <= partition "
            f"{LOW_QUANTILE:.0%} quantile"
        ),
        "high_definition": (
            f"positive endpoint degree >= partition "
            f"{HIGH_QUANTILE:.0%} quantile"
        ),
        "matched_on": [
            "label",
            f"baseline score {N_SCORE_BINS}-quantile bin",
        ],
        "n_bootstraps": N_BOOTSTRAPS,
        "seed": SEED,
        "cutoffs": cutoff_metadata,
    }

    (OUT / "analysis_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    print("\nEndpoint-propensity quintiles:")
    print(
        quintile_summary[
            [
                "split",
                "stratum",
                "n",
                "mean_score_delta",
                "mean_delta_positive_pairs",
                "mean_delta_negative_pairs",
                "fn_recovery_rate",
                "tn_break_rate",
                "accuracy_change",
            ]
        ].to_string(index=False)
    )

    print("\nMatched low-low versus high-high:")
    print(
        matched_summary[
            [
                "split",
                "regime",
                "n",
                "positive_rate",
                "mean_score_delta",
                "mean_delta_positive_pairs",
                "mean_delta_negative_pairs",
                "fn_recovery_rate",
                "tn_break_rate",
                "accuracy_change",
            ]
        ].to_string(index=False)
    )

    print("\nHigh-high minus low-low:")
    print(matched_differences.to_string(index=False))

    print(f"\nSaved under: {OUT}")


if __name__ == "__main__":
    main()
