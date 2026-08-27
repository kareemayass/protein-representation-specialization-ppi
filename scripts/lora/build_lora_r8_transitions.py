#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("paper_results/lora_r8/transitions")

BASELINE_PATHS = {
    "Intra0": Path(
        ""
        "internal/experiments/tuna_vs_pplm_fusion/predictions/"
        "tuna_intra0/pretrained_tuna_intra0_predictions_full.tsv"
    ),
    "Intra2": Path(
        "results/pretrained_tuna_intra2/"
        "pretrained_tuna_intra2_predictions_full.tsv"
    ),
}

R8_ROOT = Path(
    "results/esm2_t30_lora_tuna/"
    "joint_bce_rank8_scale2_r8_a16_"
    "lora1e4_tuna1e5_8100_h10000_fullval/"
    "inference_val_threshold__test_inference/"
    "best_by_val_accuracy/pair_trained"
)

R8_DIRS = {
    "Intra0": R8_ROOT / "intra0_inference_mode",
    "Intra2": R8_ROOT / "intra2_inference_mode",
}

THRESHOLDS = {
    "baseline": 0.50,
    "lora_r8": 0.51,
}

EXPECTED = {
    "Intra0": {
        "baseline": {"tn": 16490, "fp": 10282, "fn": 10612, "tp": 16281},
        "lora_r8": {"tn": 16112, "fp": 10660, "fn": 9424, "tp": 17469},
    },
    "Intra2": {
        "baseline": {"tn": 17079, "fp": 8945, "fn": 9423, "tp": 16601},
        "lora_r8": {"tn": 15038, "fp": 10986, "fn": 7287, "tp": 18737},
    },
}


def find_column(columns, aliases):
    lower = {str(column).lower(): column for column in columns}

    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]

    raise ValueError(
        f"Could not find any of {aliases}; columns were {list(columns)}"
    )


def has_prediction_columns(path: Path) -> bool:
    try:
        columns = pd.read_csv(path, sep="\t", nrows=0).columns
        find_column(columns, ["protein_A", "protein_a"])
        find_column(columns, ["protein_B", "protein_b"])
        find_column(columns, ["label"])
        find_column(columns, ["score"])
        return True
    except Exception:
        return False


def find_prediction_file(directory: Path) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    candidates = [
        path
        for path in directory.rglob("*.tsv")
        if has_prediction_columns(path)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No prediction TSV found under {directory}"
        )

    # The full prediction file should be the largest matching TSV.
    selected = max(candidates, key=lambda path: path.stat().st_size)

    print(f"Selected prediction file: {selected}")
    return selected


def load_predictions(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")

    protein_a = find_column(frame.columns, ["protein_A", "protein_a"])
    protein_b = find_column(frame.columns, ["protein_B", "protein_b"])
    label = find_column(frame.columns, ["label"])
    score = find_column(frame.columns, ["score"])

    result = pd.DataFrame(
        {
            "protein_A": frame[protein_a].astype(str),
            "protein_B": frame[protein_b].astype(str),
            f"{prefix}_label": pd.to_numeric(
                frame[label], errors="raise"
            ).astype(int),
            f"{prefix}_score": pd.to_numeric(
                frame[score], errors="raise"
            ).astype(float),
        }
    )

    result["pair_key"] = result.apply(
        lambda row: "||".join(
            sorted([row["protein_A"], row["protein_B"]])
        ),
        axis=1,
    )

    if result["pair_key"].duplicated().any():
        duplicates = result.loc[
            result["pair_key"].duplicated(False), "pair_key"
        ].head(10)

        raise ValueError(
            f"{path} contains duplicate unordered pairs:\n"
            + "\n".join(duplicates)
        )

    return result


def confusion(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    return {
        "tn": int(((labels == 0) & (predictions == 0)).sum()),
        "fp": int(((labels == 0) & (predictions == 1)).sum()),
        "fn": int(((labels == 1) & (predictions == 0)).sum()),
        "tp": int(((labels == 1) & (predictions == 1)).sum()),
    }


def choose_predictions(labels, scores, threshold, expected):
    candidates = {
        ">": (scores > threshold).astype(int),
        ">=": (scores >= threshold).astype(int),
    }

    matches = [
        (operator, predictions)
        for operator, predictions in candidates.items()
        if confusion(labels, predictions) == expected
    ]

    if not matches:
        observed = {
            operator: confusion(labels, predictions)
            for operator, predictions in candidates.items()
        }

        raise ValueError(
            f"No threshold convention reproduced expected counts.\n"
            f"Expected: {expected}\nObserved: {observed}"
        )

    return matches[0]


def outcome(label, prediction):
    if label == 1 and prediction == 1:
        return "TP"
    if label == 0 and prediction == 0:
        return "TN"
    if label == 0 and prediction == 1:
        return "FP"
    return "FN"


def transition_name(base_outcome, lora_outcome):
    names = {
        ("FP", "TN"): "base_fp_fixed",
        ("FN", "TP"): "base_fn_recovered",
        ("TP", "FN"): "base_tp_broken",
        ("TN", "FP"): "base_tn_broken",
        ("TP", "TP"): "tp_unchanged",
        ("TN", "TN"): "tn_unchanged",
        ("FP", "FP"): "fp_unchanged",
        ("FN", "FN"): "fn_unchanged",
    }

    return names.get(
        (base_outcome, lora_outcome),
        f"{base_outcome.lower()}_to_{lora_outcome.lower()}",
    )


def build_split(split):
    baseline_path = BASELINE_PATHS[split]
    lora_path = find_prediction_file(R8_DIRS[split])

    baseline = load_predictions(baseline_path, "baseline")
    lora = load_predictions(lora_path, "lora_r8")

    aligned = baseline.merge(
        lora[
            [
                "pair_key",
                "lora_r8_label",
                "lora_r8_score",
            ]
        ],
        on="pair_key",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    missing = aligned["_merge"].value_counts().to_dict()

    if missing.get("left_only", 0) or missing.get("right_only", 0):
        raise ValueError(
            f"{split}: pair sets do not match exactly: {missing}"
        )

    aligned = aligned.drop(columns="_merge")

    if not (
        aligned["baseline_label"] == aligned["lora_r8_label"]
    ).all():
        raise ValueError(f"{split}: label mismatch after pair alignment")

    aligned = aligned.rename(
        columns={"baseline_label": "label"}
    ).drop(columns="lora_r8_label")

    base_operator, base_predictions = choose_predictions(
        aligned["label"].to_numpy(),
        aligned["baseline_score"].to_numpy(),
        THRESHOLDS["baseline"],
        EXPECTED[split]["baseline"],
    )

    lora_operator, lora_predictions = choose_predictions(
        aligned["label"].to_numpy(),
        aligned["lora_r8_score"].to_numpy(),
        THRESHOLDS["lora_r8"],
        EXPECTED[split]["lora_r8"],
    )

    aligned["baseline_prediction"] = base_predictions
    aligned["lora_r8_prediction"] = lora_predictions

    aligned["baseline_outcome"] = [
        outcome(label, prediction)
        for label, prediction in zip(
            aligned["label"],
            aligned["baseline_prediction"],
        )
    ]

    aligned["lora_r8_outcome"] = [
        outcome(label, prediction)
        for label, prediction in zip(
            aligned["label"],
            aligned["lora_r8_prediction"],
        )
    ]

    aligned["transition"] = [
        transition_name(base, lora)
        for base, lora in zip(
            aligned["baseline_outcome"],
            aligned["lora_r8_outcome"],
        )
    ]

    aligned["score_delta"] = (
        aligned["lora_r8_score"] - aligned["baseline_score"]
    )
    aligned["absolute_score_delta"] = aligned["score_delta"].abs()

    aligned["prediction_flip"] = np.select(
        [
            (aligned["baseline_prediction"] == 1)
            & (aligned["lora_r8_prediction"] == 0),
            (aligned["baseline_prediction"] == 0)
            & (aligned["lora_r8_prediction"] == 1),
            (aligned["baseline_prediction"] == 1)
            & (aligned["lora_r8_prediction"] == 1),
        ],
        [
            "positive_to_negative",
            "negative_to_positive",
            "unchanged_positive",
        ],
        default="unchanged_negative",
    )

    transition_counts = aligned["transition"].value_counts()

    base_correct = int(
        (aligned["baseline_prediction"] == aligned["label"]).sum()
    )
    lora_correct = int(
        (aligned["lora_r8_prediction"] == aligned["label"]).sum()
    )

    summary = {
        "split": split,
        "n": int(len(aligned)),
        "baseline_threshold": THRESHOLDS["baseline"],
        "baseline_operator": base_operator,
        "lora_r8_threshold": THRESHOLDS["lora_r8"],
        "lora_r8_operator": lora_operator,
        "baseline_correct": base_correct,
        "lora_r8_correct": lora_correct,
        "base_fp_fixed": int(
            transition_counts.get("base_fp_fixed", 0)
        ),
        "base_fn_recovered": int(
            transition_counts.get("base_fn_recovered", 0)
        ),
        "base_tp_broken": int(
            transition_counts.get("base_tp_broken", 0)
        ),
        "base_tn_broken": int(
            transition_counts.get("base_tn_broken", 0)
        ),
        "tp_unchanged": int(
            transition_counts.get("tp_unchanged", 0)
        ),
        "tn_unchanged": int(
            transition_counts.get("tn_unchanged", 0)
        ),
        "fp_unchanged": int(
            transition_counts.get("fp_unchanged", 0)
        ),
        "fn_unchanged": int(
            transition_counts.get("fn_unchanged", 0)
        ),
        "total_predictions_changed": int(
            (
                aligned["baseline_prediction"]
                != aligned["lora_r8_prediction"]
            ).sum()
        ),
        "positive_to_negative_flips": int(
            (aligned["prediction_flip"] == "positive_to_negative").sum()
        ),
        "negative_to_positive_flips": int(
            (aligned["prediction_flip"] == "negative_to_positive").sum()
        ),
        "net_correct_change": int(lora_correct - base_correct),
        "mean_score_delta": float(aligned["score_delta"].mean()),
        "median_score_delta": float(aligned["score_delta"].median()),
        "mean_absolute_score_delta": float(
            aligned["absolute_score_delta"].mean()
        ),
    }

    aligned.insert(0, "split", split)

    aligned.to_csv(
        OUT
        / f"pretrained_vs_lora_r8_pair_aligned_{split.lower()}.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    long_counts = (
        aligned.groupby(
            ["baseline_outcome", "lora_r8_outcome"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    long_counts.insert(0, "split", split)

    return summary, long_counts, baseline_path, lora_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summaries = []
    transition_tables = []
    provenance = {}

    for split in ["Intra0", "Intra2"]:
        summary, transitions, baseline_path, lora_path = build_split(
            split
        )

        summaries.append(summary)
        transition_tables.append(transitions)

        provenance[split] = {
            "baseline_predictions": str(baseline_path),
            "lora_r8_predictions": str(lora_path),
        }

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUT / "pretrained_to_lora_r8_transition_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    pd.concat(
        transition_tables,
        ignore_index=True,
    ).to_csv(
        OUT / "pretrained_to_lora_r8_outcome_transition_counts.tsv",
        sep="\t",
        index=False,
    )

    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "baseline_threshold": THRESHOLDS["baseline"],
                "lora_r8_threshold": THRESHOLDS["lora_r8"],
                "lora_r8_run": (
                    "joint_bce_rank8_scale2_r8_a16_"
                    "lora1e4_tuna1e5_8100_h10000_fullval"
                ),
                "checkpoint": "best_by_val_accuracy",
                "head": "pair_trained",
                "sources": provenance,
            },
            indent=2,
        )
        + "\n"
    )

    display_columns = [
        "split",
        "base_fp_fixed",
        "base_fn_recovered",
        "base_tp_broken",
        "base_tn_broken",
        "tp_unchanged",
        "tn_unchanged",
        "fp_unchanged",
        "fn_unchanged",
        "total_predictions_changed",
        "positive_to_negative_flips",
        "negative_to_positive_flips",
        "net_correct_change",
    ]

    print("\nTransition summary:")
    print(summary_df[display_columns].to_string(index=False))
    print(f"\nSaved under: {OUT}")


if __name__ == "__main__":
    main()
