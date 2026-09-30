"""Strict pair alignment for inspecting binary PPI prediction changes.

This small companion utility does not replace the historical paper analyses.
It rejects incomplete comparisons instead of silently dropping unmatched pairs.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping

Pair = tuple[str, str]


@dataclass(frozen=True)
class Prediction:
    label: int
    score: float


def read_predictions(path: str | Path) -> dict[Pair, Prediction]:
    """Read protein_A, protein_B, label, score from a TSV.

    Labels must be exactly 0 or 1; scores must be finite probabilities.
    Self-pairs are allowed. Reversed pairs count as duplicates. IDs are kept
    as strings (including leading zeroes), with surrounding whitespace removed.
    """
    path = Path(path)
    predictions = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = reader.fieldnames or []
        required = {"protein_A", "protein_B", "label", "score"}
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path}: duplicate column names")
        if not required.issubset(columns):
            raise ValueError(f"{path}: missing columns {sorted(required - set(columns))}")
        for line, row in enumerate(reader, 2):
            if None in row or any(row[key] is None for key in required):
                raise ValueError(f"{path}:{line}: malformed TSV row")
            a, b = row["protein_A"].strip(), row["protein_B"].strip()
            if not a or not b:
                raise ValueError(f"{path}:{line}: empty protein ID")
            label = row["label"].strip()
            if label not in {"0", "1"}:
                raise ValueError(f"{path}:{line}: label must be 0 or 1")
            try:
                score = float(row["score"])
            except ValueError as error:
                raise ValueError(f"{path}:{line}: invalid score") from error
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{path}:{line}: score must be finite and in [0, 1]")
            pair = tuple(sorted((a, b)))
            if pair in predictions:
                raise ValueError(f"{path}:{line}: duplicate unordered pair {pair}")
            predictions[pair] = Prediction(int(label), score)
    if not predictions:
        raise ValueError(f"{path}: no predictions")
    return predictions


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")


def binary_metrics(predictions: Mapping[Pair, Prediction], threshold: float) -> dict:
    """Use score >= threshold. Undefined rates/MCC are represented as None."""
    validate_threshold(threshold)
    if not predictions:
        raise ValueError("No predictions to evaluate")
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for item in predictions.values():
        positive = item.score >= threshold
        key = ("tp" if positive else "fn") if item.label else ("fp" if positive else "tn")
        counts[key] += 1
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        **counts,
        "n_pairs": len(predictions),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(predictions),
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
    }


def compare_predictions(
    baseline: Mapping[Pair, Prediction],
    adapted: Mapping[Pair, Prediction],
    *,
    baseline_threshold: float = 0.5,
    adapted_threshold: float = 0.5,
) -> dict:
    """Compare identical pair sets and labels, independently of row/orientation order."""
    if baseline.keys() != adapted.keys():
        raise ValueError(
            "Pair sets differ: "
            f"{len(baseline.keys() - adapted.keys())} missing from adapted; "
            f"{len(adapted.keys() - baseline.keys())} missing from baseline"
        )
    base_metrics = binary_metrics(baseline, baseline_threshold)
    adapted_metrics = binary_metrics(adapted, adapted_threshold)
    transitions = dict(corrected=0, broken=0, unchanged_correct=0, unchanged_incorrect=0)
    deltas = []
    for pair in sorted(baseline):
        base, new = baseline[pair], adapted[pair]
        if base.label != new.label:
            raise ValueError(f"Labels disagree for pair {pair}")
        base_correct = (base.score >= baseline_threshold) == base.label
        new_correct = (new.score >= adapted_threshold) == new.label
        if base_correct == new_correct:
            key = "unchanged_correct" if base_correct else "unchanged_incorrect"
        else:
            key = "corrected" if new_correct else "broken"
        transitions[key] += 1
        deltas.append(new.score - base.score)
    return {
        "threshold_rule": "score >= threshold; thresholds supplied, never fitted",
        "baseline": base_metrics,
        "adapted": adapted_metrics,
        "transitions": transitions,
        "net_additional_correct": transitions["corrected"] - transitions["broken"],
        "mean_score_change": math.fsum(deltas) / len(deltas),
    }


def split_overlap(splits: Mapping[str, Mapping[Pair, Prediction]]) -> dict:
    """Check exact pair/protein identity overlap, not sequence homology or degree bias."""
    if len(splits) < 2:
        raise ValueError("Provide at least two splits")
    overlaps = {}
    for first, second in combinations(splits, 2):
        left, right = splits[first], splits[second]
        left_ids = {protein for pair in left for protein in pair}
        right_ids = {protein for pair in right for protein in pair}
        overlaps[f"{first} vs {second}"] = {
            "shared_pairs": len(left.keys() & right.keys()),
            "shared_proteins": len(left_ids & right_ids),
        }
    return {
        "scope": "Exact identifiers only; does not test sequence similarity or degree bias",
        "protein_disjoint": all(v["shared_proteins"] == 0 for v in overlaps.values()),
        "comparisons": overlaps,
    }
