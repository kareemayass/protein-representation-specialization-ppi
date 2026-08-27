#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


OUT = Path("paper_results/lora_r8/endpoint_structure")

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

MODELS = ["control", "endpoint", "combined"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 2026


def build_endpoint_matrix(df: pd.DataFrame):
    proteins = pd.Index(
        sorted(
            set(df["protein_A"].astype(str))
            | set(df["protein_B"].astype(str))
        )
    )
    protein_to_index = {
        protein: index for index, protein in enumerate(proteins)
    }

    a_indices = df["protein_A"].astype(str).map(
        protein_to_index
    ).to_numpy()
    b_indices = df["protein_B"].astype(str).map(
        protein_to_index
    ).to_numpy()

    rows = np.repeat(np.arange(len(df)), 2)
    columns = np.column_stack([a_indices, b_indices]).ravel()
    values = np.ones(len(rows), dtype=float)

    matrix = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(df), len(proteins)),
    )
    matrix.sum_duplicates()

    return matrix, proteins


def make_features(
    model_name,
    endpoint_matrix,
    control_raw,
    train_indices,
    evaluation_indices,
):
    if model_name == "endpoint":
        return (
            endpoint_matrix[train_indices],
            endpoint_matrix[evaluation_indices],
        )

    mean = control_raw[train_indices].mean(axis=0)
    std = control_raw[train_indices].std(axis=0)
    std[std == 0] = 1.0

    train_control = sparse.csr_matrix(
        (control_raw[train_indices] - mean) / std
    )
    evaluation_control = sparse.csr_matrix(
        (control_raw[evaluation_indices] - mean) / std
    )

    if model_name == "control":
        return train_control, evaluation_control

    if model_name == "combined":
        return (
            sparse.hstack(
                [
                    train_control,
                    endpoint_matrix[train_indices],
                ],
                format="csr",
            ),
            sparse.hstack(
                [
                    evaluation_control,
                    endpoint_matrix[evaluation_indices],
                ],
                format="csr",
            ),
        )

    raise ValueError(model_name)


def select_alpha(
    model_name,
    endpoint_matrix,
    control_raw,
    target,
    candidate_indices,
    inner_splits,
):
    mean_scores = {}

    for alpha in ALPHAS:
        scores = []

        for relative_train, relative_validation in inner_splits:
            train_indices = candidate_indices[relative_train]
            validation_indices = candidate_indices[relative_validation]

            x_train, x_validation = make_features(
                model_name,
                endpoint_matrix,
                control_raw,
                train_indices,
                validation_indices,
            )

            model = Ridge(
                alpha=alpha,
                fit_intercept=True,
                solver="lsqr",
                max_iter=10000,
                tol=1e-6,
            )
            model.fit(x_train, target[train_indices])

            prediction = model.predict(x_validation)
            scores.append(
                r2_score(target[validation_indices], prediction)
            )

        mean_scores[alpha] = float(np.mean(scores))

    selected_alpha = max(
        ALPHAS,
        key=lambda alpha: (mean_scores[alpha], -alpha),
    )

    return selected_alpha, mean_scores


def endpoint_coverage(df, train_indices, validation_indices):
    train_rows = df.iloc[train_indices]
    validation_rows = df.iloc[validation_indices]

    seen = (
        set(train_rows["protein_A"].astype(str))
        | set(train_rows["protein_B"].astype(str))
    )

    return (
        validation_rows["protein_A"].astype(str).isin(seen).to_numpy()
        & validation_rows["protein_B"].astype(str).isin(seen).to_numpy()
    )


def evaluate_split(split, path):
    df = pd.read_csv(path, sep="\t")

    required = {
        "protein_A",
        "protein_B",
        "baseline_score",
        "lora_r8_score",
        "score_delta",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df["protein_A"] = df["protein_A"].astype(str)
    df["protein_B"] = df["protein_B"].astype(str)

    target = df["score_delta"].to_numpy(dtype=float)

    baseline_score = df["baseline_score"].to_numpy(dtype=float)
    control_raw = np.column_stack(
        [
            baseline_score,
            np.abs(baseline_score - 0.5),
        ]
    )

    endpoint_matrix, proteins = build_endpoint_matrix(df)

    outer_cv = KFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    outer_splits = list(outer_cv.split(np.arange(len(df))))

    oof_predictions = {
        model_name: np.full(len(df), np.nan)
        for model_name in MODELS
    }

    fold_rows = []

    for fold, (train_indices, validation_indices) in enumerate(
        outer_splits,
        start=1,
    ):
        inner_cv = KFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + fold,
        )
        inner_splits = list(
            inner_cv.split(np.arange(len(train_indices)))
        )

        both_seen = endpoint_coverage(
            df,
            train_indices,
            validation_indices,
        )

        for model_name in MODELS:
            selected_alpha, _ = select_alpha(
                model_name,
                endpoint_matrix,
                control_raw,
                target,
                train_indices,
                inner_splits,
            )

            x_train, x_validation = make_features(
                model_name,
                endpoint_matrix,
                control_raw,
                train_indices,
                validation_indices,
            )

            model = Ridge(
                alpha=selected_alpha,
                fit_intercept=True,
                solver="lsqr",
                max_iter=10000,
                tol=1e-6,
            )
            model.fit(x_train, target[train_indices])

            prediction = model.predict(x_validation)
            oof_predictions[model_name][validation_indices] = prediction

            if both_seen.sum() >= 2:
                seen_r2 = r2_score(
                    target[validation_indices][both_seen],
                    prediction[both_seen],
                )
                seen_mae = mean_absolute_error(
                    target[validation_indices][both_seen],
                    prediction[both_seen],
                )
            else:
                seen_r2 = np.nan
                seen_mae = np.nan

            fold_rows.append(
                {
                    "split": split,
                    "fold": fold,
                    "model": model_name,
                    "selected_alpha": selected_alpha,
                    "n_train": len(train_indices),
                    "n_validation": len(validation_indices),
                    "both_endpoints_seen_fraction": float(
                        both_seen.mean()
                    ),
                    "r2": r2_score(
                        target[validation_indices],
                        prediction,
                    ),
                    "mae": mean_absolute_error(
                        target[validation_indices],
                        prediction,
                    ),
                    "rmse": np.sqrt(
                        mean_squared_error(
                            target[validation_indices],
                            prediction,
                        )
                    ),
                    "r2_both_endpoints_seen": seen_r2,
                    "mae_both_endpoints_seen": seen_mae,
                }
            )

    for model_name in MODELS:
        if np.isnan(oof_predictions[model_name]).any():
            raise RuntimeError(
                f"Incomplete OOF predictions for {split} {model_name}"
            )

    fold_df = pd.DataFrame(fold_rows)

    summary = (
        fold_df.groupby(["split", "model"], as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            mean_r2_both_seen=(
                "r2_both_endpoints_seen",
                "mean",
            ),
            mean_both_seen_fraction=(
                "both_endpoints_seen_fraction",
                "mean",
            ),
        )
    )

    summary_indexed = summary.set_index("model")

    comparisons = pd.DataFrame(
        [
            {
                "split": split,
                "comparison": "endpoint_minus_control",
                "delta_mean_r2": (
                    summary_indexed.loc["endpoint", "mean_r2"]
                    - summary_indexed.loc["control", "mean_r2"]
                ),
            },
            {
                "split": split,
                "comparison": "combined_minus_control",
                "delta_mean_r2": (
                    summary_indexed.loc["combined", "mean_r2"]
                    - summary_indexed.loc["control", "mean_r2"]
                ),
            },
            {
                "split": split,
                "comparison": "combined_minus_endpoint",
                "delta_mean_r2": (
                    summary_indexed.loc["combined", "mean_r2"]
                    - summary_indexed.loc["endpoint", "mean_r2"]
                ),
            },
        ]
    )

    oof = df[
        [
            "protein_A",
            "protein_B",
            "label",
            "baseline_score",
            "lora_r8_score",
            "score_delta",
        ]
    ].copy()
    oof.insert(0, "split", split)

    for model_name in MODELS:
        oof[f"{model_name}_oof_prediction"] = (
            oof_predictions[model_name]
        )
        oof[f"{model_name}_oof_residual"] = (
            target - oof_predictions[model_name]
        )

    # Choose full-split penalties using CV, then save shrunk protein effects.
    all_indices = np.arange(len(df))
    full_cv = KFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED + 100,
    )
    full_splits = list(full_cv.split(all_indices))

    full_parameters = []
    protein_coefficients = {}

    for model_name in MODELS:
        selected_alpha, alpha_scores = select_alpha(
            model_name,
            endpoint_matrix,
            control_raw,
            target,
            all_indices,
            full_splits,
        )

        if model_name == "endpoint":
            x_full = endpoint_matrix
            control_mean = None
            control_std = None
        else:
            control_mean = control_raw.mean(axis=0)
            control_std = control_raw.std(axis=0)
            control_std[control_std == 0] = 1.0

            control_scaled = sparse.csr_matrix(
                (control_raw - control_mean) / control_std
            )

            if model_name == "control":
                x_full = control_scaled
            else:
                x_full = sparse.hstack(
                    [control_scaled, endpoint_matrix],
                    format="csr",
                )

        model = Ridge(
            alpha=selected_alpha,
            fit_intercept=True,
            solver="lsqr",
            max_iter=10000,
            tol=1e-6,
        )
        model.fit(x_full, target)

        row = {
            "split": split,
            "model": model_name,
            "selected_alpha": selected_alpha,
            "intercept": float(model.intercept_),
            "full_fit_r2": float(model.score(x_full, target)),
            "baseline_score_coefficient": np.nan,
            "boundary_distance_coefficient": np.nan,
        }

        if model_name == "control":
            row["baseline_score_coefficient"] = float(
                model.coef_[0]
            )
            row["boundary_distance_coefficient"] = float(
                model.coef_[1]
            )

        elif model_name == "endpoint":
            protein_coefficients["endpoint_alpha"] = model.coef_.copy()

        elif model_name == "combined":
            row["baseline_score_coefficient"] = float(
                model.coef_[0]
            )
            row["boundary_distance_coefficient"] = float(
                model.coef_[1]
            )
            protein_coefficients["combined_alpha"] = (
                model.coef_[2:].copy()
            )

        row["alpha_cv_scores"] = json.dumps(alpha_scores)
        full_parameters.append(row)

    occurrences = pd.concat(
        [df["protein_A"], df["protein_B"]],
        ignore_index=True,
    ).value_counts()

    effects = pd.DataFrame({"protein": proteins})
    effects.insert(0, "split", split)
    effects["pair_occurrences"] = (
        effects["protein"].map(occurrences).astype(int)
    )
    effects["endpoint_alpha"] = protein_coefficients[
        "endpoint_alpha"
    ]
    effects["combined_alpha"] = protein_coefficients[
        "combined_alpha"
    ]

    return (
        fold_df,
        summary,
        comparisons,
        oof,
        pd.DataFrame(full_parameters),
        effects,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    all_fold = []
    all_summary = []
    all_comparisons = []
    all_parameters = []

    for split, path in INPUTS.items():
        (
            fold_df,
            summary,
            comparisons,
            oof,
            parameters,
            effects,
        ) = evaluate_split(split, path)

        all_fold.append(fold_df)
        all_summary.append(summary)
        all_comparisons.append(comparisons)
        all_parameters.append(parameters)

        oof.to_csv(
            OUT / f"ridge_oof_predictions_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

        effects.to_csv(
            OUT / f"protein_effects_{split.lower()}.tsv",
            sep="\t",
            index=False,
            float_format="%.12f",
        )

    fold_results = pd.concat(all_fold, ignore_index=True)
    summaries = pd.concat(all_summary, ignore_index=True)
    comparisons = pd.concat(all_comparisons, ignore_index=True)
    parameters = pd.concat(all_parameters, ignore_index=True)

    fold_results.to_csv(
        OUT / "ridge_cv_fold_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )
    summaries.to_csv(
        OUT / "ridge_cv_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )
    comparisons.to_csv(
        OUT / "ridge_cv_model_comparisons.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )
    parameters.to_csv(
        OUT / "ridge_full_fit_parameters.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    config = {
        "target": "lora_r8_score - baseline_score",
        "models": {
            "control": [
                "baseline_score",
                "abs(baseline_score - 0.5)",
            ],
            "endpoint": "unordered additive protein identities",
            "combined": (
                "control features plus additive protein identities"
            ),
        },
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "alphas": ALPHAS,
        "seed": SEED,
        "interpretation": (
            "Pair-held-out endpoint decomposition within each split; "
            "not unseen-protein generalisation."
        ),
    }
    (OUT / "analysis_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    print("\nCross-validated summary:")
    print(
        summaries[
            [
                "split",
                "model",
                "mean_r2",
                "sd_r2",
                "mean_mae",
                "mean_r2_both_seen",
                "mean_both_seen_fraction",
            ]
        ].to_string(index=False)
    )

    print("\nR² comparisons:")
    print(comparisons.to_string(index=False))

    print(f"\nSaved under: {OUT}")


if __name__ == "__main__":
    main()
