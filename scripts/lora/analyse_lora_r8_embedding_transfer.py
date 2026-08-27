#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


TRAIN_EFFECTS = Path(
    "paper_results/lora_r8/endpoint_structure/"
    "protein_effects_intra0.tsv"
)

TEST_EFFECTS = Path(
    "paper_results/lora_r8/endpoint_structure/"
    "protein_effects_intra2.tsv"
)

EMBEDDINGS = Path(
    "data/gold_standard/embeddings/"
    "gold_standard_original_embeddings.pt"
)

OUT = Path("paper_results/lora_r8/embedding_transfer")

TARGET = "combined_alpha"
SEED = 47
N_FOLDS = 5
N_BOOTSTRAPS = 2000
N_PERMUTATIONS = 200

PCA_COMPONENTS = [10, 20, 50, 100, 200]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def tensor_from_cache_value(value):
    if torch.is_tensor(value):
        return value

    if isinstance(value, dict):
        for key in [
            "embedding",
            "embeddings",
            "repr",
            "representations",
        ]:
            if key in value and torch.is_tensor(value[key]):
                return value[key]

    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item

    raise TypeError(
        f"Cannot parse embedding cache value type: {type(value)}"
    )


def load_mean_embeddings(proteins):
    print(f"Loading embedding cache: {EMBEDDINGS}", flush=True)

    try:
        cache = torch.load(
            EMBEDDINGS,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (TypeError, RuntimeError):
        cache = torch.load(
            EMBEDDINGS,
            map_location="cpu",
            weights_only=False,
        )

    vectors = {}
    missing = []

    for protein in sorted(proteins):
        if protein not in cache:
            missing.append(protein)
            continue

        tensor = tensor_from_cache_value(cache[protein]).float()

        if tensor.ndim == 1:
            vector = tensor
        else:
            vector = tensor.mean(dim=0)

        if vector.ndim != 1:
            raise ValueError(
                f"{protein}: expected one-dimensional mean embedding, "
                f"obtained shape {tuple(vector.shape)}"
            )

        vectors[protein] = vector.numpy().astype(np.float32)

    print(
        f"Embeddings loaded: {len(vectors)}; missing: {len(missing)}",
        flush=True,
    )

    return vectors, missing


def load_effects(path):
    frame = pd.read_csv(path, sep="\t")

    required = {
        "protein",
        "pair_occurrences",
        TARGET,
    }
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    frame["protein"] = frame["protein"].astype(str)

    if frame["protein"].duplicated().any():
        raise ValueError(f"Duplicate proteins in {path}")

    return frame


def metrics(y_true, y_prediction):
    rho = spearmanr(y_true, y_prediction).statistic

    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_prediction)),
        "spearman": float(rho),
        "mae": float(
            mean_absolute_error(y_true, y_prediction)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_true, y_prediction)
            )
        ),
    }


def bootstrap_test_metrics(y_true, y_prediction):
    rng = np.random.default_rng(SEED)
    rows = []
    n = len(y_true)

    for bootstrap in range(N_BOOTSTRAPS):
        indices = rng.integers(0, n, size=n)

        sampled_true = y_true[indices]
        sampled_prediction = y_prediction[indices]

        if np.std(sampled_true) == 0:
            continue

        rows.append(
            {
                "bootstrap": bootstrap,
                "r2": r2_score(
                    sampled_true,
                    sampled_prediction,
                ),
                "spearman": spearmanr(
                    sampled_true,
                    sampled_prediction,
                ).statistic,
            }
        )

    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = load_effects(TRAIN_EFFECTS)
    test = load_effects(TEST_EFFECTS)

    overlap = set(train["protein"]) & set(test["protein"])
    if overlap:
        raise ValueError(
            f"Intra0 and Intra2 are not protein-disjoint: "
            f"{len(overlap)} overlapping proteins"
        )

    needed = set(train["protein"]) | set(test["protein"])
    vectors, missing = load_mean_embeddings(needed)

    if missing:
        pd.DataFrame({"protein": missing}).to_csv(
            OUT / "missing_embeddings.tsv",
            sep="\t",
            index=False,
        )

    train = train[train["protein"].isin(vectors)].copy()
    test = test[test["protein"].isin(vectors)].copy()

    train = train.sort_values("protein").reset_index(drop=True)
    test = test.sort_values("protein").reset_index(drop=True)

    X_train = np.stack(
        [vectors[p] for p in train["protein"]]
    )
    X_test = np.stack(
        [vectors[p] for p in test["protein"]]
    )

    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    print(
        f"Intra0 proteins: {len(train)}; "
        f"Intra2 proteins: {len(test)}; "
        f"embedding dimension: {X_train.shape[1]}",
        flush=True,
    )

    cv = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    pipeline = Pipeline(
        [
            ("normalize", Normalizer(norm="l2")),
            (
                "pca",
                PCA(
                    random_state=SEED,
                    svd_solver="randomized",
                ),
            ),
            (
                "ridge",
                Ridge(
                    fit_intercept=True,
                    solver="lsqr",
                    max_iter=10000,
                    tol=1e-6,
                ),
            ),
        ]
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "pca__n_components": PCA_COMPONENTS,
            "ridge__alpha": RIDGE_ALPHAS,
        },
        scoring="r2",
        cv=cv,
        n_jobs=4,
        refit=True,
        return_train_score=False,
    )

    print("Selecting hyperparameters using Intra0 only...", flush=True)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Best parameters:", grid.best_params_, flush=True)
    print(
        "Best mean Intra0 CV R2:",
        grid.best_score_,
        flush=True,
    )

    # Descriptive Intra0 out-of-fold predictions using the selected
    # preprocessing and ridge settings.
    train_oof = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv,
        n_jobs=4,
        method="predict",
    )

    # Fit on all Intra0 proteins, then transfer unchanged to Intra2.
    best_model.fit(X_train, y_train)
    test_prediction = best_model.predict(X_test)

    train_metrics = metrics(y_train, train_oof)
    test_metrics = metrics(y_test, test_prediction)

    print("\nIntra0 OOF metrics:")
    print(json.dumps(train_metrics, indent=2))

    print("\nIntra2 transfer metrics:")
    print(json.dumps(test_metrics, indent=2))

    train_output = train.copy()
    train_output["predicted_combined_alpha_oof"] = train_oof
    train_output["residual"] = y_train - train_oof

    test_output = test.copy()
    test_output["predicted_combined_alpha"] = test_prediction
    test_output["residual"] = y_test - test_prediction

    train_output.to_csv(
        OUT / "embedding_alpha_predictions_intra0_oof.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    test_output.to_csv(
        OUT / "embedding_alpha_predictions_intra2.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results[
        [
            "param_pca__n_components",
            "param_ridge__alpha",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].sort_values("rank_test_score").to_csv(
        OUT / "intra0_hyperparameter_search.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    print(
        f"\nBootstrapping Intra2 proteins "
        f"({N_BOOTSTRAPS} replicates)...",
        flush=True,
    )

    bootstrap = bootstrap_test_metrics(
        y_test,
        test_prediction,
    )
    bootstrap.to_csv(
        OUT / "intra2_protein_bootstrap.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    bootstrap_summary = {
        "r2_ci_lower": float(
            bootstrap["r2"].quantile(0.025)
        ),
        "r2_ci_upper": float(
            bootstrap["r2"].quantile(0.975)
        ),
        "spearman_ci_lower": float(
            bootstrap["spearman"].quantile(0.025)
        ),
        "spearman_ci_upper": float(
            bootstrap["spearman"].quantile(0.975)
        ),
    }

    # Permute only the Intra0 targets. The embedding preprocessing,
    # PCA dimensionality and ridge penalty remain locked.
    normalizer = best_model.named_steps["normalize"]
    pca = best_model.named_steps["pca"]
    ridge_alpha = best_model.named_steps["ridge"].alpha

    X_train_pc = pca.transform(
        normalizer.transform(X_train)
    )
    X_test_pc = pca.transform(
        normalizer.transform(X_test)
    )

    rng = np.random.default_rng(SEED)
    permutation_rows = []

    print(
        f"Running {N_PERMUTATIONS} training-label permutations...",
        flush=True,
    )

    for permutation in range(N_PERMUTATIONS):
        permuted_target = rng.permutation(y_train)

        null_model = Ridge(
            alpha=ridge_alpha,
            fit_intercept=True,
            solver="lsqr",
            max_iter=10000,
            tol=1e-6,
        )
        null_model.fit(X_train_pc, permuted_target)

        null_prediction = null_model.predict(X_test_pc)

        permutation_rows.append(
            {
                "permutation": permutation,
                "r2": r2_score(y_test, null_prediction),
                "spearman": spearmanr(
                    y_test,
                    null_prediction,
                ).statistic,
            }
        )

    permutation = pd.DataFrame(permutation_rows)
    permutation.to_csv(
        OUT / "intra2_permutation_baseline.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    permutation_summary = {
        "r2_null_mean": float(permutation["r2"].mean()),
        "r2_null_sd": float(permutation["r2"].std()),
        "r2_permutation_p": float(
            (
                1
                + (
                    permutation["r2"]
                    >= test_metrics["r2"]
                ).sum()
            )
            / (N_PERMUTATIONS + 1)
        ),
        "spearman_null_mean": float(
            permutation["spearman"].mean()
        ),
        "spearman_null_sd": float(
            permutation["spearman"].std()
        ),
        "spearman_permutation_p": float(
            (
                1
                + (
                    permutation["spearman"]
                    >= test_metrics["spearman"]
                ).sum()
            )
            / (N_PERMUTATIONS + 1)
        ),
    }

    summary = {
        "target": TARGET,
        "embedding_file": str(EMBEDDINGS),
        "representation": (
            "mean-pooled frozen pretrained ESM-2 embedding, "
            "followed by L2 normalization and train-fitted PCA"
        ),
        "train_partition": "Intra0 proteins",
        "test_partition": "protein-disjoint Intra2 proteins",
        "n_train_proteins": len(train),
        "n_test_proteins": len(test),
        "n_missing_embeddings": len(missing),
        "best_parameters": grid.best_params_,
        "best_intra0_grid_cv_r2": float(grid.best_score_),
        "intra0_oof": train_metrics,
        "intra2_transfer": test_metrics,
        "intra2_bootstrap": bootstrap_summary,
        "intra2_permutation": permutation_summary,
    }

    (OUT / "embedding_transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    summary_rows = [
        {
            "evaluation": "Intra0 OOF",
            **train_metrics,
        },
        {
            "evaluation": "Intra2 transfer",
            **test_metrics,
        },
    ]

    pd.DataFrame(summary_rows).to_csv(
        OUT / "embedding_transfer_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.12f",
    )

    print("\nBootstrap confidence intervals:")
    print(json.dumps(bootstrap_summary, indent=2))

    print("\nPermutation baseline:")
    print(json.dumps(permutation_summary, indent=2))

    print(f"\nSaved under: {OUT}")


if __name__ == "__main__":
    main()
