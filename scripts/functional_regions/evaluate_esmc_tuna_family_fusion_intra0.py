#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "INTERPRO_PROJECT_ROOT",
        REPO_ROOT,
    )
).resolve()

TUNA = Path(
    os.environ.get(
        "TUNA_ROOT",
        REPO_ROOT / "external" / "TUnA-R",
    )
).resolve()

BASE = (
    TUNA
    / "results/esmc_base_intra0/best_val_auprc_diag"
    / "predictions.tsv"
)

DOMAIN_DIR = (
    ROOT
    / "results/family_compatibility_overnight"
)

DOMAIN_FILES = [
    DOMAIN_DIR
    / "learned_seed47_intra0_pair_scores.tsv.gz",

    DOMAIN_DIR
    / "learned_seed1047_intra0_pair_scores.tsv.gz",

    DOMAIN_DIR
    / "learned_seed2047_intra0_pair_scores.tsv.gz",
]

OUTDIR = (
    DOMAIN_DIR
    / "tuna_fusion_intra0"
)

SUMMARY = (
    OUTDIR
    / "fusion_summary.tsv"
)

PAIR_OUTPUT = (
    OUTDIR
    / "fusion_oof_pair_scores.tsv.gz"
)

FOLD_OUTPUT = (
    OUTDIR
    / "fusion_fold_coefficients.tsv"
)

FINAL_MODEL = (
    OUTDIR
    / "final_intra0_fusion_model.pkl"
)

EXPECTED = 53665
N_FOLDS = 5
SEED = 47


def canonical_pair(a, b):

    a = str(a).strip()
    b = str(b).strip()

    if a <= b:
        return a + "||" + b

    return b + "||" + a


def metrics(
    labels,
    scores,
    name,
):

    labels = np.asarray(
        labels,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    pred = (
        scores >= 0.5
    ).astype(
        int
    )

    return {
        "method":
            name,

        "n":
            len(labels),

        "AUROC":
            roc_auc_score(
                labels,
                scores,
            ),

        "AUPRC":
            average_precision_score(
                labels,
                scores,
            ),

        "accuracy_at_0.5":
            accuracy_score(
                labels,
                pred,
            ),

        "MCC_at_0.5":
            matthews_corrcoef(
                labels,
                pred,
            ),

        "mean_positive_score":
            float(
                scores[
                    labels == 1
                ].mean()
            ),

        "mean_negative_score":
            float(
                scores[
                    labels == 0
                ].mean()
            ),
    }


def load_base():

    df = pd.read_csv(
        BASE,
        sep="\t",
    )

    required = {
        "protein_A",
        "protein_B",
        "label",
        "score",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Base file missing columns: "
            f"{sorted(missing)}"
        )

    if len(df) != EXPECTED:
        raise RuntimeError(
            f"Expected {EXPECTED:,} "
            f"Base rows, got {len(df):,}"
        )

    df = df[
        [
            "protein_A",
            "protein_B",
            "label",
            "score",
        ]
    ].copy()

    df.rename(
        columns={
            "score":
                "tuna_score",
        },
        inplace=True,
    )

    df[
        "pair_key"
    ] = [
        canonical_pair(
            A,
            B,
        )
        for A, B
        in zip(
            df[
                "protein_A"
            ],
            df[
                "protein_B"
            ],
        )
    ]

    if df[
        "pair_key"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Base canonical pairs"
        )

    return df


def load_domain_file(
    path,
):

    df = pd.read_csv(
        path,
        sep="\t",
    )

    required = {
        "protein_A",
        "protein_B",
        "label",
        "both_domain_annotated",
        "learned_domain_probability",
        "seed",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing "
            f"{sorted(missing)}"
        )

    if len(df) != EXPECTED:
        raise RuntimeError(
            f"{path}: expected "
            f"{EXPECTED:,} rows; "
            f"got {len(df):,}"
        )

    seeds = sorted(
        df[
            "seed"
        ].unique()
    )

    if len(seeds) != 1:
        raise RuntimeError(
            f"{path}: expected one seed"
        )

    seed = int(
        seeds[
            0
        ]
    )

    out = df[
        [
            "protein_A",
            "protein_B",
            "label",
            "both_domain_annotated",
            "learned_domain_probability",
        ]
    ].copy()

    out[
        "pair_key"
    ] = [
        canonical_pair(
            A,
            B,
        )
        for A, B
        in zip(
            out[
                "protein_A"
            ],
            out[
                "protein_B"
            ],
        )
    ]

    out.rename(
        columns={
            "learned_domain_probability":
                f"domain_seed{seed}",

            "both_domain_annotated":
                f"annotated_seed{seed}",
        },
        inplace=True,
    )

    return (
        seed,
        out[
            [
                "pair_key",
                "label",
                f"annotated_seed{seed}",
                f"domain_seed{seed}",
            ]
        ],
    )


def make_fusion_model():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = load_base()

    merged = base.copy()

    domain_columns = []
    annotated_columns = []

    for path in DOMAIN_FILES:

        seed, domain = (
            load_domain_file(
                path
            )
        )

        if not np.array_equal(
            domain[
                "label"
            ].to_numpy(),
            domain[
                "label"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "Unexpected label error"
            )

        merged = merged.merge(
            domain.drop(
                columns=[
                    "label",
                ]
            ),
            on="pair_key",
            how="left",
            validate="one_to_one",
        )

        domain_columns.append(
            f"domain_seed{seed}"
        )

        annotated_columns.append(
            f"annotated_seed{seed}"
        )

    if len(merged) != EXPECTED:
        raise RuntimeError(
            "Fusion merge changed row count"
        )

    if merged[
        domain_columns
        + annotated_columns
    ].isna().any().any():

        raise RuntimeError(
            "Missing domain values after merge"
        )

    annotation_matrix = (
        merged[
            annotated_columns
        ]
        .to_numpy(
            dtype=int
        )
    )

    if not np.all(
        annotation_matrix
        == annotation_matrix[
            :,
            [0]
        ]
    ):
        raise RuntimeError(
            "Annotation masks differ "
            "across domain seeds"
        )

    annotated = (
        annotation_matrix[
            :,
            0
        ].astype(
            bool
        )
    )

    merged[
        "domain_ensemble"
    ] = (
        merged[
            domain_columns
        ]
        .mean(
            axis=1
        )
    )

    labels = (
        merged[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    tuna_scores = (
        merged[
            "tuna_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    domain_scores = (
        merged[
            "domain_ensemble"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print(
        "all pairs:",
        f"{len(merged):,}",
    )

    print(
        "domain-annotated:",
        f"{annotated.sum():,}",
        f"({100 * annotated.mean():.2f}%)",
    )

    rho, rho_p = spearmanr(
        tuna_scores[
            annotated
        ],
        domain_scores[
            annotated
        ],
    )

    print(
        "annotated TUnA/domain Spearman:",
        f"{rho:.6f}",
        f"p={rho_p:.6g}",
    )

    #
    # OOF fusion is fit/evaluated only among
    # domain-annotated Intra0 pairs.
    #
    ann_indices = np.flatnonzero(
        annotated
    )

    X = np.column_stack(
        [
            tuna_scores[
                annotated
            ],
            domain_scores[
                annotated
            ],
        ]
    )

    y = labels[
        annotated
    ]

    oof_probability = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            X,
            y,
        ),
        start=1,
    ):

        model = make_fusion_model()

        model.fit(
            X[
                train_idx
            ],
            y[
                train_idx
            ],
        )

        oof_probability[
            test_idx
        ] = (
            model.predict_proba(
                X[
                    test_idx
                ]
            )[
                :,
                1
            ]
        )

        logistic = (
            model.named_steps[
                "logistic"
            ]
        )

        fold_rows.append(
            {
                "fold":
                    fold,

                "train_n":
                    len(
                        train_idx
                    ),

                "heldout_n":
                    len(
                        test_idx
                    ),

                "standardized_tuna_coef":
                    float(
                        logistic.coef_[
                            0,
                            0
                        ]
                    ),

                "standardized_domain_coef":
                    float(
                        logistic.coef_[
                            0,
                            1
                        ]
                    ),

                "intercept":
                    float(
                        logistic.intercept_[
                            0
                        ]
                    ),

                "heldout_AUROC":
                    roc_auc_score(
                        y[
                            test_idx
                        ],
                        oof_probability[
                            test_idx
                        ],
                    ),

                "heldout_AUPRC":
                    average_precision_score(
                        y[
                            test_idx
                        ],
                        oof_probability[
                            test_idx
                        ],
                    ),
            }
        )

    if not np.isfinite(
        oof_probability
    ).all():
        raise RuntimeError(
            "OOF fusion has missing values"
        )

    #
    # Full hybrid:
    # fused OOF score where domains exist;
    # unchanged TUnA score elsewhere.
    #
    hybrid_scores = (
        tuna_scores.copy()
    )

    hybrid_scores[
        ann_indices
    ] = oof_probability

    summary_rows = []

    summary_rows.append(
        metrics(
            labels,
            tuna_scores,
            "TUnA_all",
        )
    )

    summary_rows.append(
        metrics(
            labels[
                annotated
            ],
            tuna_scores[
                annotated
            ],
            "TUnA_annotated",
        )
    )

    summary_rows.append(
        metrics(
            labels[
                annotated
            ],
            domain_scores[
                annotated
            ],
            "FamilyEnsemble_annotated",
        )
    )

    summary_rows.append(
        metrics(
            labels[
                annotated
            ],
            oof_probability,
            "OOF_Fusion_annotated",
        )
    )

    summary_rows.append(
        metrics(
            labels,
            hybrid_scores,
            "OOF_Hybrid_all",
        )
    )

    #
    # Error transitions at threshold 0.5
    # among annotated pairs.
    #
    tuna_pred = (
        tuna_scores[
            annotated
        ]
        >= 0.5
    )

    fusion_pred = (
        oof_probability
        >= 0.5
    )

    ann_labels = labels[
        annotated
    ]

    rescued_fn = int(
        (
            (ann_labels == 1)
            &
            (~tuna_pred)
            &
            fusion_pred
        ).sum()
    )

    lost_tp = int(
        (
            (ann_labels == 1)
            &
            tuna_pred
            &
            (~fusion_pred)
        ).sum()
    )

    corrected_fp = int(
        (
            (ann_labels == 0)
            &
            tuna_pred
            &
            (~fusion_pred)
        ).sum()
    )

    created_fp = int(
        (
            (ann_labels == 0)
            &
            (~tuna_pred)
            &
            fusion_pred
        ).sum()
    )

    print()
    print(
        "ANNOTATED ERROR TRANSITIONS @0.5"
    )

    print(
        "rescued baseline FN:",
        rescued_fn,
    )

    print(
        "lost baseline TP:",
        lost_tp,
    )

    print(
        "corrected baseline FP:",
        corrected_fp,
    )

    print(
        "created new FP:",
        created_fp,
    )

    #
    # Fit the final fusion model on all Intra0
    # annotated pairs. We do NOT evaluate this
    # fitted model on Intra0 as if held-out.
    # It is retained only for a future locked
    # Intra2 confirmation.
    #
    final_model = make_fusion_model()

    final_model.fit(
        X,
        y,
    )

    with FINAL_MODEL.open(
        "wb"
    ) as handle:

        pickle.dump(
            {
                "model":
                    final_model,

                "domain_seeds":
                    domain_columns,

                "features":
                    [
                        "Base_ESMC_TUnA_score",
                        "mean_domain_probability_3seeds",
                    ],

                "trained_on":
                    "Intra0 domain-annotated pairs",

                "purpose":
                    (
                        "locked future Intra2 "
                        "confirmation only"
                    ),
            },
            handle,
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        SUMMARY,
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        FOLD_OUTPUT,
        sep="\t",
        index=False,
    )

    merged[
        "both_domain_annotated"
    ] = annotated.astype(
        int
    )

    merged[
        "fusion_oof_score"
    ] = np.nan

    merged.loc[
        annotated,
        "fusion_oof_score",
    ] = oof_probability

    merged[
        "hybrid_oof_score"
    ] = hybrid_scores

    with gzip.open(
        PAIR_OUTPUT,
        "wt",
        encoding="utf-8",
    ) as handle:

        merged[
            [
                "protein_A",
                "protein_B",
                "label",
                "tuna_score",
                "domain_ensemble",
                "both_domain_annotated",
                "fusion_oof_score",
                "hybrid_oof_score",
            ]
        ].to_csv(
            handle,
            sep="\t",
            index=False,
        )

    print()
    print(
        "===== SUMMARY ====="
    )

    print(
        pd.DataFrame(
            summary_rows
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "===== FOLD COEFFICIENTS ====="
    )

    print(
        pd.DataFrame(
            fold_rows
        ).to_string(
            index=False
        )
    )

    print()
    print("WROTE:")
    print(SUMMARY)
    print(FOLD_OUTPUT)
    print(PAIR_OUTPUT)
    print(FINAL_MODEL)

    print()
    print(
        "ESMC_TUNA_DOMAIN_FUSION_INTRA0_COMPLETE"
    )


if __name__ == "__main__":
    main()
