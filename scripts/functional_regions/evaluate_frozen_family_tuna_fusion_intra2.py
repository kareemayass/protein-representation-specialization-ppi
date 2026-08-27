#!/usr/bin/env python3

from __future__ import annotations

import gzip
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)


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
    / "results/esmc_base_intra2/"
    "frozen_best_val_auprc_diag/"
    "predictions.tsv"
)

DOMAIN = (
    ROOT
    / "results/family_compatibility_intra2"
)

FUSION_MODEL = (
    ROOT
    / "results/"
    "family_compatibility_overnight/"
    "tuna_fusion_intra0/"
    "final_intra0_fusion_model.pkl"
)

OUTDIR = (
    ROOT
    / "results/"
    "family_compatibility_intra2/"
    "frozen_tuna_fusion"
)

SEEDS = [
    47,
    1047,
    2047,
]

EXPECTED = 52048


def key(
    a,
    b,
):

    a = str(a)
    b = str(b)

    return (
        a + "||" + b
        if a <= b
        else b + "||" + a
    )


def metrics(
    y,
    score,
    method,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    pred = (
        score >= 0.5
    ).astype(int)

    return {
        "method":
            method,

        "n":
            len(y),

        "AUROC":
            roc_auc_score(
                y,
                score,
            ),

        "AUPRC":
            average_precision_score(
                y,
                score,
            ),

        "accuracy_at_0.5":
            accuracy_score(
                y,
                pred,
            ),

        "MCC_at_0.5":
            matthews_corrcoef(
                y,
                pred,
            ),

        "mean_positive_score":
            float(
                score[
                    y == 1
                ].mean()
            ),

        "mean_negative_score":
            float(
                score[
                    y == 0
                ].mean()
            ),
    }


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = pd.read_csv(
        BASE,
        sep="\t",
    )

    if len(base) != EXPECTED:
        raise RuntimeError(
            f"Base rows={len(base):,}; "
            f"expected {EXPECTED:,}"
        )

    base["pair_key"] = [
        key(a, b)
        for a, b in zip(
            base["protein_A"],
            base["protein_B"],
        )
    ]

    base = base[
        [
            "protein_A",
            "protein_B",
            "pair_key",
            "label",
            "score",
        ]
    ].rename(
        columns={
            "score":
                "tuna_score",
        }
    )

    merged = base.copy()

    score_columns = []
    mask_columns = []

    for seed in SEEDS:

        path = (
            DOMAIN
            / (
                f"learned_seed{seed}_"
                "intra2_pair_scores.tsv.gz"
            )
        )

        df = pd.read_csv(
            path,
            sep="\t",
        )

        if len(df) != EXPECTED:
            raise RuntimeError(
                f"seed {seed}: "
                f"{len(df):,} rows"
            )

        df["pair_key"] = [
            key(a, b)
            for a, b in zip(
                df["protein_A"],
                df["protein_B"],
            )
        ]

        score_col = (
            f"domain_seed{seed}"
        )

        mask_col = (
            f"annotated_seed{seed}"
        )

        df = df.rename(
            columns={
                "learned_domain_probability":
                    score_col,

                "both_domain_annotated":
                    mask_col,
            }
        )

        merged = merged.merge(
            df[
                [
                    "pair_key",
                    "label",
                    score_col,
                    mask_col,
                ]
            ].drop(
                columns=[
                    "label",
                ]
            ),
            on="pair_key",
            how="left",
            validate="one_to_one",
        )

        score_columns.append(
            score_col
        )

        mask_columns.append(
            mask_col
        )

    if len(merged) != EXPECTED:
        raise RuntimeError(
            "Merge changed row count"
        )

    if merged[
        score_columns
        + mask_columns
    ].isna().any().any():
        raise RuntimeError(
            "Missing domain rows after merge"
        )

    masks = merged[
        mask_columns
    ].to_numpy(
        dtype=int
    )

    if not np.all(
        masks
        == masks[:, [0]]
    ):
        raise RuntimeError(
            "Domain annotation masks "
            "differ across seeds"
        )

    annotated = (
        masks[:, 0]
        .astype(bool)
    )

    merged[
        "domain_ensemble"
    ] = (
        merged[
            score_columns
        ].mean(
            axis=1
        )
    )

    with FUSION_MODEL.open(
        "rb"
    ) as handle:

        payload = pickle.load(
            handle
        )

    frozen_model = (
        payload["model"]
    )

    tuna = merged[
        "tuna_score"
    ].to_numpy(
        dtype=float
    )

    domain = merged[
        "domain_ensemble"
    ].to_numpy(
        dtype=float
    )

    y = merged[
        "label"
    ].to_numpy(
        dtype=int
    )

    X = np.column_stack(
        [
            tuna[
                annotated
            ],
            domain[
                annotated
            ],
        ]
    )

    fused = (
        frozen_model
        .predict_proba(
            X
        )[:, 1]
    )

    hybrid = tuna.copy()

    hybrid[
        annotated
    ] = fused

    rows = [
        metrics(
            y,
            tuna,
            "TUnA_all",
        ),

        metrics(
            y[
                annotated
            ],
            tuna[
                annotated
            ],
            "TUnA_annotated",
        ),

        metrics(
            y[
                annotated
            ],
            domain[
                annotated
            ],
            "FamilyEnsemble_annotated",
        ),

        metrics(
            y[
                annotated
            ],
            fused,
            "FrozenFusion_annotated",
        ),

        metrics(
            y,
            hybrid,
            "FrozenHybrid_all",
        ),
    ]

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTDIR
        / "frozen_intra2_summary.tsv",
        sep="\t",
        index=False,
    )

    merged[
        "both_domain_annotated"
    ] = annotated.astype(
        int
    )

    merged[
        "frozen_fusion_score"
    ] = np.nan

    merged.loc[
        annotated,
        "frozen_fusion_score",
    ] = fused

    merged[
        "frozen_hybrid_score"
    ] = hybrid

    with gzip.open(
        OUTDIR
        / "frozen_intra2_pair_scores.tsv.gz",
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
                "frozen_fusion_score",
                "frozen_hybrid_score",
            ]
        ].to_csv(
            handle,
            sep="\t",
            index=False,
        )

    print(
        "annotated:",
        f"{annotated.sum():,}",
        "/",
        f"{len(annotated):,}",
        (
            f"({100 * annotated.mean():.2f}%)"
        ),
    )

    print()
    print(summary.to_string(index=False))

    print()
    print(
        "FROZEN_INTRA2_CONFIRMATION_COMPLETE"
    )


if __name__ == "__main__":
    main()
