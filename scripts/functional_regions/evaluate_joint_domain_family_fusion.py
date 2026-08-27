#!/usr/bin/env python3

from __future__ import annotations

import gzip
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd

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

I0_DOMAIN = (
    ROOT
    / "results/domain_compatibility_overnight/"
      "tuna_fusion_intra0/fusion_oof_pair_scores.tsv.gz"
)

I0_FAMILY = (
    ROOT
    / "results/family_compatibility_overnight/"
      "tuna_fusion_intra0/fusion_oof_pair_scores.tsv.gz"
)

I2_DOMAIN = (
    ROOT
    / "results/domain_compatibility_intra2/"
      "frozen_tuna_fusion/frozen_intra2_pair_scores.tsv.gz"
)

I2_FAMILY = (
    ROOT
    / "results/family_compatibility_intra2/"
      "frozen_tuna_fusion/frozen_intra2_pair_scores.tsv.gz"
)

OUTDIR = (
    ROOT
    / "results/domain_family_joint_fusion"
)

N_FOLDS = 5
SEED = 47
N_BOOT = 1000


def key(a, b):

    a = str(a).strip()
    b = str(b).strip()

    return (
        a + "||" + b
        if a <= b
        else b + "||" + a
    )


def make_model():

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


def metrics(y, score, method):

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    pred = (
        score >= 0.5
    ).astype(int)

    return {
        "method": method,
        "n": len(y),

        "AUROC":
            roc_auc_score(y, score),

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
    }


def load_signal(
    path,
    signal,
    split,
):

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep="\t",
    )

    required = {
        "protein_A",
        "protein_B",
        "label",
        "tuna_score",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing "
            f"{sorted(missing)}"
        )

    ensemble_candidates = [
        f"{signal}_ensemble",
    ]

    mask_candidates = [
        f"both_{signal}_annotated",
    ]

    #
    # Compatibility with a Family script
    # cloned directly from the Domain version.
    #
    if signal == "family":

        ensemble_candidates.append(
            "domain_ensemble"
        )

        mask_candidates.append(
            "both_domain_annotated"
        )

    ensemble_col = next(
        (
            c
            for c in ensemble_candidates
            if c in df.columns
        ),
        None,
    )

    mask_col = next(
        (
            c
            for c in mask_candidates
            if c in df.columns
        ),
        None,
    )

    if (
        ensemble_col is None
        or mask_col is None
    ):
        raise RuntimeError(
            f"{path}: cannot identify "
            f"{signal} columns.\n"
            f"Columns={list(df.columns)}"
        )

    columns = [
        "protein_A",
        "protein_B",
        "label",
        "tuna_score",
        ensemble_col,
        mask_col,
    ]

    if (
        split == "Intra2"
        and "frozen_fusion_score"
        in df.columns
    ):
        columns.append(
            "frozen_fusion_score"
        )

    if (
        split == "Intra2"
        and "frozen_hybrid_score"
        in df.columns
    ):
        columns.append(
            "frozen_hybrid_score"
        )

    out = df[
        columns
    ].copy()

    out["pair_key"] = [
        key(a, b)
        for a, b
        in zip(
            out["protein_A"],
            out["protein_B"],
        )
    ]

    if out[
        "pair_key"
    ].duplicated().any():

        raise RuntimeError(
            f"{path}: duplicate pairs"
        )

    rename = {
        "label":
            f"label_{signal}",

        "tuna_score":
            f"tuna_{signal}",

        ensemble_col:
            f"{signal}_ensemble",

        mask_col:
            f"{signal}_annotated",
    }

    if (
        "frozen_fusion_score"
        in out.columns
    ):
        rename[
            "frozen_fusion_score"
        ] = (
            f"{signal}_frozen_fusion"
        )

    if (
        "frozen_hybrid_score"
        in out.columns
    ):
        rename[
            "frozen_hybrid_score"
        ] = (
            f"{signal}_frozen_hybrid"
        )

    return out.rename(
        columns=rename
    )


def merge_signals(
    domain,
    family,
):

    family_columns = [
        "pair_key",
        "label_family",
        "tuna_family",
        "family_ensemble",
        "family_annotated",
    ]

    for col in [
        "family_frozen_fusion",
        "family_frozen_hybrid",
    ]:
        if col in family.columns:
            family_columns.append(col)

    merged = domain.merge(
        family[
            family_columns
        ],
        on="pair_key",
        how="inner",
        validate="one_to_one",
    )

    if (
        len(merged) != len(domain)
        or len(merged) != len(family)
    ):
        raise RuntimeError(
            "Domain and Family "
            "pair sets differ"
        )

    if not np.array_equal(
        merged[
            "label_domain"
        ].to_numpy(dtype=int),
        merged[
            "label_family"
        ].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "Labels disagree"
        )

    td = merged[
        "tuna_domain"
    ].to_numpy(dtype=float)

    tf = merged[
        "tuna_family"
    ].to_numpy(dtype=float)

    if not np.allclose(
        td,
        tf,
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            "Domain and Family "
            "TUnA scores disagree"
        )

    merged[
        "label"
    ] = merged[
        "label_domain"
    ]

    merged[
        "tuna_score"
    ] = td

    return merged


def fit_models(
    df,
    mask,
):

    y = (
        df.loc[
            mask,
            "label",
        ]
        .to_numpy(
            dtype=int
        )
    )

    tuna = (
        df.loc[
            mask,
            "tuna_score",
        ]
        .to_numpy(
            dtype=float
        )
    )

    domain = (
        df.loc[
            mask,
            "domain_ensemble",
        ]
        .to_numpy(
            dtype=float
        )
    )

    family = (
        df.loc[
            mask,
            "family_ensemble",
        ]
        .to_numpy(
            dtype=float
        )
    )

    Xd = np.column_stack(
        [
            tuna,
            domain,
        ]
    )

    Xf = np.column_stack(
        [
            tuna,
            family,
        ]
    )

    Xj = np.column_stack(
        [
            tuna,
            domain,
            family,
        ]
    )

    md = make_model()
    mf = make_model()
    mj = make_model()

    md.fit(Xd, y)
    mf.fit(Xf, y)
    mj.fit(Xj, y)

    return (
        md,
        mf,
        mj,
    )


def intra0_oof(
    df,
    mask,
):

    y = (
        df.loc[
            mask,
            "label",
        ]
        .to_numpy(dtype=int)
    )

    tuna = (
        df.loc[
            mask,
            "tuna_score",
        ]
        .to_numpy(dtype=float)
    )

    domain = (
        df.loc[
            mask,
            "domain_ensemble",
        ]
        .to_numpy(dtype=float)
    )

    family = (
        df.loc[
            mask,
            "family_ensemble",
        ]
        .to_numpy(dtype=float)
    )

    Xd = np.column_stack(
        [tuna, domain]
    )

    Xf = np.column_stack(
        [tuna, family]
    )

    Xj = np.column_stack(
        [tuna, domain, family]
    )

    od = np.full(
        len(y),
        np.nan,
    )

    of = np.full(
        len(y),
        np.nan,
    )

    oj = np.full(
        len(y),
        np.nan,
    )

    folds = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    coef_rows = []

    for fold, (
        train,
        test,
    ) in enumerate(
        folds.split(
            Xj,
            y,
        ),
        start=1,
    ):

        md = make_model()
        mf = make_model()
        mj = make_model()

        md.fit(
            Xd[train],
            y[train],
        )

        mf.fit(
            Xf[train],
            y[train],
        )

        mj.fit(
            Xj[train],
            y[train],
        )

        od[test] = (
            md.predict_proba(
                Xd[test]
            )[:, 1]
        )

        of[test] = (
            mf.predict_proba(
                Xf[test]
            )[:, 1]
        )

        oj[test] = (
            mj.predict_proba(
                Xj[test]
            )[:, 1]
        )

        cj = (
            mj.named_steps[
                "logistic"
            ].coef_[0]
        )

        coef_rows.append(
            {
                "fold":
                    fold,

                "joint_tuna_coef":
                    float(cj[0]),

                "joint_domain_coef":
                    float(cj[1]),

                "joint_family_coef":
                    float(cj[2]),
            }
        )

    rows = [
        metrics(
            y,
            tuna,
            "TUnA_both",
        ),

        metrics(
            y,
            od,
            "OOF_Domain_both",
        ),

        metrics(
            y,
            of,
            "OOF_Family_both",
        ),

        metrics(
            y,
            oj,
            "OOF_DomainFamily_both",
        ),
    ]

    return (
        rows,
        coef_rows,
    )


def bootstrap_delta(
    y,
    baseline,
    candidate,
    comparison,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    candidate = np.asarray(
        candidate,
        dtype=float,
    )

    pos = np.flatnonzero(
        y == 1
    )

    neg = np.flatnonzero(
        y == 0
    )

    rng = np.random.default_rng(
        SEED
    )

    functions = {
        "AUROC":
            lambda yy, ss:
                roc_auc_score(
                    yy,
                    ss,
                ),

        "AUPRC":
            lambda yy, ss:
                average_precision_score(
                    yy,
                    ss,
                ),

        "accuracy_at_0.5":
            lambda yy, ss:
                accuracy_score(
                    yy,
                    ss >= 0.5,
                ),

        "MCC_at_0.5":
            lambda yy, ss:
                matthews_corrcoef(
                    yy,
                    ss >= 0.5,
                ),
    }

    rows = []

    for name, fn in (
        functions.items()
    ):

        observed = (
            fn(y, candidate)
            - fn(y, baseline)
        )

        deltas = np.empty(
            N_BOOT,
            dtype=float,
        )

        for i in range(
            N_BOOT
        ):

            sample = np.concatenate(
                [
                    rng.choice(
                        pos,
                        size=len(pos),
                        replace=True,
                    ),

                    rng.choice(
                        neg,
                        size=len(neg),
                        replace=True,
                    ),
                ]
            )

            deltas[i] = (
                fn(
                    y[sample],
                    candidate[sample],
                )
                -
                fn(
                    y[sample],
                    baseline[sample],
                )
            )

        rows.append(
            {
                "comparison":
                    comparison,

                "metric":
                    name,

                "observed_delta":
                    float(observed),

                "q025":
                    float(
                        np.quantile(
                            deltas,
                            0.025,
                        )
                    ),

                "median":
                    float(
                        np.quantile(
                            deltas,
                            0.5,
                        )
                    ),

                "q975":
                    float(
                        np.quantile(
                            deltas,
                            0.975,
                        )
                    ),

                "fraction_delta_gt_0":
                    float(
                        np.mean(
                            deltas > 0
                        )
                    ),
            }
        )

    return rows


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    i0 = merge_signals(
        load_signal(
            I0_DOMAIN,
            "domain",
            "Intra0",
        ),
        load_signal(
            I0_FAMILY,
            "family",
            "Intra0",
        ),
    )

    i2 = merge_signals(
        load_signal(
            I2_DOMAIN,
            "domain",
            "Intra2",
        ),
        load_signal(
            I2_FAMILY,
            "family",
            "Intra2",
        ),
    )

    for name, df in [
        ("Intra0", i0),
        ("Intra2", i2),
    ]:

        d = (
            df[
                "domain_annotated"
            ]
            .to_numpy(
                dtype=bool
            )
        )

        f = (
            df[
                "family_annotated"
            ]
            .to_numpy(
                dtype=bool
            )
        )

        print(
            f"===== {name} AVAILABILITY ====="
        )

        print(
            "all:",
            len(df),
        )

        print(
            "both:",
            int(
                (
                    d & f
                ).sum()
            ),
        )

        print(
            "domain_only:",
            int(
                (
                    d & ~f
                ).sum()
            ),
        )

        print(
            "family_only:",
            int(
                (
                    ~d & f
                ).sum()
            ),
        )

        print(
            "neither:",
            int(
                (
                    ~d & ~f
                ).sum()
            ),
        )

        print()

    i0_both = (
        i0[
            "domain_annotated"
        ].to_numpy(dtype=bool)
        &
        i0[
            "family_annotated"
        ].to_numpy(dtype=bool)
    )

    i2_both = (
        i2[
            "domain_annotated"
        ].to_numpy(dtype=bool)
        &
        i2[
            "family_annotated"
        ].to_numpy(dtype=bool)
    )

    #
    # Intra0 OOF:
    # compare all three models on exactly
    # the same both-annotated pairs.
    #
    i0_rows, coef_rows = (
        intra0_oof(
            i0,
            i0_both,
        )
    )

    pd.DataFrame(
        i0_rows
    ).to_csv(
        OUTDIR
        / "intra0_both_oof_summary.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        coef_rows
    ).to_csv(
        OUTDIR
        / "intra0_joint_fold_coefficients.tsv",
        sep="\t",
        index=False,
    )

    #
    # Fit locked models on all Intra0
    # both-annotated pairs.
    #
    md, mf, mj = fit_models(
        i0,
        i0_both,
    )

    with (
        OUTDIR
        / "frozen_intra0_models.pkl"
    ).open(
        "wb"
    ) as handle:

        pickle.dump(
            {
                "domain":
                    md,

                "family":
                    mf,

                "joint":
                    mj,

                "trained_on":
                    (
                        "Intra0 pairs with "
                        "both Domain and "
                        "Family annotations"
                    ),
            },
            handle,
        )

    #
    # Frozen Intra2 both-annotated
    # complementarity test.
    #
    y2 = (
        i2[
            "label"
        ].to_numpy(dtype=int)
    )

    tuna2 = (
        i2[
            "tuna_score"
        ].to_numpy(dtype=float)
    )

    domain2 = (
        i2[
            "domain_ensemble"
        ].to_numpy(dtype=float)
    )

    family2 = (
        i2[
            "family_ensemble"
        ].to_numpy(dtype=float)
    )

    yb = y2[
        i2_both
    ]

    tb = tuna2[
        i2_both
    ]

    db = domain2[
        i2_both
    ]

    fb = family2[
        i2_both
    ]

    pred_domain = (
        md.predict_proba(
            np.column_stack(
                [tb, db]
            )
        )[:, 1]
    )

    pred_family = (
        mf.predict_proba(
            np.column_stack(
                [tb, fb]
            )
        )[:, 1]
    )

    pred_joint = (
        mj.predict_proba(
            np.column_stack(
                [
                    tb,
                    db,
                    fb,
                ]
            )
        )[:, 1]
    )

    both_rows = [
        metrics(
            yb,
            tb,
            "TUnA_both",
        ),

        metrics(
            yb,
            pred_domain,
            "Domain_both",
        ),

        metrics(
            yb,
            pred_family,
            "Family_both",
        ),

        metrics(
            yb,
            pred_joint,
            "DomainFamily_both",
        ),
    ]

    pd.DataFrame(
        both_rows
    ).to_csv(
        OUTDIR
        / "intra2_both_frozen_summary.tsv",
        sep="\t",
        index=False,
    )

    #
    # Availability-aware system on
    # ALL 52,048 Intra2 pairs.
    #
    dmask = (
        i2[
            "domain_annotated"
        ].to_numpy(dtype=bool)
    )

    fmask = (
        i2[
            "family_annotated"
        ].to_numpy(dtype=bool)
    )

    domain_only = (
        dmask & ~fmask
    )

    family_only = (
        ~dmask & fmask
    )

    neither = (
        ~dmask & ~fmask
    )

    combined = (
        tuna2.copy()
    )

    #
    # Both available:
    # use joint Domain+Family model.
    #
    combined[
        i2_both
    ] = pred_joint

    #
    # Only one available:
    # use its already-locked
    # Intra0-trained frozen fusion.
    #
    combined[
        domain_only
    ] = (
        i2.loc[
            domain_only,
            "domain_frozen_fusion",
        ]
        .to_numpy(dtype=float)
    )

    combined[
        family_only
    ] = (
        i2.loc[
            family_only,
            "family_frozen_fusion",
        ]
        .to_numpy(dtype=float)
    )

    combined[
        neither
    ] = tuna2[
        neither
    ]

    full_rows = [
        metrics(
            y2,
            tuna2,
            "TUnA_all",
        ),

        metrics(
            y2,
            i2[
                "domain_frozen_hybrid"
            ].to_numpy(dtype=float),
            "DomainHybrid_all",
        ),

        metrics(
            y2,
            i2[
                "family_frozen_hybrid"
            ].to_numpy(dtype=float),
            "FamilyHybrid_all",
        ),

        metrics(
            y2,
            combined,
            "DomainFamilyHybrid_all",
        ),
    ]

    pd.DataFrame(
        full_rows
    ).to_csv(
        OUTDIR
        / "intra2_full_hybrid_summary.tsv",
        sep="\t",
        index=False,
    )

    #
    # Small gains -> paired bootstrap.
    #
    boot_rows = []

    boot_rows += bootstrap_delta(
        y2,
        tuna2,
        combined,
        (
            "DomainFamilyHybrid_all "
            "- TUnA_all"
        ),
    )

    boot_rows += bootstrap_delta(
        yb,
        pred_domain,
        pred_joint,
        (
            "Joint - Domain "
            "on both-annotated"
        ),
    )

    boot_rows += bootstrap_delta(
        yb,
        pred_family,
        pred_joint,
        (
            "Joint - Family "
            "on both-annotated"
        ),
    )

    pd.DataFrame(
        boot_rows
    ).to_csv(
        OUTDIR
        / "paired_bootstrap_deltas.tsv",
        sep="\t",
        index=False,
    )

    #
    # Save final pair scores.
    #
    pair_out = i2[
        [
            "protein_A",
            "protein_B",
            "label",
            "tuna_score",
            "domain_ensemble",
            "family_ensemble",
            "domain_annotated",
            "family_annotated",
        ]
    ].copy()

    pair_out[
        "both_annotated"
    ] = i2_both.astype(int)

    pair_out[
        "joint_score"
    ] = np.nan

    pair_out.loc[
        i2_both,
        "joint_score",
    ] = pred_joint

    pair_out[
        "combined_hybrid_score"
    ] = combined

    with gzip.open(
        OUTDIR
        / "intra2_joint_pair_scores.tsv.gz",
        "wt",
        encoding="utf-8",
    ) as handle:

        pair_out.to_csv(
            handle,
            sep="\t",
            index=False,
        )

    print(
        "===== INTRA0 BOTH-ANNOTATED OOF ====="
    )

    print(
        pd.DataFrame(
            i0_rows
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "===== INTRA2 BOTH-ANNOTATED FROZEN ====="
    )

    print(
        pd.DataFrame(
            both_rows
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "===== INTRA2 FULL HYBRID ====="
    )

    print(
        pd.DataFrame(
            full_rows
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "===== BOOTSTRAP DELTAS ====="
    )

    print(
        pd.DataFrame(
            boot_rows
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "WROTE:",
        OUTDIR,
    )


if __name__ == "__main__":
    main()
