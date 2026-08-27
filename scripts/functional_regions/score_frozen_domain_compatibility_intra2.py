#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
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

PREP_SCRIPT = (
    REPO_ROOT
    / "scripts/functional_regions/"
    "prepare_domain_compatibility_data.py"
)

LEARNED_SCRIPT = (
    REPO_ROOT
    / "scripts/functional_regions/"
    "train_learned_domain_compatibility.py"
)

SPANS = (
    ROOT
    / "results/bernett_gold_domain_spans/"
    "intra2_gold_domain_spans.tsv.gz"
)

EMBEDDINGS = (
    TUNA
    / "data/gold_standard/embeddings/"
    "esmc_300m_unadapted_embeddings.pt"
)

PAIRS = (
    TUNA
    / "data/gold_standard/"
    "Intra2_interaction.tsv"
)

OUTDIR = (
    ROOT
    / "results/domain_compatibility_intra2"
)

PREP_OUT = (
    OUTDIR
    / "base_domain_data_intra2.pt"
)

SUMMARY = (
    OUTDIR
    / "frozen_domain_intra2_summary.tsv"
)

SEEDS = [
    47,
    1047,
    2047,
]


def load_module(
    name,
    path,
):

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Frozen domain scoring "
            "requires a GPU compute node"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prep_module = load_module(
        "domain_prep_original",
        PREP_SCRIPT,
    )

    learned = load_module(
        "domain_learned_original",
        LEARNED_SCRIPT,
    )

    print(
        "Loading Base-ESMC embeddings "
        "with mmap...",
        flush=True,
    )

    embeddings = torch.load(
        EMBEDDINGS,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    intra2 = (
        prep_module.materialize_split(
            split_name="Intra2",
            span_path=SPANS,
            embeddings=embeddings,
        )
    )

    torch.save(
        {
            "embedding_source":
                str(EMBEDDINGS),

            "hidden_size":
                960,

            "representation":
                (
                    "L2-normalized mean Base-ESMC "
                    "gold-span embedding, averaged "
                    "within protein x InterPro ID"
                ),

            "intra2":
                intra2,
        },
        PREP_OUT,
    )

    print(
        "Intra2 materialization summary:",
        intra2["summary"],
        flush=True,
    )

    device = torch.device(
        "cuda"
    )

    vectors, mapping = (
        learned.prepare_split(
            intra2,
            device,
        )
    )

    raw_pairs = (
        learned.read_pairs(
            PAIRS
        )
    )

    if len(raw_pairs) != 52048:
        raise RuntimeError(
            f"Expected 52,048 pairs; "
            f"observed {len(raw_pairs):,}"
        )

    annotated = (
        learned.annotated_pairs(
            raw_pairs,
            mapping,
        )
    )

    print(
        "both-domain-annotated pairs:",
        f"{len(annotated):,}",
        f"({100 * len(annotated) / len(raw_pairs):.2f}%)",
        flush=True,
    )

    all_labels = np.asarray(
        [
            label
            for _, _, label
            in raw_pairs
        ],
        dtype=np.int64,
    )

    summary_rows = []

    for seed in SEEDS:

        checkpoint_path = (
            ROOT
            / "results/"
            "domain_compatibility_overnight"
            / f"learned_seed{seed}_best.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model = (
            learned.CompatibilityMLP()
            .to(device)
        )

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True,
        )

        result = (
            learned.evaluate_pairs(
                model=model,
                vectors=vectors,
                mapping=mapping,
                pairs=annotated,
                batch_size=(
                    learned.EVAL_BATCH_SIZE
                ),
                device=device,
            )
        )

        probabilities = np.full(
            len(raw_pairs),
            0.5,
            dtype=np.float64,
        )

        mask = np.zeros(
            len(raw_pairs),
            dtype=bool,
        )

        for pair_index, probability in zip(
            result["pair_indices"],
            result["probabilities"],
        ):

            probabilities[
                int(pair_index)
            ] = float(
                probability
            )

            mask[
                int(pair_index)
            ] = True

        score_path = (
            OUTDIR
            / (
                f"learned_seed{seed}_"
                "intra2_pair_scores.tsv.gz"
            )
        )

        with gzip.open(
            score_path,
            "wt",
            encoding="utf-8",
        ) as handle:

            writer = csv.writer(
                handle,
                delimiter="\t",
                lineterminator="\n",
            )

            writer.writerow(
                [
                    "pair_index",
                    "protein_A",
                    "protein_B",
                    "label",
                    "both_domain_annotated",
                    "learned_domain_probability",
                    "seed",
                    "best_epoch",
                ]
            )

            for pair_index, (
                protein_A,
                protein_B,
                label,
            ) in enumerate(
                raw_pairs
            ):

                writer.writerow(
                    [
                        pair_index,
                        protein_A,
                        protein_B,
                        label,
                        int(
                            mask[
                                pair_index
                            ]
                        ),
                        float(
                            probabilities[
                                pair_index
                            ]
                        ),
                        seed,
                        checkpoint[
                            "best_epoch"
                        ],
                    ]
                )

        ann_labels = (
            all_labels[
                mask
            ]
        )

        ann_probs = (
            probabilities[
                mask
            ]
        )

        row = {
            "seed":
                seed,

            "best_epoch":
                checkpoint[
                    "best_epoch"
                ],

            "annotated_n":
                int(mask.sum()),

            "annotated_fraction":
                float(mask.mean()),

            "annotated_AUROC":
                roc_auc_score(
                    ann_labels,
                    ann_probs,
                ),

            "annotated_AUPRC":
                average_precision_score(
                    ann_labels,
                    ann_probs,
                ),
        }

        summary_rows.append(
            row
        )

        print(
            "SEED",
            seed,
            row,
            flush=True,
        )

    with SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summary_rows[0].keys()
            ),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("WROTE:", SUMMARY)
    print(
        "FROZEN_DOMAIN_INTRA2_COMPLETE"
    )


if __name__ == "__main__":
    main()
