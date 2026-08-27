#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import torch


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

BASE_EMBEDDINGS = (
    TUNA
    / "data/gold_standard/embeddings"
    / "esmc_300m_unadapted_embeddings.pt"
)

SPAN_DIR = (
    ROOT
    / "results/bernett_gold_domain_spans"
)

OUTDIR = (
    ROOT
    / "results/domain_compatibility_overnight"
)

OUTPUT = (
    OUTDIR
    / "base_domain_data.pt"
)

SUMMARY = (
    OUTDIR
    / "prep_summary.tsv"
)

HIDDEN = 960


def materialize_split(
    *,
    split_name: str,
    span_path: Path,
    embeddings,
):

    sums = {}
    span_counts = defaultdict(int)
    names = {}

    total_rows = 0
    valid_rows = 0
    skipped_rows = 0

    with gzip.open(
        span_path,
        "rt",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        required = {
            "protein",
            "InterPro_ID",
            "Domain_name",
            "start",
            "end",
        }

        missing = (
            required
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing:
            raise RuntimeError(
                f"{span_path}: missing columns "
                f"{sorted(missing)}"
            )

        for row in reader:

            total_rows += 1

            protein = row[
                "protein"
            ]

            ipr = row[
                "InterPro_ID"
            ]

            name = row[
                "Domain_name"
            ]

            start = int(
                row["start"]
            )

            end = int(
                row["end"]
            )

            if protein not in embeddings:
                raise RuntimeError(
                    f"{split_name}: "
                    f"missing embedding for "
                    f"{protein}"
                )

            residue = embeddings[
                protein
            ]

            if (
                residue.ndim != 2
                or residue.shape[1]
                != HIDDEN
            ):
                raise RuntimeError(
                    f"{protein}: unexpected "
                    f"embedding shape "
                    f"{tuple(residue.shape)}"
                )

            length = int(
                residue.shape[0]
            )

            if (
                start < 1
                or end < start
                or end > length
            ):
                skipped_rows += 1
                continue

            valid_rows += 1

            vector = (
                residue[
                    start - 1 : end,
                    :
                ]
                .float()
                .mean(dim=0)
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            key = (
                protein,
                ipr,
            )

            if key not in sums:
                sums[key] = (
                    np.zeros(
                        HIDDEN,
                        dtype=np.float32,
                    )
                )

            sums[key] += vector

            span_counts[
                key
            ] += 1

            names[
                key
            ] = name

    records = []

    for (
        protein,
        ipr,
    ) in sorted(
        sums
    ):

        key = (
            protein,
            ipr,
        )

        vector = (
            sums[key]
            / span_counts[key]
        )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if (
            not np.isfinite(norm)
            or norm <= 0
        ):
            raise RuntimeError(
                f"{split_name}: bad "
                f"protein-domain vector "
                f"{protein} {ipr}"
            )

        vector = (
            vector / norm
        ).astype(
            np.float32
        )

        records.append(
            (
                protein,
                ipr,
                names[key],
                span_counts[key],
                vector,
            )
        )

    if not records:
        raise RuntimeError(
            f"{split_name}: no records"
        )

    vectors = np.stack(
        [
            record[4]
            for record
            in records
        ],
        axis=0,
    )

    protein_ids = [
        record[0]
        for record
        in records
    ]

    ipr_ids = [
        record[1]
        for record
        in records
    ]

    domain_names = [
        record[2]
        for record
        in records
    ]

    per_protein_span_counts = [
        int(
            record[3]
        )
        for record
        in records
    ]

    protein_to_indices = (
        defaultdict(list)
    )

    label_to_indices = (
        defaultdict(list)
    )

    for index, (
        protein,
        ipr,
    ) in enumerate(
        zip(
            protein_ids,
            ipr_ids,
        )
    ):

        protein_to_indices[
            protein
        ].append(
            index
        )

        label_to_indices[
            ipr
        ].append(
            index
        )

    label_ids = sorted(
        label_to_indices
    )

    label_names = []
    label_vectors = []
    label_counts = []

    first_name = {}

    for ipr, name in zip(
        ipr_ids,
        domain_names,
    ):
        first_name.setdefault(
            ipr,
            name,
        )

    for ipr in label_ids:

        indices = (
            label_to_indices[
                ipr
            ]
        )

        centroid = (
            vectors[
                indices
            ]
            .mean(
                axis=0
            )
        )

        norm = float(
            np.linalg.norm(
                centroid
            )
        )

        if (
            not np.isfinite(norm)
            or norm <= 0
        ):
            raise RuntimeError(
                f"{split_name}: bad "
                f"label centroid {ipr}"
            )

        centroid = (
            centroid / norm
        ).astype(
            np.float32
        )

        label_vectors.append(
            centroid
        )

        label_names.append(
            first_name[
                ipr
            ]
        )

        label_counts.append(
            len(indices)
        )

    label_vectors = (
        np.stack(
            label_vectors,
            axis=0,
        )
    )

    max_domains = max(
        len(indices)
        for indices
        in protein_to_indices.values()
    )

    mean_domains = float(
        np.mean(
            [
                len(indices)
                for indices
                in protein_to_indices.values()
            ]
        )
    )

    print(
        f"{split_name}: "
        f"raw_spans={total_rows:,} "
        f"valid_spans={valid_rows:,} "
        f"skipped={skipped_rows:,} "
        f"protein_domain_records="
        f"{len(records):,} "
        f"proteins="
        f"{len(protein_to_indices):,} "
        f"labels={len(label_ids):,}",
        flush=True,
    )

    return {
        "split":
            split_name,

        "vectors":
            torch.from_numpy(
                vectors
            ).to(
                dtype=torch.float16
            ),

        "protein_ids":
            protein_ids,

        "ipr_ids":
            ipr_ids,

        "domain_names":
            domain_names,

        "span_counts":
            per_protein_span_counts,

        "protein_to_indices":
            {
                protein:
                    list(indices)
                for protein, indices
                in protein_to_indices.items()
            },

        "label_ids":
            label_ids,

        "label_names":
            label_names,

        "label_counts":
            label_counts,

        "label_centroids":
            torch.from_numpy(
                label_vectors
            ).to(
                dtype=torch.float32
            ),

        "summary": {
            "raw_span_rows":
                total_rows,

            "valid_span_rows":
                valid_rows,

            "skipped_span_rows":
                skipped_rows,

            "protein_domain_records":
                len(records),

            "annotated_proteins":
                len(
                    protein_to_indices
                ),

            "distinct_domain_labels":
                len(
                    label_ids
                ),

            "mean_domain_labels_per_protein":
                mean_domains,

            "max_domain_labels_per_protein":
                max_domains,
        },
    }


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Base-ESMC residue "
        "embeddings with mmap...",
        flush=True,
    )

    embeddings = torch.load(
        BASE_EMBEDDINGS,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    print(
        "Base-ESMC proteins:",
        f"{len(embeddings):,}",
        flush=True,
    )

    intra1 = materialize_split(
        split_name="Intra1",
        span_path=(
            SPAN_DIR
            / "intra1_gold_domain_spans.tsv.gz"
        ),
        embeddings=embeddings,
    )

    intra0 = materialize_split(
        split_name="Intra0",
        span_path=(
            SPAN_DIR
            / "intra0_gold_domain_spans.tsv.gz"
        ),
        embeddings=embeddings,
    )

    if (
        intra1[
            "summary"
        ][
            "skipped_span_rows"
        ]
        != 19
    ):
        raise RuntimeError(
            "Expected 19 invalid Intra1 "
            "span rows; observed "
            f"{intra1['summary']['skipped_span_rows']}"
        )

    if (
        intra0[
            "summary"
        ][
            "skipped_span_rows"
        ]
        != 13
    ):
        raise RuntimeError(
            "Expected 13 invalid Intra0 "
            "span rows; observed "
            f"{intra0['summary']['skipped_span_rows']}"
        )

    payload = {
        "embedding_source":
            str(
                BASE_EMBEDDINGS
            ),

        "hidden_size":
            HIDDEN,

        "representation":
            (
                "L2-normalized mean Base-ESMC "
                "gold-span embedding, averaged "
                "within protein x InterPro ID"
            ),

        "intra1":
            intra1,

        "intra0":
            intra0,
    }

    torch.save(
        payload,
        OUTPUT,
    )

    with SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "split",
                "raw_span_rows",
                "valid_span_rows",
                "skipped_span_rows",
                "protein_domain_records",
                "annotated_proteins",
                "distinct_domain_labels",
                "mean_domain_labels_per_protein",
                "max_domain_labels_per_protein",
            ]
        )

        for split_data in (
            intra1,
            intra0,
        ):

            summary = split_data[
                "summary"
            ]

            writer.writerow(
                [
                    split_data[
                        "split"
                    ],
                    summary[
                        "raw_span_rows"
                    ],
                    summary[
                        "valid_span_rows"
                    ],
                    summary[
                        "skipped_span_rows"
                    ],
                    summary[
                        "protein_domain_records"
                    ],
                    summary[
                        "annotated_proteins"
                    ],
                    summary[
                        "distinct_domain_labels"
                    ],
                    summary[
                        "mean_domain_labels_per_protein"
                    ],
                    summary[
                        "max_domain_labels_per_protein"
                    ],
                ]
            )

    print()
    print("WROTE:")
    print(OUTPUT)
    print(SUMMARY)
    print()
    print(
        "DOMAIN_COMPATIBILITY_PREP_COMPLETE"
    )


if __name__ == "__main__":
    main()
