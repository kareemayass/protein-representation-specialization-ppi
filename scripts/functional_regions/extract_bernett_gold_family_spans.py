#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path


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

ENTRY_LIST = ROOT / "data/raw/entry.list"
PROTEIN2IPR = ROOT / "data/raw/protein2ipr.dat.gz"

OUTDIR = ROOT / "results/bernett_gold_family_spans"

PAIR_FILES = {
    "Intra1": (
        TUNA
        / "data/gold_standard/"
        "Intra1_interaction_1500_or_less.tsv"
    ),
    "Intra0": (
        TUNA
        / "data/gold_standard/"
        "Intra0_interaction_1500_or_less.tsv"
    ),
    "Intra2": (
        TUNA
        / "data/gold_standard/"
        "Intra2_interaction.tsv"
    ),
}

EXPECTED = {
    "Intra1": {
        "pairs": 129592,
        "proteins": 3884,
        "annotated": 3249,
        "distinct_ids": 2592,
    },
    "Intra0": {
        "pairs": 53665,
        "proteins": 3535,
        "annotated": 3080,
        "distinct_ids": 2507,
    },
    "Intra2": {
        "pairs": 52048,
        "proteins": 3022,
        "annotated": 2597,
        "distinct_ids": 2131,
    },
}

EXPECTED_RAW_ROWS = 895_489_222
EXPECTED_FAMILY_DEFINITIONS = 27_926


def read_pair_proteins(path, expected_pairs, expected_proteins):

    proteins = set()
    n_pairs = 0

    with path.open() as handle:

        reader = csv.reader(
            handle,
            delimiter="\t",
        )

        for row_number, row in enumerate(reader, 1):

            if not row:
                continue

            if len(row) < 3:
                raise RuntimeError(
                    f"{path}: malformed row {row_number}: {row}"
                )

            try:
                label = int(row[2])
            except ValueError:

                if n_pairs == 0:
                    continue

                raise

            if label not in (0, 1):
                raise RuntimeError(
                    f"{path}: bad label {label}"
                )

            proteins.add(row[0])
            proteins.add(row[1])
            n_pairs += 1

    if n_pairs != expected_pairs:
        raise RuntimeError(
            f"{path}: expected {expected_pairs:,} pairs, "
            f"observed {n_pairs:,}"
        )

    if len(proteins) != expected_proteins:
        raise RuntimeError(
            f"{path}: expected {expected_proteins:,} proteins, "
            f"observed {len(proteins):,}"
        )

    return proteins


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    family_entries = {}

    with ENTRY_LIST.open() as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        required = {
            "ENTRY_AC",
            "ENTRY_TYPE",
            "ENTRY_NAME",
        }

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:
            raise RuntimeError(
                f"entry.list missing {sorted(missing)}"
            )

        for row in reader:

            if row["ENTRY_TYPE"] == "Family":

                family_entries[
                    row["ENTRY_AC"]
                ] = row["ENTRY_NAME"]

    if len(family_entries) != EXPECTED_FAMILY_DEFINITIONS:
        raise RuntimeError(
            "Family vocabulary mismatch: "
            f"expected {EXPECTED_FAMILY_DEFINITIONS:,}, "
            f"observed {len(family_entries):,}"
        )

    split_proteins = {}

    for split_name, path in PAIR_FILES.items():

        split_proteins[
            split_name
        ] = read_pair_proteins(
            path,
            EXPECTED[split_name]["pairs"],
            EXPECTED[split_name]["proteins"],
        )

    names = list(
        split_proteins
    )

    for i, name_a in enumerate(names):

        for name_b in names[i + 1:]:

            overlap = (
                split_proteins[name_a]
                & split_proteins[name_b]
            )

            if overlap:
                raise RuntimeError(
                    f"{name_a}/{name_b} protein overlap: "
                    f"{len(overlap):,}"
                )

    protein_to_split = {}

    for split_name, proteins in split_proteins.items():

        for protein in proteins:
            protein_to_split[protein] = split_name

    spans = {
        split_name: set()
        for split_name
        in PAIR_FILES
    }

    raw_rows = 0
    target_family_rows = 0

    print(
        "Scanning protein2ipr.dat.gz once...",
        flush=True,
    )

    with gzip.open(
        PROTEIN2IPR,
        "rt",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            raw_rows += 1

            fields = (
                line.rstrip("\n")
                .split("\t")
            )

            # Actual InterPro file format:
            # protein, IPR, name, source, start, end
            if len(fields) != 6:
                raise RuntimeError(
                    f"raw row {raw_rows:,}: "
                    f"expected 6 fields, got {len(fields)}"
                )

            protein = fields[0]

            split_name = (
                protein_to_split.get(
                    protein
                )
            )

            if split_name is None:
                continue

            ipr = fields[1]

            if ipr not in family_entries:
                continue

            try:
                start = int(fields[4])
                end = int(fields[5])
            except ValueError as exc:
                raise RuntimeError(
                    f"non-integer coordinates at "
                    f"row {raw_rows:,}: {fields}"
                ) from exc

            # Structural corruption should fail.
            # Sequence-version mismatches are handled later
            # by materialize_split(), exactly as for Domain.
            if (
                start < 1
                or end < start
            ):
                raise RuntimeError(
                    f"invalid coordinates at row "
                    f"{raw_rows:,}: {fields}"
                )

            target_family_rows += 1

            spans[
                split_name
            ].add(
                (
                    protein,
                    ipr,
                    family_entries[ipr],
                    start,
                    end,
                )
            )

    if raw_rows != EXPECTED_RAW_ROWS:
        raise RuntimeError(
            "protein2ipr source changed: "
            f"expected {EXPECTED_RAW_ROWS:,} rows, "
            f"observed {raw_rows:,}"
        )

    summary_rows = []

    for split_name in PAIR_FILES:

        rows = sorted(
            spans[split_name],
            key=lambda x: (
                x[0],
                x[3],
                x[4],
                x[1],
            ),
        )

        annotated = {
            row[0]
            for row in rows
        }

        distinct_ids = {
            row[1]
            for row in rows
        }

        if (
            len(annotated)
            != EXPECTED[
                split_name
            ][
                "annotated"
            ]
        ):
            raise RuntimeError(
                f"{split_name}: annotated protein "
                "coverage disagrees with completed audit: "
                f"{len(annotated):,} vs "
                f"{EXPECTED[split_name]['annotated']:,}"
            )

        if (
            len(distinct_ids)
            != EXPECTED[
                split_name
            ][
                "distinct_ids"
            ]
        ):
            raise RuntimeError(
                f"{split_name}: Family-ID coverage "
                "disagrees with completed audit: "
                f"{len(distinct_ids):,} vs "
                f"{EXPECTED[split_name]['distinct_ids']:,}"
            )

        path = (
            OUTDIR
            / (
                f"{split_name.lower()}_"
                "gold_family_spans.tsv.gz"
            )
        )

        with gzip.open(
            path,
            "wt",
            encoding="utf-8",
        ) as handle:

            writer = csv.writer(
                handle,
                delimiter="\t",
                lineterminator="\n",
            )

            # Legacy field name intentionally retained:
            # the already-tested Domain materializer
            # expects "Domain_name".
            writer.writerow(
                [
                    "protein",
                    "InterPro_ID",
                    "Domain_name",
                    "start",
                    "end",
                ]
            )

            writer.writerows(
                rows
            )

        summary_rows.append(
            {
                "split":
                    split_name,
                "total_proteins":
                    len(
                        split_proteins[
                            split_name
                        ]
                    ),
                "proteins_with_family_span":
                    len(annotated),
                "distinct_family_ids":
                    len(distinct_ids),
                "distinct_family_spans":
                    len(rows),
            }
        )

        print(
            split_name,
            "proteins=",
            f"{len(split_proteins[split_name]):,}",
            "annotated=",
            f"{len(annotated):,}",
            "Family IDs=",
            f"{len(distinct_ids):,}",
            "spans=",
            f"{len(rows):,}",
            flush=True,
        )

    summary_path = (
        OUTDIR
        / "gold_family_span_summary.tsv"
    )

    with summary_path.open(
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

    print(
        "raw rows:",
        f"{raw_rows:,}",
    )

    print(
        "matching Family rows:",
        f"{target_family_rows:,}",
    )

    print(
        "FAMILY_GOLD_SPANS_COMPLETE"
    )


if __name__ == "__main__":
    main()
