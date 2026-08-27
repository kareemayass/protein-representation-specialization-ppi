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

PAIRS = (
    TUNA
    / "data/gold_standard/Intra2_interaction.tsv"
)

ENTRY_LIST = (
    ROOT
    / "data/raw/entry.list"
)

PROTEIN2IPR = (
    ROOT
    / "data/raw/protein2ipr.dat.gz"
)

OUTDIR = (
    ROOT
    / "results/bernett_gold_domain_spans"
)

OUT = (
    OUTDIR
    / "intra2_gold_domain_spans.tsv.gz"
)

SUMMARY = (
    OUTDIR
    / "intra2_gold_domain_span_summary.tsv"
)


def read_intra2_proteins():

    proteins = set()
    n_pairs = 0

    with PAIRS.open() as handle:
        reader = csv.reader(
            handle,
            delimiter="\t",
        )

        for row in reader:

            if len(row) < 3:
                raise RuntimeError(
                    f"Malformed pair row: {row}"
                )

            proteins.add(row[0])
            proteins.add(row[1])

            n_pairs += 1

    if n_pairs != 52048:
        raise RuntimeError(
            f"Expected 52,048 Intra2 pairs; "
            f"observed {n_pairs:,}"
        )

    if len(proteins) != 3022:
        raise RuntimeError(
            f"Expected 3,022 Intra2 proteins; "
            f"observed {len(proteins):,}"
        )

    return proteins


def read_domain_entries():

    domains = {}

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
                "entry.list missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            if row["ENTRY_TYPE"] != "Domain":
                continue

            domains[
                row["ENTRY_AC"]
            ] = row[
                "ENTRY_NAME"
            ]

    if len(domains) != 21357:
        raise RuntimeError(
            f"Expected 21,357 Domain definitions; "
            f"observed {len(domains):,}"
        )

    return domains


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    proteins = (
        read_intra2_proteins()
    )

    domains = (
        read_domain_entries()
    )

    print(
        "Intra2 proteins:",
        f"{len(proteins):,}",
        flush=True,
    )

    print(
        "Domain definitions:",
        f"{len(domains):,}",
        flush=True,
    )

    spans = set()

    raw_rows = 0
    target_protein_rows = 0
    target_domain_rows = 0

    print(
        "Scanning protein2ipr.dat.gz...",
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

            if len(fields) != 6:
                raise RuntimeError(
                    "Unexpected protein2ipr row "
                    f"with {len(fields)} fields "
                    f"at raw row {raw_rows:,}; expected 6"
                )

            protein = fields[0]

            if protein not in proteins:
                continue

            target_protein_rows += 1

            ipr = fields[1]

            if ipr not in domains:
                continue

            target_domain_rows += 1

            try:
                start = int(fields[4])
                end = int(fields[5])
            except ValueError as exc:
                raise RuntimeError(
                    "Non-integer coordinates for "
                    f"{protein} {ipr}: "
                    f"{fields[4]} {fields[5]}"
                ) from exc

            spans.add(
                (
                    protein,
                    ipr,
                    domains[ipr],
                    start,
                    end,
                )
            )

            if (
                raw_rows % 50_000_000
                == 0
            ):
                print(
                    "raw rows:",
                    f"{raw_rows:,}",
                    "spans:",
                    f"{len(spans):,}",
                    flush=True,
                )

    ordered = sorted(
        spans,
        key=lambda x: (
            x[0],
            x[3],
            x[4],
            x[1],
        ),
    )

    annotated = {
        row[0]
        for row in ordered
    }

    distinct_iprs = {
        row[1]
        for row in ordered
    }

    with gzip.open(
        OUT,
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
                "protein",
                "InterPro_ID",
                "Domain_name",
                "start",
                "end",
            ]
        )

        writer.writerows(
            ordered
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
                "total_proteins",
                "proteins_with_domain_span",
                "proteins_without_domain_span",
                "fraction_with_domain_span",
                "distinct_domain_ids",
                "distinct_domain_spans",
                "raw_protein2ipr_rows",
                "target_protein_rows",
                "target_domain_rows",
            ]
        )

        writer.writerow(
            [
                "Intra2",
                len(proteins),
                len(annotated),
                len(
                    proteins
                    - annotated
                ),
                (
                    len(annotated)
                    / len(proteins)
                ),
                len(distinct_iprs),
                len(ordered),
                raw_rows,
                target_protein_rows,
                target_domain_rows,
            ]
        )

    print()
    print("WROTE:", OUT)
    print("WROTE:", SUMMARY)

    print(
        "annotated proteins:",
        f"{len(annotated):,}",
    )

    print(
        "distinct domains:",
        f"{len(distinct_iprs):,}",
    )

    print(
        "distinct spans:",
        f"{len(ordered):,}",
    )


if __name__ == "__main__":
    main()
