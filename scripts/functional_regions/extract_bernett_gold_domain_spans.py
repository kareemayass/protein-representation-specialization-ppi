#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
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

INTRA1_PAIRS = (
    TUNA
    / "data/gold_standard/"
    "Intra1_interaction_1500_or_less.tsv"
)

INTRA0_PAIRS = (
    TUNA
    / "data/gold_standard/"
    "Intra0_interaction_1500_or_less.tsv"
)

OUTDIR = (
    ROOT
    / "results/bernett_gold_domain_spans"
)

INTRA1_OUT = (
    OUTDIR
    / "intra1_gold_domain_spans.tsv.gz"
)

INTRA0_OUT = (
    OUTDIR
    / "intra0_gold_domain_spans.tsv.gz"
)

SUMMARY_OUT = (
    OUTDIR
    / "gold_domain_span_summary.tsv"
)


# ============================================================
# Helpers
# ============================================================

def load_pair_proteins(path):
    proteins = set()
    n_pairs = 0

    with path.open() as handle:
        reader = csv.reader(
            handle,
            delimiter="\t",
        )

        for row_no, row in enumerate(reader, 1):

            if len(row) != 3:
                raise RuntimeError(
                    f"{path.name} row {row_no}: "
                    f"expected 3 columns, got {len(row)}"
                )

            a, b, label = row

            if label not in {"0", "1"}:
                raise RuntimeError(
                    f"{path.name} row {row_no}: "
                    f"bad label {label!r}"
                )

            proteins.add(a)
            proteins.add(b)
            n_pairs += 1

    return proteins, n_pairs


# ============================================================
# 1. Load InterPro Domain definitions
# ============================================================

domain_names = {}

with ENTRY_LIST.open() as handle:
    reader = csv.reader(
        handle,
        delimiter="\t",
    )

    for row_no, row in enumerate(reader, 1):

        if len(row) < 3:
            continue

        ipr = row[0]
        entry_type = row[1]
        name = row[2]

        if entry_type == "Domain":
            domain_names[ipr] = name


print(
    f"InterPro Domain definitions: "
    f"{len(domain_names):,}",
    flush=True,
)


# ============================================================
# 2. Bernett proteins
# ============================================================

intra1_proteins, intra1_n_pairs = load_pair_proteins(
    INTRA1_PAIRS
)

intra0_proteins, intra0_n_pairs = load_pair_proteins(
    INTRA0_PAIRS
)

all_proteins = (
    intra1_proteins
    | intra0_proteins
)

overlap = (
    intra1_proteins
    & intra0_proteins
)


print()
print("===== BERNNETT PROTEINS =====")
print(
    f"Intra1 pairs: {intra1_n_pairs:,}"
)
print(
    f"Intra1 proteins: "
    f"{len(intra1_proteins):,}"
)
print(
    f"Intra0 pairs: {intra0_n_pairs:,}"
)
print(
    f"Intra0 proteins: "
    f"{len(intra0_proteins):,}"
)
print(
    f"Intra1/Intra0 protein overlap: "
    f"{len(overlap):,}"
)
print(
    f"Union proteins to search: "
    f"{len(all_proteins):,}"
)


# ============================================================
# 3. Stream protein2ipr once
#
# Raw columns:
# 0 protein accession
# 1 InterPro ID
# 2 InterPro name
# 3 member database signature
# 4 start
# 5 end
#
# Coordinates are 1-based inclusive.
# ============================================================

spans = {
    "Intra1": set(),
    "Intra0": set(),
}

rows_scanned = 0
matching_domain_rows = 0


with gzip.open(
    PROTEIN2IPR,
    "rt",
    encoding="utf-8",
    newline="",
) as handle:

    reader = csv.reader(
        handle,
        delimiter="\t",
    )

    for row_no, row in enumerate(reader, 1):

        rows_scanned += 1

        if len(row) != 6:
            raise RuntimeError(
                f"protein2ipr row {row_no}: "
                f"expected 6 columns, got {len(row)}"
            )

        protein = row[0]

        # Cheap rejection before anything else.
        if protein not in all_proteins:
            continue

        ipr = row[1]

        if ipr not in domain_names:
            continue

        try:
            start = int(row[4])
            end = int(row[5])
        except ValueError as exc:
            raise RuntimeError(
                f"Bad coordinates at row {row_no}: "
                f"{row[4]!r}, {row[5]!r}"
            ) from exc

        if start < 1:
            raise RuntimeError(
                f"Invalid start at row {row_no}: "
                f"{start}"
            )

        if end < start:
            raise RuntimeError(
                f"Invalid span at row {row_no}: "
                f"{start}-{end}"
            )

        key = (
            protein,
            ipr,
            start,
            end,
        )

        matching_domain_rows += 1

        if protein in intra1_proteins:
            spans["Intra1"].add(key)

        if protein in intra0_proteins:
            spans["Intra0"].add(key)


print()
print(
    f"protein2ipr rows scanned: "
    f"{rows_scanned:,}"
)

print(
    f"Matching raw Domain rows before dedup: "
    f"{matching_domain_rows:,}"
)


# ============================================================
# 4. Write span files
# ============================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


def write_spans(path, rows):

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

        writer.writerow(
            [
                "protein",
                "InterPro_ID",
                "Domain_name",
                "start",
                "end",
                "length",
            ]
        )

        for protein, ipr, start, end in sorted(
            rows,
            key=lambda x: (
                x[0],
                x[2],
                x[3],
                x[1],
            ),
        ):

            writer.writerow(
                [
                    protein,
                    ipr,
                    domain_names[ipr],
                    start,
                    end,
                    end - start + 1,
                ]
            )


write_spans(
    INTRA1_OUT,
    spans["Intra1"],
)

write_spans(
    INTRA0_OUT,
    spans["Intra0"],
)


# ============================================================
# 5. Summary
# ============================================================

summary = []

for split_name, proteins in (
    ("Intra1", intra1_proteins),
    ("Intra0", intra0_proteins),
):

    split_spans = spans[split_name]

    annotated_proteins = {
        row[0]
        for row in split_spans
    }

    distinct_iprs = {
        row[1]
        for row in split_spans
    }

    span_lengths = [
        row[3] - row[2] + 1
        for row in split_spans
    ]

    summary.append(
        {
            "split":
                split_name,

            "total_proteins":
                len(proteins),

            "proteins_with_domain_span":
                len(annotated_proteins),

            "proteins_without_domain_span":
                len(
                    proteins
                    - annotated_proteins
                ),

            "fraction_with_domain_span":
                (
                    len(annotated_proteins)
                    / len(proteins)
                    if proteins
                    else 0.0
                ),

            "distinct_domain_ids":
                len(distinct_iprs),

            "distinct_domain_spans":
                len(split_spans),

            "mean_span_length":
                (
                    sum(span_lengths)
                    / len(span_lengths)
                    if span_lengths
                    else 0.0
                ),

            "min_span_length":
                min(span_lengths)
                if span_lengths
                else 0,

            "max_span_length":
                max(span_lengths)
                if span_lengths
                else 0,
        }
    )


with SUMMARY_OUT.open("w") as handle:

    columns = [
        "split",
        "total_proteins",
        "proteins_with_domain_span",
        "proteins_without_domain_span",
        "fraction_with_domain_span",
        "distinct_domain_ids",
        "distinct_domain_spans",
        "mean_span_length",
        "min_span_length",
        "max_span_length",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=columns,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(summary)


# ============================================================
# 6. Cross-check against existing protein-domain mappings
#
# Span extraction should reproduce the distinct
# (protein, InterPro_ID) assignments we already made.
# ============================================================

existing_paths = {
    "Intra1": (
        ROOT
        / "results/intra1_domains/"
        "intra1_protein_domain_assignments.tsv.gz"
    ),
    "Intra0": (
        ROOT
        / "results/intra0_domains/"
        "intra0_protein_domain_assignments.tsv.gz"
    ),
}


print()
print("===== CROSS-CHECK EXISTING MAPPINGS =====")

for split_name in ("Intra1", "Intra0"):

    extracted = {
        (protein, ipr)
        for protein, ipr, start, end
        in spans[split_name]
    }

    existing = set()

    path = existing_paths[split_name]

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        for row in reader:
            existing.add(
                (
                    row["protein"],
                    row["InterPro_ID"],
                )
            )

    missing = (
        existing
        - extracted
    )

    extra = (
        extracted
        - existing
    )

    print()
    print(split_name)

    print(
        f"  Existing protein-domain assignments: "
        f"{len(existing):,}"
    )

    print(
        f"  Extracted protein-domain assignments: "
        f"{len(extracted):,}"
    )

    print(
        f"  Existing missing from extracted: "
        f"{len(missing):,}"
    )

    print(
        f"  Extracted absent from existing: "
        f"{len(extra):,}"
    )

    if missing:
        print(
            "  WARNING: mapping mismatch detected"
        )

        for x in sorted(missing)[:10]:
            print(
                f"    missing: {x}"
            )

    if extra:
        print(
            "  WARNING: extra mapping detected"
        )

        for x in sorted(extra)[:10]:
            print(
                f"    extra: {x}"
            )


print()
print("===== SUMMARY =====")

for row in summary:
    print(
        f"{row['split']}: "
        f"{row['distinct_domain_spans']:,} spans | "
        f"{row['proteins_with_domain_span']:,}/"
        f"{row['total_proteins']:,} proteins | "
        f"{row['distinct_domain_ids']:,} Domain IDs"
    )


print()
print("WROTE:")
print(INTRA1_OUT)
print(INTRA0_OUT)
print(SUMMARY_OUT)

print()
print("BERNETT_GOLD_DOMAIN_SPANS_COMPLETE")
