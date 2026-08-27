#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path

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

ORIGINAL_PREP = (
    REPO_ROOT
    / "scripts/functional_regions/"
    "prepare_domain_compatibility_data.py"
)

EMBEDDINGS = (
    TUNA
    / "data/gold_standard/embeddings/"
    "esmc_300m_unadapted_embeddings.pt"
)

SPAN_DIR = (
    ROOT
    / "results/bernett_gold_family_spans"
)

OUTDIR = (
    ROOT
    / "results/family_compatibility_overnight"
)

OUTPUT = (
    OUTDIR
    / "base_family_data.pt"
)

SUMMARY = (
    OUTDIR
    / "prep_summary.tsv"
)


def load_module(name, path):

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

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original = load_module(
        "successful_domain_prep",
        ORIGINAL_PREP,
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

    print(
        "embedding proteins:",
        f"{len(embeddings):,}",
        flush=True,
    )

    payload = {
        "embedding_source":
            str(EMBEDDINGS),

        "hidden_size":
            960,

        "representation":
            (
                "L2-normalized mean Base-ESMC "
                "gold-Family-span embedding, "
                "averaged within protein x InterPro ID"
            ),
    }

    summaries = []

    for split_name in (
        "Intra1",
        "Intra0",
        "Intra2",
    ):

        span_path = (
            SPAN_DIR
            / (
                f"{split_name.lower()}_"
                "gold_family_spans.tsv.gz"
            )
        )

        result = (
            original.materialize_split(
                split_name=split_name,
                span_path=span_path,
                embeddings=embeddings,
            )
        )

        payload[
            split_name.lower()
        ] = result

        summary = dict(
            result["summary"]
        )

        summary[
            "split"
        ] = split_name

        summaries.append(
            summary
        )

        print(
            split_name,
            result["summary"],
            flush=True,
        )

    torch.save(
        payload,
        OUTPUT,
    )

    keys = []

    for row in summaries:

        for key in row:

            if key not in keys:
                keys.append(key)

    with SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=keys,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    print(
        "WROTE:",
        OUTPUT,
    )

    print(
        "FAMILY_PREP_COMPLETE"
    )


if __name__ == "__main__":
    main()
