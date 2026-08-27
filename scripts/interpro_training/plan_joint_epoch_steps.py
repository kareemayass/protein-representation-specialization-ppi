#!/usr/bin/env python3
"""Calculate joint-curriculum steps from complete Stage 1 exposures."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from interpro_joint.curriculum import (  # noqa: E402
    JointCurriculum,
)
from interpro_stage1.batching import (  # noqa: E402
    DistributedTokenBatchSampler,
)


DEFAULT_SPLIT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "top100_domain_pipeline_v1"
    / "splits_top64_length_filtered_v1"
)


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(
            path,
            mode="rt",
            encoding="utf-8",
            newline="",
        )

    return path.open(
        mode="rt",
        encoding="utf-8",
        newline="",
    )


def normalize_accession(raw: str) -> str:
    accession = raw.strip().split()[0]

    parts = accession.split("|")

    if (
        len(parts) >= 3
        and parts[0] in {"sp", "tr"}
    ):
        accession = parts[1]

    return accession


def read_fasta_lengths(
    path: Path,
) -> tuple[list[int], dict[str, int]]:
    ordered_lengths: list[int] = []
    lengths_by_accession: dict[str, int] = {}

    accession: str | None = None
    length = 0

    def commit() -> None:
        nonlocal accession
        nonlocal length

        if accession is None:
            return

        if accession in lengths_by_accession:
            raise RuntimeError(
                f"Duplicate FASTA accession: {accession}"
            )

        if length <= 0:
            raise RuntimeError(
                f"Empty sequence: {accession}"
            )

        lengths_by_accession[accession] = length
        ordered_lengths.append(length)

        accession = None
        length = 0

    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()

            if not line:
                continue

            if line.startswith(">"):
                commit()

                accession = normalize_accession(
                    line[1:]
                )
            else:
                if accession is None:
                    raise RuntimeError(
                        "Sequence before FASTA header"
                    )

                length += len(line)

    commit()

    if not ordered_lengths:
        raise RuntimeError(
            f"No sequences found in {path}"
        )

    return (
        ordered_lengths,
        lengths_by_accession,
    )


def read_stage2_lengths(
    *,
    path: Path,
    fasta_lengths: dict[str, int],
) -> list[int]:
    lengths: list[int] = []

    with open_text(path) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        required = {
            "protein_accession",
            "interpro_id",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                f"{path}: missing columns "
                f"{sorted(missing)}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            accession = normalize_accession(
                row["protein_accession"]
            )

            try:
                lengths.append(
                    fasta_lengths[accession]
                )
            except KeyError as error:
                raise RuntimeError(
                    f"{path}:{row_number}: "
                    f"{accession} absent from train FASTA"
                ) from error

    return lengths


def batches_per_rank(
    *,
    sequence_lengths: list[int],
    token_budget: int,
    world_size: int,
    gradient_accumulation_steps: int,
    seed: int,
) -> int:
    sampler = DistributedTokenBatchSampler(
        sequence_lengths=sequence_lengths,
        token_budget=token_budget,
        rank=0,
        world_size=world_size,
        seed=seed,
        bucket_size=512,
        shuffle=True,
        allow_oversize_singletons=False,
        batch_count_multiple=(
            world_size
            * gradient_accumulation_steps
        ),
    )

    count = len(sampler)

    if (
        count
        % gradient_accumulation_steps
        != 0
    ):
        raise RuntimeError(
            "Per-rank batch count is not divisible "
            "by gradient accumulation"
        )

    return count


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--token-budget",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=47,
    )

    parser.add_argument(
        "--format",
        choices=[
            "json",
            "shell",
        ],
        default="json",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    split_dir = args.split_dir.resolve()

    (
        stage1_lengths,
        fasta_lengths,
    ) = read_fasta_lengths(
        split_dir / "train.fasta"
    )

    stage2_lengths = read_stage2_lengths(
        path=(
            split_dir
            / (
                "train_balanced_"
                "domain_targets.tsv.gz"
            )
        ),
        fasta_lengths=fasta_lengths,
    )

    stage1_batches = batches_per_rank(
        sequence_lengths=stage1_lengths,
        token_budget=args.token_budget,
        world_size=args.world_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        seed=args.seed,
    )

    stage2_batches = batches_per_rank(
        sequence_lengths=stage2_lengths,
        token_budget=args.token_budget,
        world_size=args.world_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        seed=args.seed,
    )

    stage1_steps_per_exposure = (
        stage1_batches
        // args.gradient_accumulation_steps
    )

    stage2_steps_per_exposure = (
        stage2_batches
        // args.gradient_accumulation_steps
    )

    required_stage1_steps = (
        args.epochs
        * stage1_steps_per_exposure
    )

    # The fixed curriculum assigns 25/40 = 62.5% of
    # optimizer steps to Stage 1.
    raw_total_steps = math.ceil(
        required_stage1_steps
        / 0.625
    )

    required_multiple = math.lcm(
        40,
        args.epochs,
    )

    total_optimizer_steps = (
        math.ceil(
            raw_total_steps
            / required_multiple
        )
        * required_multiple
    )

    validate_every = (
        total_optimizer_steps
        // args.epochs
    )

    curriculum = JointCurriculum(
        total_optimizer_steps=(
            total_optimizer_steps
        ),
        seed=args.seed,
    )

    summary = curriculum.summarize()

    if (
        summary.stage1_steps
        < required_stage1_steps
    ):
        raise RuntimeError(
            "Calculated curriculum does not contain "
            "enough Stage 1 steps"
        )

    plan = {
        "epochs": args.epochs,
        "training_stage1_examples": len(
            stage1_lengths
        ),
        "training_stage2_pairs": len(
            stage2_lengths
        ),
        "token_budget": args.token_budget,
        "world_size": args.world_size,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "stage1_batches_per_rank": (
            stage1_batches
        ),
        "stage2_batches_per_rank": (
            stage2_batches
        ),
        "stage1_steps_per_complete_exposure": (
            stage1_steps_per_exposure
        ),
        "stage2_steps_per_complete_exposure": (
            stage2_steps_per_exposure
        ),
        "total_optimizer_steps": (
            total_optimizer_steps
        ),
        "validate_every": (
            validate_every
        ),
        "curriculum_stage1_steps": (
            summary.stage1_steps
        ),
        "curriculum_stage2_steps": (
            summary.stage2_steps
        ),
        "expected_stage1_exposures": (
            summary.stage1_steps
            / stage1_steps_per_exposure
        ),
        "expected_stage2_exposures": (
            summary.stage2_steps
            / stage2_steps_per_exposure
        ),
    }

    if args.format == "shell":
        print(
            "TOTAL_OPTIMIZER_STEPS="
            f"{total_optimizer_steps}"
        )

        print(
            "VALIDATE_EVERY="
            f"{validate_every}"
        )

        print(
            "STAGE1_STEPS_PER_EXPOSURE="
            f"{stage1_steps_per_exposure}"
        )

        print(
            "STAGE2_STEPS_PER_EXPOSURE="
            f"{stage2_steps_per_exposure}"
        )
    else:
        print(
            json.dumps(
                plan,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
