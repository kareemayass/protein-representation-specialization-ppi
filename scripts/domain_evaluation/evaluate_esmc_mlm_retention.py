#!/usr/bin/env python3
"""Paired ESMC MLM-retention diagnostic before vs after InterPro LoRA adaptation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from esm.models.esmc import ESMC

from interpro_joint.model import (
    DOMAIN_CLASS_COUNT,
    build_joint_model,
)
from interpro_joint.runtime import (
    load_model_checkpoint,
)
from interpro_stage1.config import (
    ESMC_MODEL_NAME,
)


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FASTA = (
    ROOT
    / "data"
    / "processed"
    / "top100_domain_pipeline_v1"
    / "mlm_diagnostic_v1"
    / "heldout_20k_len30_1024.fasta.gz"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "joint"
    / "top64_corrected_production_seed47"
    / "best_by_joint_score"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "joint"
    / "top64_corrected_production_seed47"
    / "evaluations"
    / "mlm_retention"
)

DEFAULT_MASK_FRACTION = 0.15
DEFAULT_MASK_SEED = 47
DEFAULT_TOKEN_BUDGET = 8192


@dataclass(frozen=True)
class ProteinRecord:
    accession: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fasta",
        type=Path,
        default=DEFAULT_FASTA,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=DEFAULT_MASK_FRACTION,
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=DEFAULT_MASK_SEED,
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Optional limit for smoke testing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not 0.0 < args.mask_fraction < 1.0:
        raise ValueError(
            "--mask-fraction must lie strictly between 0 and 1"
        )

    if args.token_budget <= 0:
        raise ValueError(
            "--token-budget must be positive"
        )

    if (
        args.max_proteins is not None
        and args.max_proteins <= 0
    ):
        raise ValueError(
            "--max-proteins must be positive"
        )

    return args


def normalize_accession(raw: str) -> str:
    token = raw.strip().split()[0]

    if token.startswith(">"):
        token = token[1:]

    parts = token.split("|")

    if (
        len(parts) >= 3
        and parts[0] in {"sp", "tr"}
    ):
        return parts[1]

    return token


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
        )

    return path.open(
        "r",
        encoding="utf-8",
    )


def read_fasta(
    path: Path,
) -> list[ProteinRecord]:
    records: list[ProteinRecord] = []

    header: str | None = None
    sequence_parts: list[str] = []

    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append(
                        ProteinRecord(
                            accession=(
                                normalize_accession(
                                    header
                                )
                            ),
                            sequence=(
                                "".join(
                                    sequence_parts
                                ).upper()
                            ),
                        )
                    )

                header = line
                sequence_parts = []
            else:
                if header is None:
                    raise RuntimeError(
                        f"{path}: sequence encountered "
                        "before FASTA header"
                    )

                sequence_parts.append(line)

    if header is not None:
        records.append(
            ProteinRecord(
                accession=(
                    normalize_accession(
                        header
                    )
                ),
                sequence=(
                    "".join(
                        sequence_parts
                    ).upper()
                ),
            )
        )

    if not records:
        raise RuntimeError(
            f"No FASTA records found in {path}"
        )

    accessions = [
        record.accession
        for record in records
    ]

    if len(accessions) != len(set(accessions)):
        raise RuntimeError(
            "Duplicate accessions found in diagnostic FASTA"
        )

    for record in records:
        if not (
            30
            <= record.length
            <= 1024
        ):
            raise RuntimeError(
                "Unexpected diagnostic sequence length: "
                f"{record.accession} "
                f"length={record.length}"
            )

    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def deterministic_rng(
    *,
    accession: str,
    sequence: str,
    seed: int,
) -> random.Random:
    payload = (
        f"{seed}|{accession}|{sequence}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    local_seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return random.Random(
        local_seed
    )


def make_batches(
    records: list[ProteinRecord],
    *,
    token_budget: int,
) -> list[list[ProteinRecord]]:
    """
    Greedy padded-token batching.

    Records are sorted by length so batch_size * maximum_token_length
    remains a useful approximation of actual transformer work.
    """
    ordered = sorted(
        records,
        key=lambda record: (
            record.length,
            record.accession,
        ),
    )

    batches: list[list[ProteinRecord]] = []
    current: list[ProteinRecord] = []
    current_max_tokens = 0

    for record in ordered:
        token_length = (
            record.length + 2
        )

        candidate_max = max(
            current_max_tokens,
            token_length,
        )

        candidate_cost = (
            candidate_max
            * (len(current) + 1)
        )

        if (
            current
            and candidate_cost
            > token_budget
        ):
            batches.append(current)
            current = []
            current_max_tokens = 0

        current.append(record)

        current_max_tokens = max(
            current_max_tokens,
            token_length,
        )

    if current:
        batches.append(current)

    return batches


def native_esmc(
    encoder,
):
    """
    Return the underlying ESMC object when the encoder is PEFT-wrapped.
    """
    if hasattr(
        encoder,
        "get_base_model",
    ):
        candidate = (
            encoder.get_base_model()
        )

        if hasattr(
            candidate,
            "sequence_head",
        ):
            return candidate

    if hasattr(
        encoder,
        "sequence_head",
    ):
        return encoder

    raise RuntimeError(
        "Could not locate underlying ESMC sequence_head"
    )


def assert_sequence_heads_identical(
    *,
    base_model: ESMC,
    adapted_encoder,
) -> None:
    """
    Verify that the pretrained MLM head itself was not changed.
    """
    adapted_base = native_esmc(
        adapted_encoder
    )

    base_state = (
        base_model
        .sequence_head
        .state_dict()
    )

    adapted_state = (
        adapted_base
        .sequence_head
        .state_dict()
    )

    if set(base_state) != set(
        adapted_state
    ):
        raise RuntimeError(
            "Base/adapted sequence-head state keys differ"
        )

    for key in base_state:
        if not torch.equal(
            base_state[key].detach().cpu(),
            adapted_state[key].detach().cpu(),
        ):
            raise RuntimeError(
                "Pretrained sequence head differs "
                f"after adapter construction/loading: {key}"
            )


def build_models(
    *,
    checkpoint: Path,
    device: torch.device,
):
    print(
        "Loading base ESMC-300M...",
        flush=True,
    )

    base_model = ESMC.from_pretrained(
        ESMC_MODEL_NAME,
        device=device,
    )

    print(
        "Loading InterPro-adapted ESMC-300M...",
        flush=True,
    )

    adapted_base = ESMC.from_pretrained(
        ESMC_MODEL_NAME,
        device=device,
    )

    adapted_joint = build_joint_model(
        encoder=adapted_base,
        stage1_head_type="mlp",
        stage1_dropout=0.1,
        stage2_dropout=0.1,
        domain_class_count=(
            DOMAIN_CLASS_COUNT
        ),
    )

    trainer_state = (
        load_model_checkpoint(
            model=adapted_joint,
            checkpoint_dir=checkpoint,
        )
    )

    if (
        trainer_state.get(
            "completed_epochs"
        )
        != 5
        or trainer_state.get(
            "global_optimizer_step"
        )
        != 42_040
    ):
        raise RuntimeError(
            "Unexpected selected checkpoint: "
            f"{trainer_state}"
        )

    base_model.eval()
    adapted_joint.eval()

    for parameter in (
        base_model.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    for parameter in (
        adapted_joint.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    assert_sequence_heads_identical(
        base_model=base_model,
        adapted_encoder=(
            adapted_joint.encoder
        ),
    )

    return (
        base_model,
        adapted_joint,
        trainer_state,
    )


def tokenize_and_mask_batch(
    *,
    records: list[ProteinRecord],
    tokenizer,
    mask_fraction: float,
    mask_seed: int,
    device: torch.device,
):
    encoded_rows: list[
        torch.Tensor
    ] = []

    masked_positions_by_row: list[
        torch.Tensor
    ] = []

    eligible_counts: list[int] = []

    special_ids = set(
        int(token_id)
        for token_id
        in tokenizer.all_special_ids
    )

    mask_token_id = int(
        tokenizer.mask_token_id
    )

    pad_token_id = int(
        tokenizer.pad_token_id
    )

    for record in records:
        token_ids = (
            tokenizer.encode(
                record.sequence,
                add_special_tokens=True,
            )
        )

        encoded = torch.tensor(
            [
                int(token_id)
                for token_id
                in token_ids
            ],
            dtype=torch.long,
        )

        if encoded.numel() != (
            record.length + 2
        ):
            raise RuntimeError(
                "Unexpected token count for "
                f"{record.accession}: "
                f"sequence_length={record.length}, "
                f"tokens={encoded.numel()}"
            )

        eligible_positions = [
            index
            for index in range(
                1,
                record.length + 1,
            )
            if int(
                encoded[index].item()
            )
            not in special_ids
        ]

        if not eligible_positions:
            raise RuntimeError(
                "No mask-eligible residue tokens for "
                f"{record.accession}"
            )

        requested_masks = int(
            math.floor(
                (
                    mask_fraction
                    * len(
                        eligible_positions
                    )
                )
                + 0.5
            )
        )

        mask_count = max(
            1,
            min(
                len(
                    eligible_positions
                ),
                requested_masks,
            ),
        )

        rng = deterministic_rng(
            accession=record.accession,
            sequence=record.sequence,
            seed=mask_seed,
        )

        selected = sorted(
            rng.sample(
                eligible_positions,
                mask_count,
            )
        )

        encoded_rows.append(
            encoded
        )

        masked_positions_by_row.append(
            torch.tensor(
                selected,
                dtype=torch.long,
            )
        )

        eligible_counts.append(
            len(
                eligible_positions
            )
        )

    maximum_length = max(
        row.numel()
        for row in encoded_rows
    )

    original_tokens = torch.full(
        (
            len(encoded_rows),
            maximum_length,
        ),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    for row_index, encoded in enumerate(
        encoded_rows
    ):
        original_tokens[
            row_index,
            : encoded.numel(),
        ] = encoded

    masked_tokens = (
        original_tokens.clone()
    )

    for (
        row_index,
        positions,
    ) in enumerate(
        masked_positions_by_row
    ):
        masked_tokens[
            row_index,
            positions,
        ] = mask_token_id

    return (
        original_tokens.to(
            device,
            non_blocking=True,
        ),
        masked_tokens.to(
            device,
            non_blocking=True,
        ),
        [
            positions.to(
                device,
                non_blocking=True,
            )
            for positions
            in masked_positions_by_row
        ],
        eligible_counts,
    )


def masked_metrics(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    if logits.ndim != 2:
        raise RuntimeError(
            "Expected [masked_positions, vocab] logits, "
            f"observed {tuple(logits.shape)}"
        )

    logits = logits.float()
    targets = targets.long()

    losses = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    predictions = torch.argmax(
        logits,
        dim=-1,
    )

    top1 = predictions.eq(
        targets
    )

    top5_predictions = torch.topk(
        logits,
        k=5,
        dim=-1,
    ).indices

    top5 = (
        top5_predictions
        .eq(
            targets.unsqueeze(-1)
        )
        .any(
            dim=-1
        )
    )

    return {
        "losses": (
            losses.detach().cpu()
        ),
        "top1": (
            top1.detach().cpu()
        ),
        "top5": (
            top5.detach().cpu()
        ),
    }


@torch.inference_mode()
def evaluate(
    *,
    records: list[ProteinRecord],
    base_model: ESMC,
    adapted_joint,
    mask_fraction: float,
    mask_seed: int,
    token_budget: int,
    progress_every: int,
    device: torch.device,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    batches = make_batches(
        records,
        token_budget=token_budget,
    )

    tokenizer = (
        base_model.tokenizer
    )

    if int(
        tokenizer.mask_token_id
    ) != 32:
        raise RuntimeError(
            "Unexpected ESMC mask token ID: "
            f"{tokenizer.mask_token_id}"
        )

    rows: list[
        dict[str, object]
    ] = []

    pooled = {
        "base_loss_sum": 0.0,
        "adapted_loss_sum": 0.0,
        "base_top1_sum": 0,
        "adapted_top1_sum": 0,
        "base_top5_sum": 0,
        "adapted_top5_sum": 0,
        "masked_count": 0,
        "eligible_count": 0,
    }

    for batch_index, batch in enumerate(
        batches,
        start=1,
    ):
        (
            original_tokens,
            masked_tokens,
            mask_positions,
            eligible_counts,
        ) = tokenize_and_mask_batch(
            records=batch,
            tokenizer=tokenizer,
            mask_fraction=mask_fraction,
            mask_seed=mask_seed,
            device=device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            base_output = (
                base_model(
                    sequence_tokens=(
                        masked_tokens
                    )
                )
            )

            adapted_output = (
                adapted_joint.encoder(
                    sequence_tokens=(
                        masked_tokens
                    )
                )
            )

        base_logits = (
            base_output
            .sequence_logits
        )

        adapted_logits = (
            adapted_output
            .sequence_logits
        )

        if (
            base_logits.shape
            != adapted_logits.shape
        ):
            raise RuntimeError(
                "Base/adapted logit shapes differ: "
                f"{tuple(base_logits.shape)} vs "
                f"{tuple(adapted_logits.shape)}"
            )

        for row_index, record in enumerate(
            batch
        ):
            positions = (
                mask_positions[
                    row_index
                ]
            )

            targets = original_tokens[
                row_index,
                positions,
            ]

            base_result = (
                masked_metrics(
                    logits=base_logits[
                        row_index,
                        positions,
                        :,
                    ],
                    targets=targets,
                )
            )

            adapted_result = (
                masked_metrics(
                    logits=adapted_logits[
                        row_index,
                        positions,
                        :,
                    ],
                    targets=targets,
                )
            )

            base_losses = (
                base_result["losses"]
            )
            adapted_losses = (
                adapted_result[
                    "losses"
                ]
            )

            base_top1 = (
                base_result["top1"]
            )
            adapted_top1 = (
                adapted_result[
                    "top1"
                ]
            )

            base_top5 = (
                base_result["top5"]
            )
            adapted_top5 = (
                adapted_result[
                    "top5"
                ]
            )

            masked_count = int(
                positions.numel()
            )

            eligible_count = int(
                eligible_counts[
                    row_index
                ]
            )

            base_ce = float(
                base_losses.mean().item()
            )
            adapted_ce = float(
                adapted_losses.mean().item()
            )

            base_acc = float(
                base_top1
                .float()
                .mean()
                .item()
            )
            adapted_acc = float(
                adapted_top1
                .float()
                .mean()
                .item()
            )

            base_top5_acc = float(
                base_top5
                .float()
                .mean()
                .item()
            )
            adapted_top5_acc = float(
                adapted_top5
                .float()
                .mean()
                .item()
            )

            rows.append(
                {
                    "protein_accession": (
                        record.accession
                    ),
                    "sequence_length": (
                        record.length
                    ),
                    "eligible_residue_tokens": (
                        eligible_count
                    ),
                    "masked_residues": (
                        masked_count
                    ),
                    "realized_mask_fraction": (
                        masked_count
                        / eligible_count
                    ),
                    "base_masked_ce": (
                        base_ce
                    ),
                    "adapted_masked_ce": (
                        adapted_ce
                    ),
                    "delta_ce_adapted_minus_base": (
                        adapted_ce
                        - base_ce
                    ),
                    "base_top1_accuracy": (
                        base_acc
                    ),
                    "adapted_top1_accuracy": (
                        adapted_acc
                    ),
                    "delta_top1_adapted_minus_base": (
                        adapted_acc
                        - base_acc
                    ),
                    "base_top5_accuracy": (
                        base_top5_acc
                    ),
                    "adapted_top5_accuracy": (
                        adapted_top5_acc
                    ),
                    "delta_top5_adapted_minus_base": (
                        adapted_top5_acc
                        - base_top5_acc
                    ),
                    "adapted_worse_ce": int(
                        adapted_ce
                        > base_ce
                    ),
                }
            )

            pooled[
                "base_loss_sum"
            ] += float(
                base_losses.sum().item()
            )

            pooled[
                "adapted_loss_sum"
            ] += float(
                adapted_losses.sum().item()
            )

            pooled[
                "base_top1_sum"
            ] += int(
                base_top1.sum().item()
            )

            pooled[
                "adapted_top1_sum"
            ] += int(
                adapted_top1.sum().item()
            )

            pooled[
                "base_top5_sum"
            ] += int(
                base_top5.sum().item()
            )

            pooled[
                "adapted_top5_sum"
            ] += int(
                adapted_top5.sum().item()
            )

            pooled[
                "masked_count"
            ] += masked_count

            pooled[
                "eligible_count"
            ] += eligible_count

        if (
            batch_index == 1
            or batch_index
            % progress_every
            == 0
            or batch_index
            == len(batches)
        ):
            print(
                "progress "
                f"{batch_index:,}/"
                f"{len(batches):,} batches; "
                f"{len(rows):,}/"
                f"{len(records):,} proteins",
                flush=True,
            )

        del (
            original_tokens,
            masked_tokens,
            base_output,
            adapted_output,
            base_logits,
            adapted_logits,
        )

    return (
        rows,
        pooled,
    )


def mean(
    values: list[float],
) -> float:
    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def median(
    values: list[float],
) -> float:
    return float(
        np.median(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def quantile(
    values: list[float],
    q: float,
) -> float:
    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def build_summary(
    *,
    rows: list[dict[str, object]],
    pooled: dict[str, object],
    fasta: Path,
    fasta_sha256: str,
    checkpoint: Path,
    trainer_state: dict,
    mask_fraction: float,
    mask_seed: int,
    token_budget: int,
    sequence_logit_dim: int,
) -> dict[str, object]:
    base_ce = [
        float(
            row["base_masked_ce"]
        )
        for row in rows
    ]

    adapted_ce = [
        float(
            row["adapted_masked_ce"]
        )
        for row in rows
    ]

    delta_ce = [
        float(
            row[
                "delta_ce_adapted_minus_base"
            ]
        )
        for row in rows
    ]

    base_top1 = [
        float(
            row[
                "base_top1_accuracy"
            ]
        )
        for row in rows
    ]

    adapted_top1 = [
        float(
            row[
                "adapted_top1_accuracy"
            ]
        )
        for row in rows
    ]

    base_top5 = [
        float(
            row[
                "base_top5_accuracy"
            ]
        )
        for row in rows
    ]

    adapted_top5 = [
        float(
            row[
                "adapted_top5_accuracy"
            ]
        )
        for row in rows
    ]

    masked_count = int(
        pooled["masked_count"]
    )

    eligible_count = int(
        pooled["eligible_count"]
    )

    worse_count = sum(
        int(
            row["adapted_worse_ce"]
        )
        for row in rows
    )

    return {
        "experiment": (
            "ESMC-300M MLM retention diagnostic"
        ),
        "interpretation": (
            "Paired comparison of the native pretrained "
            "ESMC sequence-prediction objective with and "
            "without the selected InterPro LoRA adapter."
        ),
        "input": {
            "fasta": str(
                fasta.resolve()
            ),
            "fasta_sha256": (
                fasta_sha256
            ),
            "proteins": len(rows),
            "minimum_sequence_length": min(
                int(
                    row[
                        "sequence_length"
                    ]
                )
                for row in rows
            ),
            "maximum_sequence_length": max(
                int(
                    row[
                        "sequence_length"
                    ]
                )
                for row in rows
            ),
        },
        "masking": {
            "requested_fraction": (
                mask_fraction
            ),
            "seed": mask_seed,
            "scheme": (
                "Per-protein deterministic random sample "
                "without replacement of round(0.15 * "
                "eligible residue tokens); BOS, EOS, PAD, "
                "MASK, UNK, and other tokenizer special "
                "tokens are never selected."
            ),
            "eligible_residue_tokens": (
                eligible_count
            ),
            "masked_residues": (
                masked_count
            ),
            "realized_fraction": (
                masked_count
                / eligible_count
            ),
        },
        "model": {
            "base_model": (
                str(
                    ESMC_MODEL_NAME
                )
            ),
            "checkpoint": str(
                checkpoint.resolve()
            ),
            "selected_checkpoint_epochs": (
                trainer_state.get(
                    "completed_epochs"
                )
            ),
            "selected_checkpoint_step": (
                trainer_state.get(
                    "global_optimizer_step"
                )
            ),
            "sequence_head_verified_identical": (
                True
            ),
            "sequence_logit_dimension": (
                sequence_logit_dim
            ),
            "token_budget": (
                token_budget
            ),
        },
        "equal_protein_weighted": {
            "base_masked_ce": (
                mean(base_ce)
            ),
            "adapted_masked_ce": (
                mean(adapted_ce)
            ),
            "delta_ce_adapted_minus_base": (
                mean(delta_ce)
            ),
            "base_top1_accuracy": (
                mean(base_top1)
            ),
            "adapted_top1_accuracy": (
                mean(adapted_top1)
            ),
            "delta_top1_adapted_minus_base": (
                mean(adapted_top1)
                - mean(base_top1)
            ),
            "base_top5_accuracy": (
                mean(base_top5)
            ),
            "adapted_top5_accuracy": (
                mean(adapted_top5)
            ),
            "delta_top5_adapted_minus_base": (
                mean(adapted_top5)
                - mean(base_top5)
            ),
        },
        "masked_residue_weighted": {
            "base_masked_ce": (
                float(
                    pooled[
                        "base_loss_sum"
                    ]
                )
                / masked_count
            ),
            "adapted_masked_ce": (
                float(
                    pooled[
                        "adapted_loss_sum"
                    ]
                )
                / masked_count
            ),
            "base_top1_accuracy": (
                int(
                    pooled[
                        "base_top1_sum"
                    ]
                )
                / masked_count
            ),
            "adapted_top1_accuracy": (
                int(
                    pooled[
                        "adapted_top1_sum"
                    ]
                )
                / masked_count
            ),
            "base_top5_accuracy": (
                int(
                    pooled[
                        "base_top5_sum"
                    ]
                )
                / masked_count
            ),
            "adapted_top5_accuracy": (
                int(
                    pooled[
                        "adapted_top5_sum"
                    ]
                )
                / masked_count
            ),
        },
        "paired_protein_differences": {
            "proteins_adapted_worse_ce": (
                worse_count
            ),
            "fraction_adapted_worse_ce": (
                worse_count
                / len(rows)
            ),
            "median_delta_ce": (
                median(delta_ce)
            ),
            "q25_delta_ce": (
                quantile(
                    delta_ce,
                    0.25,
                )
            ),
            "q75_delta_ce": (
                quantile(
                    delta_ce,
                    0.75,
                )
            ),
        },
    }


def write_outputs(
    *,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        output_dir
        / "per_protein_mlm_retention.tsv.gz"
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    for path in (
        table_path,
        summary_path,
    ):
        if (
            path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"{path} already exists; "
                "use --overwrite to replace it"
            )

    rows = sorted(
        rows,
        key=lambda row: str(
            row[
                "protein_accession"
            ]
        ),
    )

    fieldnames = [
        "protein_accession",
        "sequence_length",
        "eligible_residue_tokens",
        "masked_residues",
        "realized_mask_fraction",
        "base_masked_ce",
        "adapted_masked_ce",
        "delta_ce_adapted_minus_base",
        "base_top1_accuracy",
        "adapted_top1_accuracy",
        "delta_top1_adapted_minus_base",
        "base_top5_accuracy",
        "adapted_top5_accuracy",
        "delta_top5_adapted_minus_base",
        "adapted_worse_ce",
    ]

    with gzip.open(
        table_path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "MLM_RETENTION_EVALUATION_COMPLETE"
    )
    print(
        f"Per-protein results: {table_path}"
    )
    print(
        f"Summary: {summary_path}"
    )


def main() -> None:
    args = parse_args()

    if not args.fasta.is_file():
        raise FileNotFoundError(
            args.fasta
        )

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(
            args.checkpoint
        )

    required_checkpoint_files = (
        "trainer_state.json",
        "shared_lora_adapter.pt",
        "stage1_head.pt",
        "stage2_domain_head.pt",
    )

    for filename in (
        required_checkpoint_files
    ):
        path = (
            args.checkpoint
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    records = read_fasta(
        args.fasta
    )

    if args.max_proteins is not None:
        records = records[
            : args.max_proteins
        ]

    print(
        f"Proteins: {len(records):,}"
    )
    print(
        "Length range: "
        f"{min(r.length for r in records):,}-"
        f"{max(r.length for r in records):,}"
    )
    print(
        f"Mask fraction: {args.mask_fraction}"
    )
    print(
        f"Mask seed: {args.mask_seed}"
    )
    print(
        f"Token budget: {args.token_budget:,}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required"
        )

    device = torch.device(
        "cuda"
    )

    (
        base_model,
        adapted_joint,
        trainer_state,
    ) = build_models(
        checkpoint=args.checkpoint,
        device=device,
    )

    print(
        "Verified: base and adapted models use "
        "identical pretrained sequence-head weights."
    )

    # One small shape preflight before the full evaluation.
    first = records[0]

    first_tokens = torch.tensor(
        [
            base_model
            .tokenizer
            .encode(
                first.sequence,
                add_special_tokens=True,
            )
        ],
        dtype=torch.long,
        device=device,
    )

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ),
    ):
        preflight_output = (
            base_model(
                sequence_tokens=(
                    first_tokens
                )
            )
        )

    sequence_logit_dim = int(
        preflight_output
        .sequence_logits
        .shape[-1]
    )

    print(
        "Native ESMC sequence-logit dimension: "
        f"{sequence_logit_dim}"
    )

    del (
        first_tokens,
        preflight_output,
    )

    rows, pooled = evaluate(
        records=records,
        base_model=base_model,
        adapted_joint=adapted_joint,
        mask_fraction=(
            args.mask_fraction
        ),
        mask_seed=args.mask_seed,
        token_budget=(
            args.token_budget
        ),
        progress_every=(
            args.progress_every
        ),
        device=device,
    )

    summary = build_summary(
        rows=rows,
        pooled=pooled,
        fasta=args.fasta,
        fasta_sha256=(
            sha256_file(
                args.fasta
            )
        ),
        checkpoint=args.checkpoint,
        trainer_state=(
            trainer_state
        ),
        mask_fraction=(
            args.mask_fraction
        ),
        mask_seed=args.mask_seed,
        token_budget=(
            args.token_budget
        ),
        sequence_logit_dim=(
            sequence_logit_dim
        ),
    )

    write_outputs(
        rows=rows,
        summary=summary,
        output_dir=(
            args.output_dir
        ),
        overwrite=args.overwrite,
    )

    print()
    print(
        json.dumps(
            summary[
                "equal_protein_weighted"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        json.dumps(
            summary[
                "paired_protein_differences"
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
