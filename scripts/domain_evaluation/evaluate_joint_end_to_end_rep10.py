"""Validation-selected Stage 1 -> Stage 2 end-to-end Domain evaluation."""

from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from esm.models.esmc import ESMC
from interpro_stage1.config import ESMC_MODEL_NAME
from interpro_joint.data import (
    JointStage1Collator,
    JointStage1Dataset,
)
from interpro_joint.model_rep_preserve import build_joint_model
from interpro_joint.runtime import load_model_checkpoint


ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "top100_domain_pipeline_v1"
    / "splits_top64_length_filtered_v1"
)

AUDIT_DIR = (
    ROOT
    / "audits"
    / "end_to_end_resolvable_instances"
)

CHECKPOINT = (
    ROOT
    / "results"
    / "joint"
    / "rep_preserve_alpha10_seed47"
    / "best_by_joint_score"
)

OUTPUT = (
    ROOT
    / "results"
    / "joint"
    / "rep_preserve_alpha10_seed47"
    / "evaluations"
    / "end_to_end"
)

THRESHOLDS = tuple(
    round(value / 100.0, 2)
    for value in range(5, 100, 5)
)

MAXIMUM_GAPS = (
    0,
    1,
    2,
    3,
    5,
    10,
)

MINIMUM_SPAN_LENGTH = 10
MINIMUM_IOU = 0.5

# An unmatched prediction is ignored when at least half of its residues lie within an excluded ambiguous-overlap region.
IGNORE_FRACTION = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "validation",
            "test",
            "all",
        ),
        default="all",
    )

    parser.add_argument(
        "--token-budget",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(
            path,
            mode="rt",
            encoding="utf-8",
        )

    return path.open(
        mode="rt",
        encoding="utf-8",
    )


def harmonic_mean(
    precision: float,
    recall: float,
) -> float:
    if precision + recall == 0.0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def span_length(
    span: tuple[int, int],
) -> int:
    return (
        span[1]
        - span[0]
        + 1
    )


def intersection_length(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    return max(
        0,
        min(
            first[1],
            second[1],
        )
        - max(
            first[0],
            second[0],
        )
        + 1,
    )


def span_iou(
    first: tuple[int, int],
    second: tuple[int, int],
) -> float:
    overlap = intersection_length(
        first,
        second,
    )

    if overlap == 0:
        return 0.0

    union = (
        span_length(first)
        + span_length(second)
        - overlap
    )

    return overlap / union


def mask_to_spans(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Convert a Boolean mask into zero-based inclusive spans."""

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.size == 0:
        return []

    padded = np.pad(
        mask.astype(
            np.int8
        ),
        (1, 1),
        constant_values=0,
    )

    transitions = np.diff(
        padded
    )

    starts = np.flatnonzero(
        transitions == 1
    )

    ends = (
        np.flatnonzero(
            transitions == -1
        )
        - 1
    )

    return [
        (
            int(start),
            int(end),
        )
        for start, end in zip(
            starts,
            ends,
            strict=True,
        )
    ]


def decode_probabilities(
    probability: np.ndarray,
    *,
    threshold: float,
    maximum_gap: int,
) -> list[tuple[int, int]]:
    """
    Apply the existing Stage 1 decoding policy:
        threshold
        -> fill short internal gaps
        -> remove components shorter than 10 residues
    """

    mask = (
        np.asarray(
            probability
        )
        >= threshold
    )

    if maximum_gap > 0:
        for start, end in mask_to_spans(
            ~mask
        ):
            gap_length = (
                end
                - start
                + 1
            )

            if gap_length > maximum_gap:
                continue

            if (
                start == 0
                or end == mask.size - 1
            ):
                continue

            if (
                mask[start - 1]
                and mask[end + 1]
            ):
                mask[
                    start : end + 1
                ] = True

    for start, end in mask_to_spans(
        mask
    ):
        component_length = (
            end
            - start
            + 1
        )

        if (
            component_length
            < MINIMUM_SPAN_LENGTH
        ):
            mask[
                start : end + 1
            ] = False

    return mask_to_spans(
        mask
    )


def read_selected_domains() -> tuple[
    tuple[str, ...],
    dict[str, int],
]:
    path = (
        SPLIT_DIR
        / "selected_domains.txt"
    )

    domains = tuple(
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )

    if (
        len(domains) != 64
        or len(set(domains)) != 64
    ):
        raise RuntimeError(
            "Expected 64 unique selected Domains"
        )

    return (
        domains,
        {
            domain: class_index
            for class_index, domain
            in enumerate(domains)
        },
    )


def load_gold(
    split: str,
    *,
    class_by_domain: dict[str, int],
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[tuple[int, int]]],
]:
    gold: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    gold_path = (
        AUDIT_DIR
        / split
        / "gold_regions.tsv.gz"
    )

    if not gold_path.is_file():
        raise FileNotFoundError(
            gold_path
        )

    with open_text(
        gold_path
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        for row in reader:
            if (
                row["evaluation_status"]
                != "evaluable"
            ):
                continue

            accession = (
                row[
                    "protein_accession"
                ].strip()
            )

            interpro_id = (
                row[
                    "interpro_id"
                ].strip()
            )

            if (
                interpro_id
                not in class_by_domain
            ):
                raise RuntimeError(
                    "Unknown evaluable Domain: "
                    f"{interpro_id}"
                )

            # Convert one-based inclusive coordinates to zero-based inclusive coordinates.
            span = (
                int(row["start"]) - 1,
                int(row["end"]) - 1,
            )

            gold[
                accession
            ].append(
                {
                    "domain": (
                        interpro_id
                    ),
                    "class_index": (
                        class_by_domain[
                            interpro_id
                        ]
                    ),
                    "span": span,
                }
            )

    for records in gold.values():
        records.sort(
            key=lambda record: (
                record["span"],
                record["domain"],
            )
        )

    ignored: dict[
        str,
        list[tuple[int, int]],
    ] = defaultdict(list)

    ignore_path = (
        AUDIT_DIR
        / split
        / "ambiguous_ignore_spans.tsv.gz"
    )

    if not ignore_path.is_file():
        raise FileNotFoundError(
            ignore_path
        )

    with open_text(
        ignore_path
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        for row in reader:
            accession = (
                row[
                    "protein_accession"
                ].strip()
            )

            ignored[
                accession
            ].append(
                (
                    int(row["start"]) - 1,
                    int(row["end"]) - 1,
                )
            )

    for spans in ignored.values():
        spans.sort()

    return (
        dict(gold),
        dict(ignored),
    )


def prediction_is_ignored(
    predicted_span: tuple[int, int],
    ignored_regions: list[
        tuple[int, int]
    ],
) -> bool:
    covered = sum(
        intersection_length(
            predicted_span,
            ignored_region,
        )
        for ignored_region
        in ignored_regions
    )

    fraction = (
        covered
        / span_length(
            predicted_span
        )
    )

    return (
        fraction
        >= IGNORE_FRACTION
    )


def maximum_cardinality_matches(
    gold_spans: list[
        tuple[int, int]
    ],
    predicted_spans: list[
        tuple[int, int]
    ],
) -> list[tuple[int, int]]:
    """
    Exact maximum-cardinality bipartite matching.

    Used during validation selection, where only the number of
    localization matches affects precision, recall and F1.
    """

    adjacency = []

    for gold_span in gold_spans:
        candidates = sorted(
            (
                (
                    predicted_index,
                    span_iou(
                        gold_span,
                        predicted_span,
                    ),
                )
                for (
                    predicted_index,
                    predicted_span,
                ) in enumerate(
                    predicted_spans
                )
            ),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        adjacency.append(
            [
                predicted_index
                for (
                    predicted_index,
                    score,
                ) in candidates
                if score >= MINIMUM_IOU
            ]
        )

    matched_gold_by_prediction: dict[
        int,
        int,
    ] = {}

    def augment(
        gold_index: int,
        visited_predictions: set[int],
    ) -> bool:
        for predicted_index in (
            adjacency[
                gold_index
            ]
        ):
            if (
                predicted_index
                in visited_predictions
            ):
                continue

            visited_predictions.add(
                predicted_index
            )

            previous_gold = (
                matched_gold_by_prediction
                .get(
                    predicted_index
                )
            )

            if (
                previous_gold is None
                or augment(
                    previous_gold,
                    visited_predictions,
                )
            ):
                matched_gold_by_prediction[
                    predicted_index
                ] = gold_index

                return True

        return False

    gold_order = sorted(
        range(
            len(gold_spans)
        ),
        key=lambda index: (
            len(
                adjacency[
                    index
                ]
            ),
            index,
        ),
    )

    for gold_index in gold_order:
        augment(
            gold_index,
            set(),
        )

    return sorted(
        (
            gold_index,
            predicted_index,
        )
        for (
            predicted_index,
            gold_index,
        ) in (
            matched_gold_by_prediction
            .items()
        )
    )


def maximum_cardinality_iou_matches(
    gold_spans: list[
        tuple[int, int]
    ],
    predicted_spans: list[
        tuple[int, int]
    ],
) -> list[tuple[int, int]]:
    """
    Maximize match count first and total IoU second.

    This is used for the final test assignment.
    """

    if (
        not gold_spans
        or not predicted_spans
    ):
        return []

    size = max(
        len(gold_spans),
        len(predicted_spans),
    )

    benefit = np.zeros(
        (
            size,
            size,
        ),
        dtype=np.float64,
    )

    valid = np.zeros(
        (
            len(gold_spans),
            len(predicted_spans),
        ),
        dtype=bool,
    )

    for (
        gold_index,
        gold_span,
    ) in enumerate(
        gold_spans
    ):
        for (
            predicted_index,
            predicted_span,
        ) in enumerate(
            predicted_spans
        ):
            score = span_iou(
                gold_span,
                predicted_span,
            )

            if score < MINIMUM_IOU:
                continue

            valid[
                gold_index,
                predicted_index,
            ] = True

            # Any additional valid match is more valuable
            # than every possible IoU tie-break combined.
            benefit[
                gold_index,
                predicted_index,
            ] = (
                size
                + 1
                + score
            )

    rows, columns = (
        linear_sum_assignment(
            -benefit
        )
    )

    return [
        (
            int(gold_index),
            int(predicted_index),
        )
        for (
            gold_index,
            predicted_index,
        ) in zip(
            rows,
            columns,
            strict=True,
        )
        if (
            gold_index
            < len(gold_spans)
            and predicted_index
            < len(predicted_spans)
            and valid[
                gold_index,
                predicted_index
            ]
        )
    ]


def iter_token_budget_batches(
    lengths: list[int],
    *,
    token_budget: int,
):
    ordered = sorted(
        range(
            len(lengths)
        ),
        key=lambda index: (
            lengths[index],
            index,
        ),
    )

    batch: list[int] = []
    maximum_tokens = 0

    for index in ordered:
        token_count = (
            lengths[index]
            + 2
        )

        proposed_maximum = max(
            maximum_tokens,
            token_count,
        )

        proposed_cost = (
            proposed_maximum
            * (len(batch) + 1)
        )

        if (
            batch
            and proposed_cost
            > token_budget
        ):
            yield batch

            batch = []
            maximum_tokens = 0

        batch.append(
            index
        )

        maximum_tokens = max(
            maximum_tokens,
            token_count,
        )

    if batch:
        yield batch


def build_dataset(
    split: str,
) -> JointStage1Dataset:
    return JointStage1Dataset(
        fasta_path=(
            SPLIT_DIR
            / f"{split}.fasta"
        ),
        annotations_path=(
            SPLIT_DIR
            / (
                f"{split}_"
                "annotations.tsv.gz"
            )
        ),
        selected_domains_path=(
            SPLIT_DIR
            / "selected_domains.txt"
        ),
    )


def build_model(
    class_count: int,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "A CUDA GPU is required"
        )

    encoder = ESMC.from_pretrained(
        ESMC_MODEL_NAME,
        device=device,
    )

    model = build_joint_model(
        encoder=encoder,
        stage1_head_type="mlp",
        stage1_dropout=0.1,
        stage2_dropout=0.1,
        domain_class_count=(
            class_count
        ),
    )

    trainer_state = (
        load_model_checkpoint(
            model=model,
            checkpoint_dir=(
                CHECKPOINT
            ),
        )
    )

    if (
        trainer_state.get("completed_epochs")
        != 4
        or trainer_state.get(
            "global_optimizer_step"
        )
        != 33_632
    ):
        raise RuntimeError(
            "Unexpected selected checkpoint: "
            f"{trainer_state}"
        )

    model.to(
        device
    )

    model.eval()

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    return (
        model,
        trainer_state,
        device,
    )


def infer_validation_probabilities(
    *,
    model,
    dataset: JointStage1Dataset,
    device: torch.device,
    token_budget: int,
    progress_every: int,
) -> dict[str, np.ndarray]:
    collator = (
        JointStage1Collator()
    )

    output: dict[
        str,
        np.ndarray,
    ] = {}

    work = list(
        iter_token_budget_batches(
            dataset.sequence_lengths,
            token_budget=(
                token_budget
            ),
        )
    )

    print(
        json.dumps(
            {
                "event": (
                    "validation_inference_start"
                ),
                "proteins": len(
                    dataset
                ),
                "batches": len(
                    work
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    with torch.inference_mode():
        for (
            batch_number,
            indices,
        ) in enumerate(
            work,
            start=1,
        ):
            examples = [
                dataset[index]
                for index in indices
            ]

            batch = collator(
                examples
            )

            sequence_tokens = (
                batch[
                    "sequence_tokens"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = (
                    model.forward_stage1(
                        sequence_tokens=(
                            sequence_tokens
                        ),
                        residue_length=(
                            batch[
                                "labels"
                            ].shape[1]
                        ),
                    )
                )

            probability = (
                torch.sigmoid(
                    logits.float()
                )
                .squeeze(-1)
                .cpu()
            )

            for (
                row,
                accession,
            ) in enumerate(
                batch[
                    "accessions"
                ]
            ):
                length = int(
                    batch[
                        "lengths"
                    ][row]
                )

                output[
                    accession
                ] = (
                    probability[
                        row,
                        :length,
                    ]
                    .numpy()
                    .astype(
                        np.float16
                    )
                )

            if (
                progress_every > 0
                and (
                    batch_number
                    % progress_every
                    == 0
                    or batch_number
                    == len(work)
                )
            ):
                print(
                    json.dumps(
                        {
                            "event": (
                                "validation_inference_progress"
                            ),
                            "batch": (
                                batch_number
                            ),
                            "batches": (
                                len(work)
                            ),
                            "proteins_complete": (
                                len(output)
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if (
        set(output)
        != set(
            dataset.sequences
        )
    ):
        raise RuntimeError(
            "Validation probability "
            "accession mismatch"
        )

    return output


def localization_score(
    *,
    dataset: JointStage1Dataset,
    probabilities: dict[
        str,
        np.ndarray,
    ],
    gold: dict[
        str,
        list[dict[str, object]],
    ],
    ignored: dict[
        str,
        list[tuple[int, int]],
    ],
    threshold: float,
    maximum_gap: int,
) -> dict[str, float | int]:
    true_positive = 0
    gold_count = 0
    prediction_count = 0
    ignored_prediction_count = 0

    for record in dataset.records:
        accession = (
            record.protein_accession
        )

        predicted_spans = (
            decode_probabilities(
                probabilities[
                    accession
                ],
                threshold=threshold,
                maximum_gap=(
                    maximum_gap
                ),
            )
        )

        gold_spans = [
            item["span"]
            for item in gold.get(
                accession,
                [],
            )
        ]

        matches = (
            maximum_cardinality_matches(
                gold_spans,
                predicted_spans,
            )
        )

        matched_predictions = {
            predicted_index
            for (
                _gold_index,
                predicted_index,
            ) in matches
        }

        ignored_here = sum(
            prediction_is_ignored(
                predicted_span,
                ignored.get(
                    accession,
                    [],
                ),
            )
            for (
                predicted_index,
                predicted_span,
            ) in enumerate(
                predicted_spans
            )
            if (
                predicted_index
                not in matched_predictions
            )
        )

        true_positive += len(
            matches
        )

        gold_count += len(
            gold_spans
        )

        prediction_count += len(
            predicted_spans
        )

        ignored_prediction_count += (
            ignored_here
        )

    scored_predictions = (
        prediction_count
        - ignored_prediction_count
    )

    precision = (
        true_positive
        / scored_predictions
        if scored_predictions
        else 0.0
    )

    recall = (
        true_positive
        / gold_count
        if gold_count
        else 0.0
    )

    return {
        "threshold": threshold,
        "maximum_gap": maximum_gap,
        "minimum_span_length": (
            MINIMUM_SPAN_LENGTH
        ),
        "true_positive_matches": (
            true_positive
        ),
        "evaluable_gold_regions": (
            gold_count
        ),
        "all_predictions": (
            prediction_count
        ),
        "ignored_predictions": (
            ignored_prediction_count
        ),
        "scored_predictions": (
            scored_predictions
        ),
        "precision": precision,
        "recall": recall,
        "f1": harmonic_mean(
            precision,
            recall,
        ),
    }


def select_decoding(
    *,
    dataset: JointStage1Dataset,
    probabilities: dict[
        str,
        np.ndarray,
    ],
    gold: dict[
        str,
        list[dict[str, object]],
    ],
    ignored: dict[
        str,
        list[tuple[int, int]],
    ],
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int]],
]:
    rows = []

    for maximum_gap in (
        MAXIMUM_GAPS
    ):
        for threshold in (
            THRESHOLDS
        ):
            row = (
                localization_score(
                    dataset=dataset,
                    probabilities=(
                        probabilities
                    ),
                    gold=gold,
                    ignored=ignored,
                    threshold=(
                        threshold
                    ),
                    maximum_gap=(
                        maximum_gap
                    ),
                )
            )

            rows.append(
                row
            )

            print(
                json.dumps(
                    {
                        "event": (
                            "validation_grid"
                        ),
                        **row,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    selected = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            -row[
                "maximum_gap"
            ],
            -abs(
                row["threshold"]
                - 0.5
            ),
        ),
    )

    return (
        selected,
        rows,
    )


def classify_spans(
    *,
    model,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    spans: list[
        tuple[int, int]
    ],
    class_count: int,
) -> torch.Tensor:
    if not spans:
        return torch.empty(
            (
                0,
                class_count,
            ),
            dtype=torch.float32,
        )

    target_mask = torch.zeros(
        (
            len(spans),
            embeddings.shape[1],
        ),
        dtype=torch.bool,
        device=embeddings.device,
    )

    for (
        row,
        (
            start,
            end,
        ),
    ) in enumerate(
        spans
    ):
        # Token zero is BOS.
        target_mask[
            row,
            start + 1 : end + 2,
        ] = True

    representation = (
        model.pool_visible_regions(
            embeddings=(
                embeddings.expand(
                    len(spans),
                    -1,
                    -1,
                )
            ),
            attention_mask=(
                attention_mask.expand(
                    len(spans),
                    -1,
                )
            ),
            target_token_mask=(
                target_mask
            ),
        )
    )

    return (
        model.stage2_domain_head(
            representation
        )
        .float()
        .cpu()
    )


def evaluate_test(
    *,
    model,
    dataset: JointStage1Dataset,
    device: torch.device,
    domains: tuple[str, ...],
    gold: dict[
        str,
        list[dict[str, object]],
    ],
    ignored: dict[
        str,
        list[tuple[int, int]],
    ],
    selected: dict[str, object],
    token_budget: int,
    progress_every: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    collator = (
        JointStage1Collator()
    )

    work = list(
        iter_token_budget_batches(
            dataset.sequence_lengths,
            token_budget=(
                token_budget
            ),
        )
    )

    totals: defaultdict[
        str,
        int,
    ] = defaultdict(int)

    matched_iou_sum = 0.0
    matched_probability_sum = 0.0
    matched_nll_sum = 0.0

    prediction_rows: list[
        dict[str, object]
    ] = []

    gold_rows: list[
        dict[str, object]
    ] = []

    with torch.inference_mode():
        for (
            batch_number,
            indices,
        ) in enumerate(
            work,
            start=1,
        ):
            examples = [
                dataset[index]
                for index in indices
            ]

            batch = collator(
                examples
            )

            sequence_tokens = (
                batch[
                    "sequence_tokens"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            lengths = (
                batch[
                    "lengths"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            token_positions = (
                torch.arange(
                    sequence_tokens.shape[1],
                    device=device,
                )
                .unsqueeze(0)
            )

            attention_mask = (
                token_positions
                < (
                    lengths.unsqueeze(1)
                    + 2
                )
            )

            residue_length = (
                batch[
                    "labels"
                ].shape[1]
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                embeddings = (
                    model.encode(
                        sequence_tokens=(
                            sequence_tokens
                        ),
                        attention_mask=(
                            attention_mask
                        ),
                    )
                )

                stage1_logits = (
                    model.stage1_head(
                        embeddings[
                            :,
                            1 : 1
                            + residue_length,
                            :,
                        ]
                    )
                )

            probability = (
                torch.sigmoid(
                    stage1_logits.float()
                )
                .squeeze(-1)
            )

            for (
                row,
                accession,
            ) in enumerate(
                batch[
                    "accessions"
                ]
            ):
                sequence_length = int(
                    batch[
                        "lengths"
                    ][row]
                )

                predicted_spans = (
                    decode_probabilities(
                        probability[
                            row,
                            :sequence_length,
                        ]
                        .cpu()
                        .numpy(),
                        threshold=float(
                            selected[
                                "threshold"
                            ]
                        ),
                        maximum_gap=int(
                            selected[
                                "maximum_gap"
                            ]
                        ),
                    )
                )

                gold_here = gold.get(
                    accession,
                    [],
                )

                gold_spans = [
                    item["span"]
                    for item in gold_here
                ]

                combined_spans = (
                    predicted_spans
                    + gold_spans
                )

                logits = classify_spans(
                    model=model,
                    embeddings=(
                        embeddings[
                            row : row + 1
                        ]
                    ),
                    attention_mask=(
                        attention_mask[
                            row : row + 1
                        ]
                    ),
                    spans=(
                        combined_spans
                    ),
                    class_count=(
                        len(domains)
                    ),
                )

                predicted_logits = logits[
                    : len(
                        predicted_spans
                    )
                ]

                oracle_logits = logits[
                    len(
                        predicted_spans
                    ) :
                ]

                if predicted_spans:
                    predicted_classes = (
                        predicted_logits
                        .argmax(
                            dim=1
                        )
                        .tolist()
                    )

                    predicted_top5 = (
                        predicted_logits
                        .topk(
                            k=min(
                                5,
                                len(domains),
                            ),
                            dim=1,
                        )
                        .indices
                        .tolist()
                    )
                else:
                    predicted_classes = []
                    predicted_top5 = []

                if gold_here:
                    oracle_classes = (
                        oracle_logits
                        .argmax(
                            dim=1
                        )
                        .tolist()
                    )

                    oracle_top5 = (
                        oracle_logits
                        .topk(
                            k=min(
                                5,
                                len(domains),
                            ),
                            dim=1,
                        )
                        .indices
                        .tolist()
                    )
                else:
                    oracle_classes = []
                    oracle_top5 = []

                matches = (
                    maximum_cardinality_iou_matches(
                        gold_spans,
                        predicted_spans,
                    )
                )

                gold_by_prediction = {
                    predicted_index: gold_index
                    for (
                        gold_index,
                        predicted_index,
                    ) in matches
                }

                prediction_by_gold = {
                    gold_index: predicted_index
                    for (
                        gold_index,
                        predicted_index,
                    ) in matches
                }

                ignored_predictions = {
                    predicted_index
                    for (
                        predicted_index,
                        predicted_span,
                    ) in enumerate(
                        predicted_spans
                    )
                    if (
                        predicted_index
                        not in gold_by_prediction
                        and prediction_is_ignored(
                            predicted_span,
                            ignored.get(
                                accession,
                                [],
                            ),
                        )
                    )
                }

                correct_ids = 0
                top5_ids = 0

                for (
                    gold_index,
                    predicted_index,
                ) in matches:
                    target = int(
                        gold_here[
                            gold_index
                        ][
                            "class_index"
                        ]
                    )

                    correct_ids += int(
                        predicted_classes[
                            predicted_index
                        ]
                        == target
                    )

                    top5_ids += int(
                        target
                        in predicted_top5[
                            predicted_index
                        ]
                    )

                    log_probability = (
                        torch.log_softmax(
                            predicted_logits[
                                predicted_index
                            ],
                            dim=0,
                        )
                    )

                    matched_probability_sum += float(
                        log_probability[
                            target
                        ]
                        .exp()
                        .item()
                    )

                    matched_nll_sum += float(
                        -log_probability[
                            target
                        ].item()
                    )

                    matched_iou_sum += (
                        span_iou(
                            gold_spans[
                                gold_index
                            ],
                            predicted_spans[
                                predicted_index
                            ],
                        )
                    )

                oracle_correct = 0
                oracle_top5_correct = 0

                for (
                    gold_index,
                    item,
                ) in enumerate(
                    gold_here
                ):
                    target = int(
                        item[
                            "class_index"
                        ]
                    )

                    oracle_correct += int(
                        oracle_classes[
                            gold_index
                        ]
                        == target
                    )

                    oracle_top5_correct += int(
                        target
                        in oracle_top5[
                            gold_index
                        ]
                    )

                scored_predictions = (
                    len(
                        predicted_spans
                    )
                    - len(
                        ignored_predictions
                    )
                )

                totals[
                    "proteins"
                ] += 1

                totals[
                    "evaluable_gold_regions"
                ] += len(
                    gold_here
                )

                totals[
                    "all_predictions"
                ] += len(
                    predicted_spans
                )

                totals[
                    "ignored_predictions"
                ] += len(
                    ignored_predictions
                )

                totals[
                    "scored_predictions"
                ] += (
                    scored_predictions
                )

                totals[
                    "localization_matches"
                ] += len(
                    matches
                )

                totals[
                    "correct_ids_among_matches"
                ] += correct_ids

                totals[
                    "top5_ids_among_matches"
                ] += top5_ids

                totals[
                    "oracle_correct_ids"
                ] += oracle_correct

                totals[
                    "oracle_top5_ids"
                ] += (
                    oracle_top5_correct
                )

                for (
                    predicted_index,
                    predicted_span,
                ) in enumerate(
                    predicted_spans
                ):
                    gold_index = (
                        gold_by_prediction
                        .get(
                            predicted_index
                        )
                    )

                    predicted_class = (
                        predicted_classes[
                            predicted_index
                        ]
                    )

                    if gold_index is not None:
                        target = gold_here[
                            gold_index
                        ]

                        id_correct = int(
                            predicted_class
                            == target[
                                "class_index"
                            ]
                        )

                        status = (
                            "matched_correct_id"
                            if id_correct
                            else "matched_wrong_id"
                        )

                        matched_domain = (
                            target[
                                "domain"
                            ]
                        )

                        matched_iou = (
                            span_iou(
                                predicted_span,
                                target[
                                    "span"
                                ],
                            )
                        )

                    elif (
                        predicted_index
                        in ignored_predictions
                    ):
                        id_correct = ""
                        status = (
                            "ignored_ambiguous_overlap"
                        )
                        matched_domain = ""
                        matched_iou = ""

                    else:
                        id_correct = ""
                        status = (
                            "unmatched_false_positive"
                        )
                        matched_domain = ""
                        matched_iou = ""

                    prediction_rows.append(
                        {
                            "protein_accession": (
                                accession
                            ),
                            "predicted_start": (
                                predicted_span[
                                    0
                                ]
                                + 1
                            ),
                            "predicted_end": (
                                predicted_span[
                                    1
                                ]
                                + 1
                            ),
                            "predicted_interpro_id": (
                                domains[
                                    predicted_class
                                ]
                            ),
                            "matched_gold_interpro_id": (
                                matched_domain
                            ),
                            "matched_iou": (
                                matched_iou
                            ),
                            "id_correct": (
                                id_correct
                            ),
                            "status": (
                                status
                            ),
                        }
                    )

                for (
                    gold_index,
                    item,
                ) in enumerate(
                    gold_here
                ):
                    predicted_index = (
                        prediction_by_gold
                        .get(
                            gold_index
                        )
                    )

                    end_to_end_correct = int(
                        predicted_index
                        is not None
                        and predicted_classes[
                            predicted_index
                        ]
                        == item[
                            "class_index"
                        ]
                    )

                    gold_rows.append(
                        {
                            "protein_accession": (
                                accession
                            ),
                            "gold_start": (
                                item[
                                    "span"
                                ][0]
                                + 1
                            ),
                            "gold_end": (
                                item[
                                    "span"
                                ][1]
                                + 1
                            ),
                            "gold_interpro_id": (
                                item[
                                    "domain"
                                ]
                            ),
                            "matched_prediction": (
                                ""
                                if predicted_index
                                is None
                                else predicted_index
                            ),
                            "end_to_end_correct": (
                                end_to_end_correct
                            ),
                            "oracle_predicted_interpro_id": (
                                domains[
                                    oracle_classes[
                                        gold_index
                                    ]
                                ]
                            ),
                            "oracle_correct": int(
                                oracle_classes[
                                    gold_index
                                ]
                                == item[
                                    "class_index"
                                ]
                            ),
                        }
                    )

            if (
                progress_every > 0
                and (
                    batch_number
                    % progress_every
                    == 0
                    or batch_number
                    == len(work)
                )
            ):
                print(
                    json.dumps(
                        {
                            "event": (
                                "test_inference_progress"
                            ),
                            "batch": (
                                batch_number
                            ),
                            "batches": (
                                len(work)
                            ),
                            "proteins_complete": (
                                totals[
                                    "proteins"
                                ]
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    gold_count = totals[
        "evaluable_gold_regions"
    ]

    scored_predictions = totals[
        "scored_predictions"
    ]

    localization_matches = totals[
        "localization_matches"
    ]

    correct_ids = totals[
        "correct_ids_among_matches"
    ]

    localization_precision = (
        localization_matches
        / scored_predictions
        if scored_predictions
        else 0.0
    )

    localization_recall = (
        localization_matches
        / gold_count
        if gold_count
        else 0.0
    )

    end_to_end_precision = (
        correct_ids
        / scored_predictions
        if scored_predictions
        else 0.0
    )

    end_to_end_recall = (
        correct_ids
        / gold_count
        if gold_count
        else 0.0
    )

    metrics = {
        **dict(totals),
        "localization_precision_iou_0_5": (
            localization_precision
        ),
        "localization_recall_iou_0_5": (
            localization_recall
        ),
        "localization_f1_iou_0_5": (
            harmonic_mean(
                localization_precision,
                localization_recall,
            )
        ),
        "conditional_id_accuracy_among_matched": (
            correct_ids
            / localization_matches
            if localization_matches
            else 0.0
        ),
        "conditional_top5_accuracy_among_matched": (
            totals[
                "top5_ids_among_matches"
            ]
            / localization_matches
            if localization_matches
            else 0.0
        ),
        "end_to_end_precision": (
            end_to_end_precision
        ),
        "end_to_end_recall": (
            end_to_end_recall
        ),
        "end_to_end_f1": (
            harmonic_mean(
                end_to_end_precision,
                end_to_end_recall,
            )
        ),
        "oracle_gold_span_id_accuracy": (
            totals[
                "oracle_correct_ids"
            ]
            / gold_count
            if gold_count
            else 0.0
        ),
        "oracle_gold_span_top5_accuracy": (
            totals[
                "oracle_top5_ids"
            ]
            / gold_count
            if gold_count
            else 0.0
        ),
        "mean_matched_iou": (
            matched_iou_sum
            / localization_matches
            if localization_matches
            else None
        ),
        "mean_matched_correct_class_probability": (
            matched_probability_sum
            / localization_matches
            if localization_matches
            else None
        ),
        "mean_matched_cross_entropy": (
            matched_nll_sum
            / localization_matches
            if localization_matches
            else None
        ),
    }

    return (
        metrics,
        prediction_rows,
        gold_rows,
    )


def write_tsv_gz(
    path: Path,
    *,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"No rows generated for {path}"
        )

    with gzip.open(
        path,
        mode="wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0]
            ),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main() -> None:
    options = parse_args()

    if options.token_budget <= 0:
        raise ValueError(
            "--token-budget must be positive"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    domains, class_by_domain = (
        read_selected_domains()
    )

    (
        model,
        checkpoint_state,
        device,
    ) = build_model(
        len(domains)
    )

    selection_path = (
        OUTPUT
        / "selected_decoding.json"
    )

    grid_path = (
        OUTPUT
        / "validation_decoding_grid.tsv"
    )

    selection = None

    if options.mode in {
        "validation",
        "all",
    }:
        validation_dataset = (
            build_dataset(
                "validation"
            )
        )

        (
            validation_gold,
            validation_ignored,
        ) = load_gold(
            "validation",
            class_by_domain=(
                class_by_domain
            ),
        )

        probabilities = (
            infer_validation_probabilities(
                model=model,
                dataset=(
                    validation_dataset
                ),
                device=device,
                token_budget=(
                    options.token_budget
                ),
                progress_every=(
                    options.progress_every
                ),
            )
        )

        selected, grid = (
            select_decoding(
                dataset=(
                    validation_dataset
                ),
                probabilities=(
                    probabilities
                ),
                gold=(
                    validation_gold
                ),
                ignored=(
                    validation_ignored
                ),
            )
        )

        with grid_path.open(
            mode="wt",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    grid[0]
                ),
                delimiter="\t",
                lineterminator="\n",
            )

            writer.writeheader()

            writer.writerows(
                sorted(
                    grid,
                    key=lambda row: (
                        -row["f1"]
                    ),
                )
            )

        selection = {
            "status": "COMPLETE",
            "selection_split": (
                "validation"
            ),
            "objective": (
                "localization_span_f1_at_iou_0.5"
            ),
            "selected": selected,
            "search": {
                "thresholds": (
                    THRESHOLDS
                ),
                "maximum_gaps": (
                    MAXIMUM_GAPS
                ),
                "minimum_span_length": (
                    MINIMUM_SPAN_LENGTH
                ),
            },
            "checkpoint_state": (
                checkpoint_state
            ),
            "grid": str(
                grid_path.resolve()
            ),
        }

        selection_path.write_text(
            json.dumps(
                selection,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                {
                    "event": (
                        "validation_selected"
                    ),
                    **selection,
                },
                indent=2,
                sort_keys=True,
            )
        )

        print(
            "JOINT_END_TO_END_"
            "VALIDATION_COMPLETE",
            flush=True,
        )

        if (
            options.mode
            == "validation"
        ):
            return

    if selection is None:
        if not selection_path.is_file():
            raise FileNotFoundError(
                "Missing validation selection: "
                f"{selection_path}"
            )

        selection = json.loads(
            selection_path.read_text(
                encoding="utf-8"
            )
        )

    selected = selection[
        "selected"
    ]

    test_dataset = (
        build_dataset(
            "test"
        )
    )

    (
        test_gold,
        test_ignored,
    ) = load_gold(
        "test",
        class_by_domain=(
            class_by_domain
        ),
    )

    (
        metrics,
        prediction_rows,
        gold_rows,
    ) = evaluate_test(
        model=model,
        dataset=test_dataset,
        device=device,
        domains=domains,
        gold=test_gold,
        ignored=test_ignored,
        selected=selected,
        token_budget=(
            options.token_budget
        ),
        progress_every=(
            options.progress_every
        ),
    )

    prediction_path = (
        OUTPUT
        / "test_predictions.tsv.gz"
    )

    gold_path = (
        OUTPUT
        / "test_gold_regions.tsv.gz"
    )

    write_tsv_gz(
        prediction_path,
        rows=prediction_rows,
    )

    write_tsv_gz(
        gold_path,
        rows=gold_rows,
    )

    summary = {
        "status": "COMPLETE",
        "checkpoint": str(
            CHECKPOINT.resolve()
        ),
        "checkpoint_state": (
            checkpoint_state
        ),
        "decoding_selected_on_validation": (
            selection
        ),
        "test_policy": {
            "minimum_iou": (
                MINIMUM_IOU
            ),
            "different_id_overlap_instances": (
                "excluded"
            ),
            "unaffected_instances_on_same_protein": (
                "retained"
            ),
            "unmatched_prediction_ignore_fraction": (
                IGNORE_FRACTION
            ),
        },
        "test_metrics": (
            metrics
        ),
        "outputs": {
            "selection": str(
                selection_path.resolve()
            ),
            "validation_grid": str(
                grid_path.resolve()
            ),
            "test_predictions": str(
                prediction_path.resolve()
            ),
            "test_gold_regions": str(
                gold_path.resolve()
            ),
        },
    }

    summary_path = (
        OUTPUT
        / "test_summary.json"
    )

    summary[
        "outputs"
    ][
        "summary"
    ] = str(
        summary_path.resolve()
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "JOINT_END_TO_END_"
        "TEST_COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
