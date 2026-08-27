"""Datasets and collators for joint Domain span/ID training."""

from __future__ import annotations
import csv
import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

import torch
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from torch.utils.data import Dataset


PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 2


def open_text(
    path: Path,
) -> TextIO:
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


def normalize_accession(
    raw: str,
) -> str:
    accession = raw.strip().split()[0]

    parts = accession.split("|")

    if (
        len(parts) >= 3
        and parts[0] in {"sp", "tr"}
    ):
        accession = parts[1]

    if not accession:
        raise ValueError(
            "Encountered an empty accession"
        )

    return accession


def read_fasta(
    path: str | Path,
) -> dict[str, str]:
    path = Path(path)

    sequences: dict[str, str] = {}

    current_accession: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        nonlocal current_accession
        nonlocal chunks

        if current_accession is None:
            return

        sequence = "".join(chunks).strip()

        if not sequence:
            raise RuntimeError(
                f"{path}: empty sequence for "
                f"{current_accession}"
            )

        if current_accession in sequences:
            raise RuntimeError(
                f"{path}: duplicate FASTA accession "
                f"{current_accession}"
            )

        sequences[current_accession] = sequence

        current_accession = None
        chunks = []

    with open_text(path) as handle:
        for line_number, raw in enumerate(
            handle,
            start=1,
        ):
            line = raw.strip()

            if not line:
                continue

            if line.startswith(">"):
                commit()

                current_accession = (
                    normalize_accession(
                        line[1:]
                    )
                )
                continue

            if current_accession is None:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    "sequence before first header"
                )

            chunks.append(line)

    commit()

    if not sequences:
        raise RuntimeError(
            f"No sequences found in {path}"
        )

    return sequences


def read_selected_domains(
    path: str | Path,
) -> tuple[
    tuple[str, ...],
    dict[str, int],
]:
    path = Path(path)

    domains = tuple(
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )

    if not domains:
        raise RuntimeError(
            f"{path}: no Domains were found"
        )

    if len(set(domains)) != len(domains):
        raise RuntimeError(
            f"{path}: duplicate Domain IDs"
        )

    class_by_domain = {
        interpro_id: class_index
        for class_index, interpro_id
        in enumerate(domains)
    }

    return domains, class_by_domain


@dataclass(
    frozen=True,
    slots=True,
)
class DomainSpan:
    start: int
    end: int


@dataclass(
    frozen=True,
    slots=True,
)
class Stage1ProteinRecord:
    protein_accession: str
    sequence_length: int
    spans: tuple[DomainSpan, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class Stage2PairRecord:
    protein_accession: str
    interpro_id: str
    class_index: int
    sequence_length: int
    spans: tuple[DomainSpan, ...]


def read_domain_annotations(
    *,
    path: str | Path,
    sequences: dict[str, str],
    selected_domains: set[str],
) -> dict[
    tuple[str, str],
    tuple[DomainSpan, ...],
]:
    path = Path(path)

    grouped: dict[
        tuple[str, str],
        list[DomainSpan],
    ] = defaultdict(list)

    with open_text(path) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        required = {
            "protein_accession",
            "interpro_id",
            "start",
            "end",
        }

        observed = set(
            reader.fieldnames or []
        )

        missing = required - observed

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

            interpro_id = (
                row["interpro_id"].strip()
            )

            if interpro_id not in selected_domains:
                raise RuntimeError(
                    f"{path}:{row_number}: "
                    f"unexpected Domain "
                    f"{interpro_id}"
                )

            sequence = sequences.get(
                accession
            )

            if sequence is None:
                raise RuntimeError(
                    f"{path}:{row_number}: "
                    f"{accession} is absent "
                    "from split FASTA"
                )

            start = int(row["start"])
            end = int(row["end"])

            if not (
                1
                <= start
                <= end
                <= len(sequence)
            ):
                raise RuntimeError(
                    f"{path}:{row_number}: "
                    f"invalid coordinates "
                    f"{accession} "
                    f"{start}-{end}; "
                    f"length={len(sequence)}"
                )

            grouped[
                accession,
                interpro_id,
            ].append(
                DomainSpan(
                    start=start,
                    end=end,
                )
            )

    return {
        key: tuple(
            sorted(
                spans,
                key=lambda span: (
                    span.start,
                    span.end,
                ),
            )
        )
        for key, spans in grouped.items()
    }


class JointStage1Dataset(Dataset):
    """
    Full-protein binary Domain-span dataset.

    One output channel marks residues belonging to at least one
    retained Domain annotation.
    """

    def __init__(
        self,
        *,
        fasta_path: str | Path,
        annotations_path: str | Path,
        selected_domains_path: str | Path,
    ) -> None:
        self.fasta_path = Path(
            fasta_path
        )

        self.annotations_path = Path(
            annotations_path
        )

        (
            self.selected_domains,
            self.class_by_domain,
        ) = read_selected_domains(
            selected_domains_path
        )

        self.sequences = read_fasta(
            self.fasta_path
        )

        pair_annotations = (
            read_domain_annotations(
                path=self.annotations_path,
                sequences=self.sequences,
                selected_domains=set(
                    self.selected_domains
                ),
            )
        )

        spans_by_accession: dict[
            str,
            list[DomainSpan],
        ] = defaultdict(list)

        for (
            accession,
            _interpro_id,
        ), spans in pair_annotations.items():
            spans_by_accession[
                accession
            ].extend(spans)

        self.records = tuple(
            Stage1ProteinRecord(
                protein_accession=accession,
                sequence_length=len(sequence),
                spans=tuple(
                    sorted(
                        spans_by_accession.get(
                            accession,
                            [],
                        ),
                        key=lambda span: (
                            span.start,
                            span.end,
                        ),
                    )
                ),
            )
            for accession, sequence
            in self.sequences.items()
        )

        self.sequence_lengths = [
            record.sequence_length
            for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def build_labels(
        self,
        record: Stage1ProteinRecord,
    ) -> torch.Tensor:
        labels = torch.zeros(
            (
                record.sequence_length,
                1,
            ),
            dtype=torch.float32,
        )

        for span in record.spans:
            labels[
                span.start - 1 : span.end,
                0,
            ] = 1.0

        return labels

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        record = self.records[index]

        return {
            "protein_accession": (
                record.protein_accession
            ),
            "sequence": self.sequences[
                record.protein_accession
            ],
            "sequence_length": (
                record.sequence_length
            ),
            "labels": self.build_labels(
                record
            ),
            "span_count": len(
                record.spans
            ),
        }


class JointStage2Dataset(Dataset):
    """
    Balanced protein-Domain pair dataset.

    All annotation segments belonging to a selected pair are retained
    and later combined into one visible target-token mask.
    """

    def __init__(
        self,
        *,
        fasta_path: str | Path,
        targets_path: str | Path,
        target_annotations_path: str | Path,
        selected_domains_path: str | Path,
    ) -> None:
        self.fasta_path = Path(
            fasta_path
        )

        self.targets_path = Path(
            targets_path
        )

        self.target_annotations_path = (
            Path(
                target_annotations_path
            )
        )

        (
            self.selected_domains,
            self.class_by_domain,
        ) = read_selected_domains(
            selected_domains_path
        )

        self.sequences = read_fasta(
            self.fasta_path
        )

        annotations = read_domain_annotations(
            path=self.target_annotations_path,
            sequences=self.sequences,
            selected_domains=set(
                self.selected_domains
            ),
        )

        records: list[
            Stage2PairRecord
        ] = []

        seen_pairs: set[
            tuple[str, str]
        ] = set()

        with open_text(
            self.targets_path
        ) as handle:
            reader = csv.DictReader(
                handle,
                delimiter="\t",
            )

            required = {
                "protein_accession",
                "interpro_id",
            }

            observed = set(
                reader.fieldnames or []
            )

            missing = required - observed

            if missing:
                raise RuntimeError(
                    f"{self.targets_path}: "
                    f"missing columns "
                    f"{sorted(missing)}"
                )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                accession = (
                    normalize_accession(
                        row[
                            "protein_accession"
                        ]
                    )
                )

                interpro_id = (
                    row["interpro_id"].strip()
                )

                pair = (
                    accession,
                    interpro_id,
                )

                if pair in seen_pairs:
                    raise RuntimeError(
                        f"{self.targets_path}:"
                        f"{row_number}: duplicate "
                        f"target pair {pair}"
                    )

                seen_pairs.add(pair)

                sequence = self.sequences.get(
                    accession
                )

                if sequence is None:
                    raise RuntimeError(
                        f"{self.targets_path}:"
                        f"{row_number}: "
                        f"{accession} absent "
                        "from split FASTA"
                    )

                if (
                    interpro_id
                    not in self.class_by_domain
                ):
                    raise RuntimeError(
                        f"{self.targets_path}:"
                        f"{row_number}: "
                        f"unexpected Domain "
                        f"{interpro_id}"
                    )

                spans = annotations.get(
                    pair
                )

                if not spans:
                    raise RuntimeError(
                        f"{self.targets_path}:"
                        f"{row_number}: target "
                        f"{pair} has no annotations"
                    )

                records.append(
                    Stage2PairRecord(
                        protein_accession=(
                            accession
                        ),
                        interpro_id=(
                            interpro_id
                        ),
                        class_index=(
                            self.class_by_domain[
                                interpro_id
                            ]
                        ),
                        sequence_length=(
                            len(sequence)
                        ),
                        spans=spans,
                    )
                )

        annotation_pairs = set(
            annotations
        )

        if annotation_pairs != seen_pairs:
            missing_annotations = (
                seen_pairs
                - annotation_pairs
            )

            extra_annotations = (
                annotation_pairs
                - seen_pairs
            )

            raise RuntimeError(
                "Balanced target/annotation "
                "pair mismatch: "
                f"missing={len(missing_annotations):,}, "
                f"extra={len(extra_annotations):,}"
            )

        self.records = tuple(records)

        self.sequence_lengths = [
            record.sequence_length
            for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        record = self.records[index]

        return {
            "protein_accession": (
                record.protein_accession
            ),
            "interpro_id": (
                record.interpro_id
            ),
            "class_index": (
                record.class_index
            ),
            "sequence": self.sequences[
                record.protein_accession
            ],
            "sequence_length": (
                record.sequence_length
            ),
            "spans": tuple(
                (
                    span.start,
                    span.end,
                )
                for span in record.spans
            ),
        }


class JointStage1Collator:
    """Collate full proteins with one binary residue-label channel."""

    def __init__(
        self,
        tokenizer: EsmSequenceTokenizer
        | None = None,
    ) -> None:
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else EsmSequenceTokenizer()
        )

    def encode(
        self,
        sequence: str,
    ) -> torch.Tensor:
        token_ids = self.tokenizer.encode(
            sequence,
            add_special_tokens=True,
        )

        encoded = torch.tensor(
            [
                int(token_id)
                for token_id in token_ids
            ],
            dtype=torch.long,
        )

        if encoded.numel() != (
            len(sequence) + 2
        ):
            raise RuntimeError(
                "Unexpected Stage 1 "
                "token count"
            )

        if encoded[0].item() != (
            BOS_TOKEN_ID
        ):
            raise RuntimeError(
                "Missing BOS token"
            )

        if encoded[-1].item() != (
            EOS_TOKEN_ID
        ):
            raise RuntimeError(
                "Missing EOS token"
            )

        return encoded

    def __call__(
        self,
        examples: list[
            dict[str, object]
        ],
    ) -> dict[str, object]:
        if not examples:
            raise ValueError(
                "Cannot collate an empty "
                "Stage 1 batch"
            )

        encoded = [
            self.encode(
                str(example["sequence"])
            )
            for example in examples
        ]

        lengths = [
            int(
                example[
                    "sequence_length"
                ]
            )
            for example in examples
        ]

        maximum_length = max(lengths)

        sequence_tokens = torch.full(
            (
                len(examples),
                maximum_length + 2,
            ),
            fill_value=PAD_TOKEN_ID,
            dtype=torch.long,
        )

        labels = torch.zeros(
            (
                len(examples),
                maximum_length,
                1,
            ),
            dtype=torch.float32,
        )

        residue_mask = torch.zeros(
            (
                len(examples),
                maximum_length,
            ),
            dtype=torch.bool,
        )

        for row, example in enumerate(
            examples
        ):
            sequence_tokens[
                row,
                : encoded[row].numel(),
            ] = encoded[row]

            example_labels = example[
                "labels"
            ]

            if not isinstance(
                example_labels,
                torch.Tensor,
            ):
                raise TypeError(
                    "Stage 1 labels must "
                    "be tensors"
                )

            expected = (
                lengths[row],
                1,
            )

            if tuple(
                example_labels.shape
            ) != expected:
                raise RuntimeError(
                    "Stage 1 label-shape "
                    "mismatch: "
                    f"expected={expected}, "
                    f"observed="
                    f"{tuple(example_labels.shape)}"
                )

            labels[
                row,
                : lengths[row],
            ] = example_labels

            residue_mask[
                row,
                : lengths[row],
            ] = True

        return {
            "accessions": [
                str(
                    example[
                        "protein_accession"
                    ]
                )
                for example in examples
            ],
            "lengths": torch.tensor(
                lengths,
                dtype=torch.long,
            ),
            "sequence_tokens": (
                sequence_tokens
            ),
            "labels": labels,
            "residue_mask": (
                residue_mask
            ),
        }


@dataclass(
    frozen=True,
    slots=True,
)
class JointStage2Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    class_labels: torch.Tensor
    target_token_mask: torch.Tensor
    protein_accessions: tuple[
        str,
        ...,
    ]
    interpro_ids: tuple[
        str,
        ...,
    ]


class JointStage2Collator:
    """
    Collate unchanged proteins and union all target segments into one
    visible target-token mask.
    """

    def __init__(
        self,
        tokenizer: EsmSequenceTokenizer
        | None = None,
    ) -> None:
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else EsmSequenceTokenizer()
        )

    def encode(
        self,
        sequence: str,
    ) -> torch.Tensor:
        token_ids = self.tokenizer.encode(
            sequence,
            add_special_tokens=True,
        )

        encoded = torch.tensor(
            [
                int(token_id)
                for token_id in token_ids
            ],
            dtype=torch.long,
        )

        if encoded.numel() != (
            len(sequence) + 2
        ):
            raise RuntimeError(
                "Unexpected Stage 2 "
                "token count"
            )

        if encoded[0].item() != (
            BOS_TOKEN_ID
        ):
            raise RuntimeError(
                "Missing BOS token"
            )

        if encoded[-1].item() != (
            EOS_TOKEN_ID
        ):
            raise RuntimeError(
                "Missing EOS token"
            )

        return encoded

    def __call__(
        self,
        examples: list[
            dict[str, object]
        ],
    ) -> JointStage2Batch:
        if not examples:
            raise ValueError(
                "Cannot collate an empty "
                "Stage 2 batch"
            )

        encoded = [
            self.encode(
                str(example["sequence"])
            )
            for example in examples
        ]

        maximum_token_length = max(
            tokens.numel()
            for tokens in encoded
        )

        input_ids = torch.full(
            (
                len(examples),
                maximum_token_length,
            ),
            fill_value=PAD_TOKEN_ID,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (
                len(examples),
                maximum_token_length,
            ),
            dtype=torch.bool,
        )

        target_token_mask = torch.zeros(
            (
                len(examples),
                maximum_token_length,
            ),
            dtype=torch.bool,
        )

        for row, example in enumerate(
            examples
        ):
            tokens = encoded[row]

            input_ids[
                row,
                : tokens.numel(),
            ] = tokens

            attention_mask[
                row,
                : tokens.numel(),
            ] = True

            sequence_length = int(
                example[
                    "sequence_length"
                ]
            )

            spans = example["spans"]

            if not isinstance(
                spans,
                tuple,
            ):
                raise TypeError(
                    "Stage 2 spans must "
                    "be a tuple"
                )

            for raw_span in spans:
                start, end = raw_span

                start = int(start)
                end = int(end)

                if not (
                    1
                    <= start
                    <= end
                    <= sequence_length
                ):
                    raise RuntimeError(
                        "Invalid Stage 2 "
                        f"coordinates: "
                        f"{start}-{end}; "
                        f"length={sequence_length}"
                    )

                # BOS occupies token index zero.
                # Residue coordinates remain identical
                # to their token indices.
                target_token_mask[
                    row,
                    start : end + 1,
                ] = True

            if not target_token_mask[
                row
            ].any():
                raise RuntimeError(
                    "Stage 2 target mask "
                    "is empty"
                )

            if target_token_mask[
                row,
                0,
            ]:
                raise RuntimeError(
                    "BOS was included in "
                    "target mask"
                )

            if target_token_mask[
                row,
                sequence_length + 1,
            ]:
                raise RuntimeError(
                    "EOS was included in "
                    "target mask"
                )

        return JointStage2Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            class_labels=torch.tensor(
                [
                    int(
                        example[
                            "class_index"
                        ]
                    )
                    for example in examples
                ],
                dtype=torch.long,
            ),
            target_token_mask=(
                target_token_mask
            ),
            protein_accessions=tuple(
                str(
                    example[
                        "protein_accession"
                    ]
                )
                for example in examples
            ),
            interpro_ids=tuple(
                str(
                    example[
                        "interpro_id"
                    ]
                )
                for example in examples
            ),
        )
