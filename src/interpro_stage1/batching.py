"""Length-aware batching and collation for InterPro Stage 1."""

from __future__ import annotations
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import torch
from torch.utils.data import Sampler

PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 2
NUMBER_OF_LABELS = 2


def padded_token_cost(batch: Sequence[int], sequence_lengths: Sequence[int],) -> int:
    """
    Return padded ESMC token cost for one batch, with each sequence having length L + 2.
    (ESMC adds BOS and EOS)
    """
    if not batch:
        return 0

    # Find the longest tokenized sequence in the batch.
    maximum_token_length = max(sequence_lengths[index] + 2 for index in batch)

    return len(batch) * maximum_token_length


def _split_batches_until_divisible(batches: list[list[int]], world_size: int,) -> list[list[int]]:
    """
    Split existing batches until the batch count is divisible by world size.

    Splitting preserves every protein exactly once and cannot increase the
    padded-token cost of either resulting batch.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    batches = [list(batch) for batch in batches]


    while len(batches) % world_size != 0:

        candidates = [
            (len(batch), batch_index)
            for batch_index, batch in enumerate(batches)
            if len(batch) > 1
        ]

        if not candidates:
            raise RuntimeError(
                "Cannot make batch count divisible by world size: "
                "all batches contain only one protein"
            )


        # Select the largest splittable batch and remove it for splitting.
        _, split_index = max(candidates)
        batch = batches.pop(split_index)

        midpoint = len(batch) // 2

        left = batch[:midpoint]
        right = batch[midpoint:]

        if not left or not right:
            raise RuntimeError("Internal batch split produced an empty batch")

        batches.insert(split_index, right)
        batches.insert(split_index, left)

    return batches


def build_token_budget_batches(
    sequence_lengths: Sequence[int],
    token_budget: int,
    *,
    seed: int,
    epoch: int,
    world_size: int,
    bucket_size: int = 512,
    shuffle: bool = True,
    allow_oversize_singletons: bool = False,
    batch_count_multiple: int | None = None,
) -> list[list[int]]:
    """
    Build deterministic length-aware batches.
    The final number of batches is made divisible by world_size by splitting existing batches.
    """

    if not sequence_lengths:
        raise ValueError("sequence_lengths is empty")

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    if world_size <= 0:
        raise ValueError("world_size must be positive")

    for index, length in enumerate(sequence_lengths):
        if length <= 0:
            raise ValueError(
                f"Non-positive sequence length at index {index}: {length}"
            )

        if (
            length + 2 > token_budget
            and not allow_oversize_singletons
        ):
            raise ValueError(
                f"Protein index {index} requires {length + 2} tokens, "
                f"which exceeds token budget {token_budget}"
            )

    rng = random.Random(seed + epoch)

    indices = list(range(len(sequence_lengths)))
    indices.sort(key=lambda index: sequence_lengths[index])

    buckets = [
        indices[start : start + bucket_size]
        for start in range(0, len(indices), bucket_size)
    ]

    if shuffle:
        for bucket in buckets:
            rng.shuffle(bucket)

        rng.shuffle(buckets)

    ordered_indices = [
        index
        for bucket in buckets
        for index in bucket
    ]

    # Greedily pack proteins into batches without exceeding the budget.
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_maximum_token_length = 0

    for index in ordered_indices:
        token_length = sequence_lengths[index] + 2

        if token_length > token_budget:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_maximum_token_length = 0

            batches.append([index])
            continue

        proposed_maximum = max(
            current_maximum_token_length,
            token_length,
        )

        proposed_cost = (
            len(current_batch) + 1
        ) * proposed_maximum

        if current_batch and proposed_cost > token_budget:
            batches.append(current_batch)
            current_batch = [index]
            current_maximum_token_length = token_length
        else:
            current_batch.append(index)
            current_maximum_token_length = proposed_maximum

    if current_batch:
        batches.append(current_batch)

    required_multiple = (
        world_size
        if batch_count_multiple is None
        else int(batch_count_multiple)
    )

    if required_multiple <= 0:
        raise ValueError(
            "batch_count_multiple must be positive"
        )

    if required_multiple % world_size != 0:
        raise ValueError(
            "batch_count_multiple must be divisible by world_size"
        )

    batches = _split_batches_until_divisible(
        batches=batches,
        world_size=required_multiple,
    )

    for batch_index, batch in enumerate(batches):
        cost = padded_token_cost(
            batch=batch,
            sequence_lengths=sequence_lengths,
        )

        if cost > token_budget:
            is_valid_oversize_singleton = (
                allow_oversize_singletons
                and len(batch) == 1
                and sequence_lengths[batch[0]] + 2 > token_budget
            )

            if not is_valid_oversize_singleton:
                raise RuntimeError(
                    f"Batch {batch_index} exceeds token budget: "
                    f"{cost} > {token_budget}"
                )

    flattened = [
        index
        for batch in batches
        for index in batch
    ]

    if len(flattened) != len(sequence_lengths):
        raise RuntimeError(
            "Batch construction changed the number of proteins"
        )

    if len(set(flattened)) != len(sequence_lengths):
        raise RuntimeError(
            "Batch construction duplicated one or more proteins"
        )

    if set(flattened) != set(range(len(sequence_lengths))):
        raise RuntimeError(
            "Batch construction omitted one or more proteins"
        )

    return batches


class DistributedTokenBatchSampler(Sampler[list[int]]):
    """
    Rank-specific view of deterministic global token-budget batches.

    All ranks receive exactly the same number of batches.
    """

    def __init__(
        self,
        sequence_lengths: Sequence[int],
        token_budget: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        bucket_size: int = 512,
        shuffle: bool = True,
        allow_oversize_singletons: bool = False,
        batch_count_multiple: int | None = None,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank must be in [0, {world_size}), observed {rank}"
            )

        self.sequence_lengths = list(sequence_lengths)
        self.token_budget = token_budget
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.allow_oversize_singletons = (
            allow_oversize_singletons
        )
        self.batch_count_multiple = batch_count_multiple
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_batches(self) -> list[list[int]]:
        global_batches = build_token_budget_batches(
            sequence_lengths=self.sequence_lengths,
            token_budget=self.token_budget,
            seed=self.seed,
            epoch=self.epoch,
            world_size=self.world_size,
            bucket_size=self.bucket_size,
            shuffle=self.shuffle,
            allow_oversize_singletons=(
                self.allow_oversize_singletons
            ),
            batch_count_multiple=(
                self.batch_count_multiple
            ),
        )

        # Each rank gets the same number of batches
        return global_batches[
            self.rank :: self.world_size
        ]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._rank_batches()

    def __len__(self) -> int:
        return len(self._rank_batches())


@dataclass(frozen=True)
class Stage1Batch:
    accessions: list[str] # Example: ["P12345", "Q68721"]
    lengths: torch.Tensor
    sequence_tokens: torch.Tensor
    labels: torch.Tensor
    residue_mask: torch.Tensor

    def as_dict(self) -> dict[str, object]:
        return {
            "accessions": self.accessions,
            "lengths": self.lengths,
            "sequence_tokens": self.sequence_tokens,
            "labels": self.labels,
            "residue_mask": self.residue_mask,
        }


class Stage1Collator:
    """Tokenize and pad Stage 1 examples."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        examples: list[dict],
    ) -> dict[str, object]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        accessions: list[str] = []
        lengths: list[int] = []
        encoded_sequences: list[torch.Tensor] = []

        for example in examples:
            sequence = example["sequence"]
            accession = example["protein_accession"]
            sequence_length = int(example["sequence_length"])

            if len(sequence) != sequence_length:
                raise ValueError(
                    f"Sequence-length mismatch for {accession}: "
                    f"sequence={len(sequence)}, record={sequence_length}"
                )

            token_ids = self.tokenizer.encode(
                sequence,
                add_special_tokens=True,
            )

            encoded = torch.tensor(
                token_ids,
                dtype=torch.long,
            )

            expected_token_length = sequence_length + 2

            if encoded.numel() != expected_token_length:
                raise RuntimeError(
                    f"Unexpected token count for {accession}: "
                    f"expected={expected_token_length}, "
                    f"observed={encoded.numel()}"
                )

            if encoded[0].item() != BOS_TOKEN_ID:
                raise RuntimeError(
                    f"Missing BOS token for {accession}"
                )

            if encoded[-1].item() != EOS_TOKEN_ID:
                raise RuntimeError(
                    f"Missing EOS token for {accession}"
                )

            accessions.append(accession)
            lengths.append(sequence_length)
            encoded_sequences.append(encoded)

        batch_size = len(examples)
        maximum_length = max(lengths)
        maximum_token_length = maximum_length + 2

        sequence_tokens = torch.full(
            (batch_size, maximum_token_length),
            fill_value=PAD_TOKEN_ID,
            dtype=torch.long,
        )

        labels = torch.zeros(
            (
                batch_size,
                maximum_length,
                NUMBER_OF_LABELS,
            ),
            dtype=torch.float32,
        )

        residue_mask = torch.zeros(
            (batch_size, maximum_length),
            dtype=torch.bool,
        )

        for batch_index, example in enumerate(examples):
            sequence_length = lengths[batch_index]
            encoded = encoded_sequences[batch_index]

            sequence_tokens[
                batch_index,
                : encoded.numel(),
            ] = encoded

            example_labels = example["labels"].to(
                dtype=torch.float32
            )

            expected_label_shape = (
                sequence_length,
                NUMBER_OF_LABELS,
            )

            if tuple(example_labels.shape) != expected_label_shape:
                raise ValueError(
                    f"Label-shape mismatch for "
                    f"{accessions[batch_index]}: "
                    f"expected={expected_label_shape}, "
                    f"observed={tuple(example_labels.shape)}"
                )

            labels[
                batch_index,
                :sequence_length,
                :,
            ] = example_labels

            residue_mask[
                batch_index,
                :sequence_length,
            ] = True

        return Stage1Batch(
            accessions=accessions,
            lengths=torch.tensor(
                lengths,
                dtype=torch.long,
            ),
            sequence_tokens=sequence_tokens,
            labels=labels,
            residue_mask=residue_mask,
        ).as_dict()
