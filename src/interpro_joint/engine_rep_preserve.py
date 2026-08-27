"""Distributed optimizer-step engine for joint curriculum training."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Literal
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn

from interpro_joint.representation_regularization import (
    reference_hidden_states,
    representation_regularization_components,
)


TaskName = Literal["stage1", "stage2"]


@dataclass(frozen=True)
class LoaderState:
    epoch: int
    batches_consumed: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizerStepResult:
    optimizer_step: int
    task: TaskName
    phase: str
    local_loss_sum: float
    local_example_count: float
    global_example_count: float
    local_rep_loss_sum: float
    local_rep_protein_count: float
    global_rep_protein_count: float
    loader_epoch: int
    loader_batches_consumed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RestartableDistributedLoader:
    """
    Maintain a restartable distributed DataLoader deterministically.

    Stage 1 and Stage 2 each own one instance and therefore maintain
    independent sampler epochs. DistributedTokenBatchSampler ensures
    every rank recieves the same number of batches.
    """

    def __init__(
        self,
        *,
        loader,
        sampler,
        initial_epoch: int = 0,
        initial_batches_consumed: int = 0,
    ) -> None:
        if initial_epoch < 0:
            raise ValueError(
                "initial_epoch cannot be negative"
            )

        if initial_batches_consumed < 0:
            raise ValueError(
                "initial_batches_consumed cannot be negative"
            )

        if len(loader) <= 0:
            raise ValueError(
                "Cannot cycle an empty DataLoader"
            )

        self.loader = loader
        self.sampler = sampler
        self.epoch = int(initial_epoch)
        self.batches_consumed = 0
        self.iterator = None

        self._reset_iterator()

        for _ in range(
            int(initial_batches_consumed)
        ):
            try:
                next(self.iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "Saved loader position exceeds the "
                    "current loader length"
                ) from error

            self.batches_consumed += 1

    def _reset_iterator(self) -> None:
        self.sampler.set_epoch(
            self.epoch
        )

        self.iterator = iter(
            self.loader
        )

    def next_batch(self):
        assert self.iterator is not None

        try:
            batch = next(
                self.iterator
            )
        except StopIteration:
            self.epoch += 1
            self.batches_consumed = 0
            self._reset_iterator()

            try:
                batch = next(
                    self.iterator
                )
            except StopIteration as error:
                raise RuntimeError(
                    "Restarted DataLoader produced no batches"
                ) from error

        self.batches_consumed += 1

        return batch

    def next_window(
        self,
        microbatch_count: int,
    ) -> list[Any]:
        if microbatch_count <= 0:
            raise ValueError(
                "microbatch_count must be positive"
            )

        return [
            self.next_batch()
            for _ in range(
                microbatch_count
            )
        ]

    def state_dict(self) -> dict[str, int]:
        return LoaderState(
            epoch=self.epoch,
            batches_consumed=(
                self.batches_consumed
            ),
        ).as_dict()


def move_stage1_batch(
    batch: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    return {
        "accessions": batch["accessions"],
        "lengths": batch["lengths"].to(
            device,
            non_blocking=True,
        ),
        "sequence_tokens": batch[
            "sequence_tokens"
        ].to(
            device,
            non_blocking=True,
        ),
        "labels": batch["labels"].to(
            device,
            non_blocking=True,
        ),
        "residue_mask": batch[
            "residue_mask"
        ].to(
            device,
            non_blocking=True,
        ),
    }


def move_stage2_batch(
    batch,
    device: torch.device,
) -> dict[str, object]:
    return {
        "input_ids": batch.input_ids.to(
            device,
            non_blocking=True,
        ),
        "attention_mask": (
            batch.attention_mask.to(
                device,
                non_blocking=True,
            )
        ),
        "class_labels": (
            batch.class_labels.to(
                device,
                non_blocking=True,
            )
        ),
        "target_token_mask": (
            batch.target_token_mask.to(
                device,
                non_blocking=True,
            )
        ),
        "protein_accessions": (
            batch.protein_accessions
        ),
        "interpro_ids": batch.interpro_ids,
    }


def stage1_loss_components(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    if logits.shape != labels.shape:
        raise ValueError(
            "Stage 1 logit/label mismatch: "
            f"{tuple(logits.shape)} != "
            f"{tuple(labels.shape)}"
        )

    if residue_mask.shape != logits.shape[:2]:
        raise ValueError(
            "Stage 1 residue-mask mismatch"
        )

    element_loss = (
        F.binary_cross_entropy_with_logits(
            logits.float(),
            labels.float(),
            reduction="none",
        )
    )

    mask = residue_mask.unsqueeze(
        -1
    ).to(element_loss.dtype)

    loss_sum = (
        element_loss * mask
    ).sum()

    valid_count = (
        residue_mask.sum().to(
            element_loss.dtype
        )
        * logits.shape[-1]
    )

    if valid_count.item() <= 0:
        raise RuntimeError(
            "Stage 1 batch has no valid labels"
        )

    return loss_sum, valid_count


def stage2_loss_components(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    if logits.ndim != 2:
        raise ValueError(
            "Stage 2 logits must be [batch, classes]"
        )

    if labels.ndim != 1:
        raise ValueError(
            "Stage 2 labels must be [batch]"
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Stage 2 batch-size mismatch"
        )

    loss_sum = F.cross_entropy(
        logits.float(),
        labels.long(),
        reduction="sum",
    )

    example_count = torch.tensor(
        float(labels.numel()),
        device=logits.device,
        dtype=loss_sum.dtype,
    )

    if example_count.item() <= 0:
        raise RuntimeError(
            "Stage 2 batch has no examples"
        )

    return loss_sum, example_count


def raw_window_count(
    *,
    task: TaskName,
    raw_window: list[Any],
) -> float:
    if task == "stage1":
        return float(
            sum(
                int(
                    batch[
                        "residue_mask"
                    ].sum().item()
                )
                * int(
                    batch["labels"].shape[-1]
                )
                for batch in raw_window
            )
        )

    if task == "stage2":
        return float(
            sum(
                int(
                    batch.class_labels.numel()
                )
                for batch in raw_window
            )
        )

    raise ValueError(
        f"Unsupported task: {task!r}"
    )


def raw_window_protein_count(
    *,
    task: TaskName,
    raw_window: list[Any],
) -> float:
    """
    Count sequence rows used by the representation loss.

    For Stage 2 this follows training-example sampling: if the same
    protein occurs in multiple target-region examples, each row counts
    separately, matching the downstream task sampling.
    """
    if task == "stage1":
        return float(
            sum(
                int(
                    batch[
                        "sequence_tokens"
                    ].shape[0]
                )
                for batch in raw_window
            )
        )

    if task == "stage2":
        return float(
            sum(
                int(
                    batch.input_ids.shape[0]
                )
                for batch in raw_window
            )
        )

    raise ValueError(
        f"Unsupported task: {task!r}"
    )


def run_optimizer_step(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    optimizer,
    scheduler,
    stage1_loader: RestartableDistributedLoader,
    stage2_loader: RestartableDistributedLoader,
    curriculum,
    global_optimizer_step: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    rep_reg_alpha: float,
) -> OptimizerStepResult:
    """
    Every microbatch in the accumulation window uses the same task.
    The loss is scaled to the exact global mean across all examples
    or valid residue labels on all ranks.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive"
        )

    if (
        accelerator.gradient_accumulation_steps
        != gradient_accumulation_steps
    ):
        raise RuntimeError(
            "Accelerator and trainer gradient-accumulation "
            "settings differ"
        )

    if rep_reg_alpha < 0:
        raise ValueError(
            "rep_reg_alpha cannot be negative"
        )

    decision = curriculum.decision(
        global_optimizer_step
    )

    task: TaskName = decision.task

    active_loader = (
        stage1_loader
        if task == "stage1"
        else stage2_loader
    )

    raw_window = active_loader.next_window(
        gradient_accumulation_steps
    )

    local_window_count = torch.tensor(
        raw_window_count(
            task=task,
            raw_window=raw_window,
        ),
        device=accelerator.device,
        dtype=torch.float64,
    )

    global_window_count = accelerator.reduce(
        local_window_count,
        reduction="sum",
    )

    if global_window_count.item() <= 0:
        raise RuntimeError(
            "Global accumulation window is empty"
        )

    local_rep_window_count = torch.tensor(
        raw_window_protein_count(
            task=task,
            raw_window=raw_window,
        ),
        device=accelerator.device,
        dtype=torch.float64,
    )

    global_rep_window_count = accelerator.reduce(
        local_rep_window_count,
        reduction="sum",
    )

    if global_rep_window_count.item() <= 0:
        raise RuntimeError(
            "Global representation window is empty"
        )

    local_loss_sum = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    local_count_sum = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    local_rep_loss_sum = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    local_rep_count_sum = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    optimizer_step_performed = False

    for microbatch_index, raw_batch in enumerate(
        raw_window
    ):
        with accelerator.accumulate(
            model
        ):
            if task == "stage1":
                batch = move_stage1_batch(
                    raw_batch,
                    accelerator.device,
                )

                residue_length = int(
                    batch["labels"].shape[1]
                )

                (
                    logits,
                    adapted_features,
                ) = model(
                    task="stage1",
                    sequence_tokens=(
                        batch["sequence_tokens"]
                    ),
                    residue_length=(
                        residue_length
                    ),
                    return_representation_features=True,
                )

                attention_mask = (
                    batch["sequence_tokens"].ne(
                        accelerator.unwrap_model(
                            model
                        ).encoder.tokenizer.pad_token_id
                    )
                )

                (
                    loss_sum,
                    example_count,
                ) = stage1_loss_components(
                    logits=logits,
                    labels=batch["labels"],
                    residue_mask=(
                        batch["residue_mask"]
                    ),
                )

            else:
                batch = move_stage2_batch(
                    raw_batch,
                    accelerator.device,
                )

                (
                    logits,
                    adapted_features,
                ) = model(
                    task="stage2",
                    input_ids=batch["input_ids"],
                    attention_mask=(
                        batch["attention_mask"]
                    ),
                    target_token_mask=(
                        batch["target_token_mask"]
                    ),
                    return_representation_features=True,
                )

                attention_mask = batch[
                    "attention_mask"
                ]

                (
                    loss_sum,
                    example_count,
                ) = stage2_loss_components(
                    logits=logits,
                    labels=batch["class_labels"],
                )

            reference_features = reference_hidden_states(
                encoder=(
                    accelerator.unwrap_model(
                        model
                    ).encoder
                ),
                sequence_tokens=(
                    batch["sequence_tokens"]
                    if task == "stage1"
                    else batch["input_ids"]
                ),
                attention_mask=attention_mask,
            )

            (
                rep_loss_sum,
                rep_protein_count,
            ) = representation_regularization_components(
                adapted_features=adapted_features,
                reference_features=reference_features,
                attention_mask=attention_mask,
            )

            # Accelerate divides backward loss by the gradient-
            # accumulation count, while DDP averages gradients
            # across ranks.
            scaled_task_loss = (
                loss_sum
                * accelerator.num_processes
                * gradient_accumulation_steps
                / global_window_count
            )

            scaled_rep_loss = (
                rep_loss_sum
                * accelerator.num_processes
                * gradient_accumulation_steps
                / global_rep_window_count
            )

            scaled_loss = (
                scaled_task_loss
                + rep_reg_alpha
                * scaled_rep_loss
            )

            accelerator.backward(
                scaled_loss
            )

            if accelerator.sync_gradients:
                if (
                    microbatch_index
                    != gradient_accumulation_steps - 1
                ):
                    raise RuntimeError(
                        "Accelerate synchronized before the final "
                        "microbatch in the optimizer window"
                    )

                accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_step_performed = True

        local_loss_sum += (
            loss_sum.detach().double()
        )

        local_count_sum += (
            example_count.detach().double()
        )

        local_rep_loss_sum += (
            rep_loss_sum.detach().double()
        )

        local_rep_count_sum += (
            rep_protein_count.detach().double()
        )

    if not optimizer_step_performed:
        raise RuntimeError(
            "Optimizer step was not performed"
        )

    return OptimizerStepResult(
        optimizer_step=(
            global_optimizer_step
        ),
        task=task,
        phase=decision.phase,
        local_loss_sum=float(
            local_loss_sum.item()
        ),
        local_example_count=float(
            local_count_sum.item()
        ),
        global_example_count=float(
            global_window_count.item()
        ),
        local_rep_loss_sum=float(
            local_rep_loss_sum.item()
        ),
        local_rep_protein_count=float(
            local_rep_count_sum.item()
        ),
        global_rep_protein_count=float(
            global_rep_window_count.item()
        ),
        loader_epoch=active_loader.epoch,
        loader_batches_consumed=(
            active_loader.batches_consumed
        ),
    )
