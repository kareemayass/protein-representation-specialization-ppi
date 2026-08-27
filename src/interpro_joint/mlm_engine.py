"""MLM-preserving optimizer and calibration engine."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn

from interpro_joint.data import (
    PAD_TOKEN_ID,
)
from interpro_joint.engine import (
    RestartableDistributedLoader,
    move_stage1_batch,
    move_stage2_batch,
    raw_window_count,
    stage1_loss_components,
    stage2_loss_components,
)
from interpro_joint.mlm import (
    GradientPairStats,
    capture_lora_gradients,
    compare_gradient_sets,
    make_mlm_copy,
    mlm_loss_components,
)


@dataclass(frozen=True)
class TaskCalibrationSummary:
    task: str
    batches: int
    target_gradient_ratio: float
    median_task_gradient_norm: float
    median_mlm_gradient_norm: float
    median_cosine_similarity: float
    median_lambda_candidate: float
    minimum_cosine_similarity: float
    maximum_cosine_similarity: float
    batch_statistics: tuple[
        dict[str, float],
        ...,
    ]

    def as_dict(
        self,
    ) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MLMCalibrationResult:
    mask_fraction: float
    target_gradient_ratio: float
    stage1: TaskCalibrationSummary
    stage2: TaskCalibrationSummary

    @property
    def stage1_lambda(
        self,
    ) -> float:
        return float(
            self.stage1
            .median_lambda_candidate
        )

    @property
    def stage2_lambda(
        self,
    ) -> float:
        return float(
            self.stage2
            .median_lambda_candidate
        )

    def as_dict(
        self,
    ) -> dict[str, object]:
        result = asdict(self)

        result["stage1_lambda"] = (
            self.stage1_lambda
        )

        result["stage2_lambda"] = (
            self.stage2_lambda
        )

        return result


@dataclass(frozen=True)
class MLMOptimizerStepResult:
    optimizer_step: int
    task: str
    phase: str

    local_task_loss_sum: float
    local_task_count: float
    global_task_count: float

    local_mlm_sequence_loss_sum: float
    local_mlm_sequence_count: float
    global_mlm_sequence_count: float

    local_mlm_token_loss_sum: float
    local_mlm_token_count: float

    lambda_mlm: float

    loader_epoch: int
    loader_batches_consumed: int

    def as_dict(
        self,
    ) -> dict[str, object]:
        return asdict(self)

    @property
    def local_loss_sum(
        self,
    ) -> float:
        """Compatibility alias for baseline trainer logging."""
        return self.local_task_loss_sum

    @property
    def local_example_count(
        self,
    ) -> float:
        """Compatibility alias for baseline trainer logging."""
        return self.local_task_count


def _capture_rng_state(
    device: torch.device,
) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": (
            torch.get_rng_state()
        ),
        "torch_cuda": (
            torch.cuda.get_rng_state(
                device
            )
            if device.type == "cuda"
            else None
        ),
    }


def _restore_rng_state(
    state: dict[str, object],
    device: torch.device,
) -> None:
    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch_cpu"]
    )

    if (
        device.type == "cuda"
        and state["torch_cuda"]
        is not None
    ):
        torch.cuda.set_rng_state(
            state["torch_cuda"],
            device=device,
        )


def _mlm_source(
    *,
    task: str,
    batch: dict[str, object],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    if task == "stage1":
        sequence_tokens = batch[
            "sequence_tokens"
        ]

        attention_mask = (
            sequence_tokens.ne(
                PAD_TOKEN_ID
            )
        )

        return (
            sequence_tokens,
            attention_mask,
        )

    if task == "stage2":
        return (
            batch["input_ids"],
            batch["attention_mask"],
        )

    raise ValueError(
        f"Unsupported task: {task!r}"
    )


def _task_forward_and_loss(
    *,
    task: str,
    model: nn.Module,
    batch: dict[str, object],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    if task == "stage1":
        residue_length = int(
            batch["labels"].shape[1]
        )

        logits = model(
            task="stage1",
            sequence_tokens=(
                batch[
                    "sequence_tokens"
                ]
            ),
            residue_length=(
                residue_length
            ),
        )

        return stage1_loss_components(
            logits=logits,
            labels=batch["labels"],
            residue_mask=(
                batch["residue_mask"]
            ),
        )

    if task == "stage2":
        logits = model(
            task="stage2",
            input_ids=(
                batch["input_ids"]
            ),
            attention_mask=(
                batch["attention_mask"]
            ),
            target_token_mask=(
                batch[
                    "target_token_mask"
                ]
            ),
        )

        return stage2_loss_components(
            logits=logits,
            labels=(
                batch["class_labels"]
            ),
        )

    raise ValueError(
        f"Unsupported task: {task!r}"
    )


def _move_batch(
    *,
    task: str,
    raw_batch,
    device: torch.device,
) -> dict[str, object]:
    if task == "stage1":
        return move_stage1_batch(
            raw_batch,
            device,
        )

    if task == "stage2":
        return move_stage2_batch(
            raw_batch,
            device,
        )

    raise ValueError(
        f"Unsupported task: {task!r}"
    )


def _local_raw_mlm_sequence_count(
    *,
    task: str,
    raw_window: list[Any],
) -> float:
    if task == "stage1":
        return float(
            sum(
                int(
                    batch[
                        "sequence_tokens"
                    ].shape[0]
                )
                for batch
                in raw_window
            )
        )

    if task == "stage2":
        return float(
            sum(
                int(
                    batch.input_ids.shape[0]
                )
                for batch
                in raw_window
            )
        )

    raise ValueError(
        f"Unsupported task: {task!r}"
    )


def _global_mean_scale(
    *,
    accelerator: Accelerator,
    local_count: torch.Tensor,
) -> tuple[
    torch.Tensor,
    float,
]:
    """
    Return the global count and multiplier for a local loss sum.

    Accelerator divides backward() losses by the configured gradient-
    accumulation count, and DDP averages gradients across ranks.
    Multiplying by world_size * grad_accum / global_count therefore
    yields the exact global mean gradient.
    """
    global_count = accelerator.reduce(
        local_count.detach().double(),
        reduction="sum",
    )

    if global_count.item() <= 0:
        raise RuntimeError(
            "Global loss denominator "
            "is not positive"
        )

    multiplier = (
        accelerator.num_processes
        * accelerator
        .gradient_accumulation_steps
        / float(
            global_count.item()
        )
    )

    return (
        global_count,
        multiplier,
    )


def _calibrate_one_batch(
    *,
    task: str,
    raw_batch,
    accelerator: Accelerator,
    model: nn.Module,
    mask_fraction: float,
    target_gradient_ratio: float,
) -> GradientPairStats:
    """
    Measure task and MLM gradients on the same distributed batch.

    No optimizer update is performed.
    """
    batch = _move_batch(
        task=task,
        raw_batch=raw_batch,
        device=accelerator.device,
    )

    model.zero_grad(
        set_to_none=True
    )

    (
        task_loss_sum,
        task_count,
    ) = _task_forward_and_loss(
        task=task,
        model=model,
        batch=batch,
    )

    (
        _global_task_count,
        task_multiplier,
    ) = _global_mean_scale(
        accelerator=accelerator,
        local_count=task_count,
    )

    accelerator.backward(
        task_loss_sum
        * task_multiplier
    )

    task_gradients = (
        capture_lora_gradients(
            model
        )
    )

    model.zero_grad(
        set_to_none=True
    )

    (
        source_tokens,
        attention_mask,
    ) = _mlm_source(
        task=task,
        batch=batch,
    )

    (
        masked_tokens,
        prediction_mask,
    ) = make_mlm_copy(
        sequence_tokens=(
            source_tokens
        ),
        attention_mask=(
            attention_mask
        ),
        mask_fraction=(
            mask_fraction
        ),
    )

    mlm_logits = model(
        task="mlm",
        sequence_tokens=(
            masked_tokens
        ),
        attention_mask=(
            attention_mask
        ),
    )

    (
        mlm_sequence_loss_sum,
        mlm_sequence_count,
        _mlm_token_loss_sum,
        _mlm_token_count,
    ) = mlm_loss_components(
        logits=mlm_logits,
        targets=source_tokens,
        prediction_mask=(
            prediction_mask
        ),
    )

    (
        _global_mlm_count,
        mlm_multiplier,
    ) = _global_mean_scale(
        accelerator=accelerator,
        local_count=(
            mlm_sequence_count
        ),
    )

    accelerator.backward(
        mlm_sequence_loss_sum
        * mlm_multiplier
    )

    mlm_gradients = (
        capture_lora_gradients(
            model
        )
    )

    model.zero_grad(
        set_to_none=True
    )

    return compare_gradient_sets(
        task_gradients=(
            task_gradients
        ),
        mlm_gradients=(
            mlm_gradients
        ),
        target_ratio=(
            target_gradient_ratio
        ),
    )


def _summarize_calibration(
    *,
    task: str,
    statistics: list[
        GradientPairStats
    ],
    target_gradient_ratio: float,
) -> TaskCalibrationSummary:
    if not statistics:
        raise RuntimeError(
            f"No calibration statistics "
            f"for {task}"
        )

    task_norms = [
        item.task_norm
        for item in statistics
    ]

    mlm_norms = [
        item.mlm_norm
        for item in statistics
    ]

    cosines = [
        item.cosine_similarity
        for item in statistics
    ]

    lambdas = [
        item.lambda_candidate
        for item in statistics
    ]

    return TaskCalibrationSummary(
        task=task,
        batches=len(statistics),
        target_gradient_ratio=(
            target_gradient_ratio
        ),
        median_task_gradient_norm=float(
            median(task_norms)
        ),
        median_mlm_gradient_norm=float(
            median(mlm_norms)
        ),
        median_cosine_similarity=float(
            median(cosines)
        ),
        median_lambda_candidate=float(
            median(lambdas)
        ),
        minimum_cosine_similarity=float(
            min(cosines)
        ),
        maximum_cosine_similarity=float(
            max(cosines)
        ),
        batch_statistics=tuple(
            item.as_dict()
            for item in statistics
        ),
    )


def calibrate_mlm_lambdas(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    stage1_loader,
    stage1_sampler,
    stage2_loader,
    stage2_sampler,
    batches_per_task: int,
    mask_fraction: float,
    target_gradient_ratio: float,
) -> MLMCalibrationResult:
    """
    Calibrate lambda_1 and lambda_2 before optimizer step zero.

    The first `batches_per_task` distributed batches are examined
    without parameter updates. Loader sampler state and RNG state are
    restored afterward, so production training still begins from the
    original epoch-zero data stream and random state.
    """
    if batches_per_task <= 0:
        raise ValueError(
            "batches_per_task must "
            "be positive"
        )

    if not 0.0 < mask_fraction < 1.0:
        raise ValueError(
            "mask_fraction must be "
            "in (0, 1)"
        )

    if not (
        0.0
        < target_gradient_ratio
        < 1.0
    ):
        raise ValueError(
            "target_gradient_ratio must "
            "be in (0, 1)"
        )

    rng_state = _capture_rng_state(
        accelerator.device
    )

    model.train()

    stage1_sampler.set_epoch(0)
    stage2_sampler.set_epoch(0)

    stage1_iterator = iter(
        stage1_loader
    )

    stage2_iterator = iter(
        stage2_loader
    )

    stage1_statistics: list[
        GradientPairStats
    ] = []

    stage2_statistics: list[
        GradientPairStats
    ] = []

    try:
        for _ in range(
            batches_per_task
        ):
            try:
                raw_batch = next(
                    stage1_iterator
                )
            except StopIteration as error:
                raise RuntimeError(
                    "Stage 1 loader does not "
                    "contain enough calibration "
                    "batches"
                ) from error

            stage1_statistics.append(
                _calibrate_one_batch(
                    task="stage1",
                    raw_batch=raw_batch,
                    accelerator=accelerator,
                    model=model,
                    mask_fraction=(
                        mask_fraction
                    ),
                    target_gradient_ratio=(
                        target_gradient_ratio
                    ),
                )
            )

        for _ in range(
            batches_per_task
        ):
            try:
                raw_batch = next(
                    stage2_iterator
                )
            except StopIteration as error:
                raise RuntimeError(
                    "Stage 2 loader does not "
                    "contain enough calibration "
                    "batches"
                ) from error

            stage2_statistics.append(
                _calibrate_one_batch(
                    task="stage2",
                    raw_batch=raw_batch,
                    accelerator=accelerator,
                    model=model,
                    mask_fraction=(
                        mask_fraction
                    ),
                    target_gradient_ratio=(
                        target_gradient_ratio
                    ),
                )
            )

    finally:
        model.zero_grad(
            set_to_none=True
        )

        # Production training starts from the same
        # sampler epoch and RNG state it would have
        # used without calibration.
        stage1_sampler.set_epoch(0)
        stage2_sampler.set_epoch(0)

        _restore_rng_state(
            rng_state,
            accelerator.device,
        )

    return MLMCalibrationResult(
        mask_fraction=mask_fraction,
        target_gradient_ratio=(
            target_gradient_ratio
        ),
        stage1=(
            _summarize_calibration(
                task="stage1",
                statistics=(
                    stage1_statistics
                ),
                target_gradient_ratio=(
                    target_gradient_ratio
                ),
            )
        ),
        stage2=(
            _summarize_calibration(
                task="stage2",
                statistics=(
                    stage2_statistics
                ),
                target_gradient_ratio=(
                    target_gradient_ratio
                ),
            )
        ),
    )


def run_optimizer_step_with_mlm(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    optimizer,
    scheduler,
    stage1_loader: (
        RestartableDistributedLoader
    ),
    stage2_loader: (
        RestartableDistributedLoader
    ),
    curriculum,
    global_optimizer_step: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    stage1_lambda: float,
    stage2_lambda: float,
    mask_fraction: float,
) -> MLMOptimizerStepResult:
    """
    Perform one original curriculum optimizer step with auxiliary MLM.

    For every task microbatch:
      1. run the normal unmodified InterPro input and backpropagate;
      2. create a masked copy of the same proteins;
      3. run the frozen pretrained ESMC sequence head and backpropagate
         lambda * MLM loss;
      4. perform exactly one optimizer update after accumulation.

    The two backward passes accumulate into the same LoRA gradients.
    No optimizer step occurs between the task and MLM branches.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps "
            "must be positive"
        )

    if (
        accelerator
        .gradient_accumulation_steps
        != gradient_accumulation_steps
    ):
        raise RuntimeError(
            "Accelerator and trainer "
            "gradient-accumulation "
            "settings differ"
        )

    if stage1_lambda < 0.0:
        raise ValueError(
            "stage1_lambda cannot "
            "be negative"
        )

    if stage2_lambda < 0.0:
        raise ValueError(
            "stage2_lambda cannot "
            "be negative"
        )

    decision = curriculum.decision(
        global_optimizer_step
    )

    task = decision.task

    lambda_mlm = (
        stage1_lambda
        if task == "stage1"
        else stage2_lambda
    )

    active_loader = (
        stage1_loader
        if task == "stage1"
        else stage2_loader
    )

    raw_window = (
        active_loader.next_window(
            gradient_accumulation_steps
        )
    )

    local_task_window_count = (
        torch.tensor(
            raw_window_count(
                task=task,
                raw_window=raw_window,
            ),
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    global_task_window_count = (
        accelerator.reduce(
            local_task_window_count,
            reduction="sum",
        )
    )

    if (
        global_task_window_count
        .item()
        <= 0
    ):
        raise RuntimeError(
            "Global task accumulation "
            "window is empty"
        )

    local_mlm_window_count = (
        torch.tensor(
            _local_raw_mlm_sequence_count(
                task=task,
                raw_window=raw_window,
            ),
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    global_mlm_window_count = (
        accelerator.reduce(
            local_mlm_window_count,
            reduction="sum",
        )
    )

    if (
        global_mlm_window_count
        .item()
        <= 0
    ):
        raise RuntimeError(
            "Global MLM accumulation "
            "window is empty"
        )

    local_task_loss_sum = (
        torch.zeros(
            1,
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    local_task_count_sum = (
        torch.zeros(
            1,
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    local_mlm_sequence_loss_sum = (
        torch.zeros(
            1,
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    local_mlm_sequence_count_sum = (
        torch.zeros(
            1,
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    local_mlm_token_loss_sum = (
        torch.zeros(
            1,
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    local_mlm_token_count_sum = (
        torch.zeros(
            1,
            device=(
                accelerator.device
            ),
            dtype=torch.float64,
        )
    )

    optimizer_step_performed = False

    for (
        microbatch_index,
        raw_batch,
    ) in enumerate(
        raw_window
    ):
        with accelerator.accumulate(
            model
        ):
            batch = _move_batch(
                task=task,
                raw_batch=raw_batch,
                device=(
                    accelerator.device
                ),
            )

            # --------------------------------------------------
            # Auxiliary MLM contribution FIRST.
            #
            # Keep this backward unsynchronized. Its LoRA
            # gradients accumulate locally and are then included
            # in the synchronized task backward below.
            #
            # This avoids performing two DDP reductions for one
            # microbatch while still allowing the task head
            # gradients to be synchronized normally.
            # --------------------------------------------------

            (
                source_tokens,
                attention_mask,
            ) = _mlm_source(
                task=task,
                batch=batch,
            )

            (
                masked_tokens,
                prediction_mask,
            ) = make_mlm_copy(
                sequence_tokens=(
                    source_tokens
                ),
                attention_mask=(
                    attention_mask
                ),
                mask_fraction=(
                    mask_fraction
                ),
            )

            with accelerator.no_sync(
                model
            ):
                mlm_logits = model(
                    task="mlm",
                    sequence_tokens=(
                        masked_tokens
                    ),
                    attention_mask=(
                        attention_mask
                    ),
                )

                (
                    mlm_sequence_loss_sum,
                    mlm_sequence_count,
                    mlm_token_loss_sum,
                    mlm_token_count,
                ) = mlm_loss_components(
                    logits=mlm_logits,
                    targets=source_tokens,
                    prediction_mask=(
                        prediction_mask
                    ),
                )

                mlm_scaled_loss = (
                    mlm_sequence_loss_sum
                    * accelerator.num_processes
                    * gradient_accumulation_steps
                    / global_mlm_window_count
                )

                accelerator.backward(
                    lambda_mlm
                    * mlm_scaled_loss
                )

            # --------------------------------------------------
            # Normal InterPro contribution SECOND.
            #
            # On the final accumulation microbatch this backward
            # performs the DDP synchronization. The synchronized
            # LoRA gradient therefore contains both the accumulated
            # MLM contribution and the task contribution.
            # --------------------------------------------------

            (
                task_loss_sum,
                task_count,
            ) = _task_forward_and_loss(
                task=task,
                model=model,
                batch=batch,
            )

            task_scaled_loss = (
                task_loss_sum
                * accelerator.num_processes
                * gradient_accumulation_steps
                / global_task_window_count
            )

            accelerator.backward(
                task_scaled_loss
            )

            if accelerator.sync_gradients:
                if (
                    microbatch_index
                    != (
                        gradient_accumulation_steps
                        - 1
                    )
                ):
                    raise RuntimeError(
                        "Accelerate synchronized "
                        "before the final "
                        "microbatch"
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

                optimizer_step_performed = (
                    True
                )

        local_task_loss_sum += (
            task_loss_sum
            .detach()
            .double()
        )

        local_task_count_sum += (
            task_count
            .detach()
            .double()
        )

        (
            local_mlm_sequence_loss_sum
        ) += (
            mlm_sequence_loss_sum
            .detach()
            .double()
        )

        (
            local_mlm_sequence_count_sum
        ) += (
            mlm_sequence_count
            .detach()
            .double()
        )

        local_mlm_token_loss_sum += (
            mlm_token_loss_sum
            .detach()
            .double()
        )

        local_mlm_token_count_sum += (
            mlm_token_count
            .detach()
            .double()
        )

    if not optimizer_step_performed:
        raise RuntimeError(
            "Optimizer step was not "
            "performed"
        )

    return MLMOptimizerStepResult(
        optimizer_step=(
            global_optimizer_step
        ),
        task=task,
        phase=decision.phase,

        local_task_loss_sum=float(
            local_task_loss_sum.item()
        ),
        local_task_count=float(
            local_task_count_sum.item()
        ),
        global_task_count=float(
            global_task_window_count.item()
        ),

        local_mlm_sequence_loss_sum=float(
            local_mlm_sequence_loss_sum
            .item()
        ),
        local_mlm_sequence_count=float(
            local_mlm_sequence_count_sum
            .item()
        ),
        global_mlm_sequence_count=float(
            global_mlm_window_count
            .item()
        ),

        local_mlm_token_loss_sum=float(
            local_mlm_token_loss_sum
            .item()
        ),
        local_mlm_token_count=float(
            local_mlm_token_count_sum
            .item()
        ),

        lambda_mlm=float(
            lambda_mlm
        ),

        loader_epoch=(
            active_loader.epoch
        ),
        loader_batches_consumed=(
            active_loader
            .batches_consumed
        ),
    )


def calibrate_task_lambda(
    *,
    task: str,
    accelerator: Accelerator,
    model: nn.Module,
    calibration_loader,
    calibration_sampler,
    batches: int,
    mask_fraction: float,
    target_gradient_ratio: float,
    sampler_epoch: int = 0,
) -> TaskCalibrationSummary:
    """
    Calibrate the fixed auxiliary-MLM coefficient for one task.

    This function is intended to be called only after that task has
    received its initial task-only stabilization updates.

    Calibration:
      * performs no optimizer or scheduler steps;
      * measures raw task and MLM gradients separately;
      * targets the requested MLM/task LoRA-gradient norm ratio;
      * records gradient cosine similarity for diagnosis only;
      * uses a dedicated calibration loader, so the production
        training loader is not consumed;
      * restores all RNG state afterward, so calibration does not
        perturb subsequent training randomness.
    """
    if task not in {
        "stage1",
        "stage2",
    }:
        raise ValueError(
            "task must be 'stage1' "
            f"or 'stage2'; observed {task!r}"
        )

    if batches <= 0:
        raise ValueError(
            "batches must be positive"
        )

    if not 0.0 < mask_fraction < 1.0:
        raise ValueError(
            "mask_fraction must be in (0, 1)"
        )

    if not (
        0.0
        < target_gradient_ratio
        < 1.0
    ):
        raise ValueError(
            "target_gradient_ratio must "
            "be in (0, 1)"
        )

    if sampler_epoch < 0:
        raise ValueError(
            "sampler_epoch cannot "
            "be negative"
        )

    rng_state = _capture_rng_state(
        accelerator.device
    )

    previous_sampler_epoch = int(
        getattr(
            calibration_sampler,
            "epoch",
            0,
        )
    )

    was_training = model.training

    statistics: list[
        GradientPairStats
    ] = []

    accelerator.wait_for_everyone()

    try:
        model.train()

        calibration_sampler.set_epoch(
            sampler_epoch
        )

        iterator = iter(
            calibration_loader
        )

        for batch_index in range(
            batches
        ):
            try:
                raw_batch = next(
                    iterator
                )
            except StopIteration as error:
                raise RuntimeError(
                    f"{task} calibration loader "
                    "does not contain enough "
                    f"batches for requested "
                    f"calibration count={batches}"
                ) from error

            statistics.append(
                _calibrate_one_batch(
                    task=task,
                    raw_batch=raw_batch,
                    accelerator=accelerator,
                    model=model,
                    mask_fraction=(
                        mask_fraction
                    ),
                    target_gradient_ratio=(
                        target_gradient_ratio
                    ),
                )
            )

    finally:
        model.zero_grad(
            set_to_none=True
        )

        calibration_sampler.set_epoch(
            previous_sampler_epoch
        )

        _restore_rng_state(
            rng_state,
            accelerator.device,
        )

        if not was_training:
            model.eval()

        accelerator.wait_for_everyone()

    summary = _summarize_calibration(
        task=task,
        statistics=statistics,
        target_gradient_ratio=(
            target_gradient_ratio
        ),
    )

    if not (
        math.isfinite(
            summary.median_lambda_candidate
        )
        and summary
        .median_lambda_candidate
        > 0.0
    ):
        raise RuntimeError(
            f"{task} produced invalid "
            "MLM lambda: "
            f"{summary.median_lambda_candidate}"
        )

    return summary
