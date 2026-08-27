"""Runtime utilities for joint InterPro curriculum training."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from peft import (
    get_peft_model_state_dict,
)
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)
from torch.optim import AdamW

from interpro_joint.engine import (
    move_stage1_batch,
    move_stage2_batch,
    stage1_loss_components,
    stage2_loss_components,
)
from interpro_joint.model import (
    JOINT_ADAPTER_NAME,
)


def update_early_stopping(
    *,
    current_score: float,
    best_score: float,
    patience_counter: int,
    active: bool,
    min_delta: float,
) -> tuple[float, int, bool]:
    """Update maximization-based early-stopping state."""
    if min_delta < 0:
        raise ValueError(
            "min_delta cannot be negative"
        )

    if not active:
        return (
            best_score,
            0,
            False,
        )

    improved = (
        current_score
        > best_score + min_delta
    )

    if improved:
        return (
            current_score,
            0,
            True,
        )

    return (
        best_score,
        patience_counter + 1,
        False,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_multiplier(
    *,
    step: int,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )

    if step < warmup_steps:
        return float(step + 1) / float(
            max(1, warmup_steps)
        )

    progress = (
        float(step - warmup_steps)
        / float(
            max(
                1,
                total_steps - warmup_steps,
            )
        )
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )


def build_optimizer(
    *,
    model,
    classifier_lr: float,
    lora_lr: float,
    weight_decay: float,
) -> AdamW:
    head_parameters = [
        parameter
        for module in (
            model.stage1_head,
            model.stage2_domain_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    adapter_parameters = [
        parameter
        for parameter
        in model.encoder.parameters()
        if parameter.requires_grad
    ]

    if not head_parameters:
        raise RuntimeError(
            "No trainable head parameters"
        )

    if not adapter_parameters:
        raise RuntimeError(
            "No trainable LoRA parameters"
        )

    head_ids = {
        id(parameter)
        for parameter in head_parameters
    }

    adapter_ids = {
        id(parameter)
        for parameter in adapter_parameters
    }

    if head_ids & adapter_ids:
        raise RuntimeError(
            "Optimizer parameter groups overlap"
        )

    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    observed_ids = (
        head_ids
        | adapter_ids
    )

    if expected_ids != observed_ids:
        raise RuntimeError(
            "Optimizer does not contain exactly "
            "the trainable model parameters"
        )

    return AdamW(
        [
            {
                "name": "heads",
                "params": head_parameters,
                "lr": classifier_lr,
            },
            {
                "name": "shared_lora",
                "params": adapter_parameters,
                "lr": lora_lr,
            },
        ],
        weight_decay=weight_decay,
    )


def gather_rng_states(
    accelerator: Accelerator,
) -> list[dict]:
    local_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(
                accelerator.device
            )
            if accelerator.device.type
            == "cuda"
            else None
        ),
    }

    if accelerator.num_processes == 1:
        return [local_state]

    gathered: list[
        dict | None
    ] = [
        None
        for _ in range(
            accelerator.num_processes
        )
    ]

    dist.all_gather_object(
        gathered,
        local_state,
    )

    return [
        state
        for state in gathered
        if state is not None
    ]


def restore_rng_state(
    state: dict,
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


def broadcast_object(
    accelerator: Accelerator,
    value,
):
    if accelerator.num_processes == 1:
        return value

    container = [
        value
        if accelerator.is_main_process
        else None
    ]

    dist.broadcast_object_list(
        container,
        src=0,
    )

    return container[0]


def append_jsonl(
    path: Path,
    record: dict,
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
        )


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    output_dir: Path,
    checkpoint_name: str,
    trainer_state: dict,
    configuration: dict,
) -> None:
    accelerator.wait_for_everyone()

    rng_states = gather_rng_states(
        accelerator
    )

    if accelerator.is_main_process:
        final_dir = (
            output_dir
            / checkpoint_name
        )

        temporary_dir = (
            output_dir
            / f".{checkpoint_name}.tmp"
        )

        if temporary_dir.exists():
            shutil.rmtree(
                temporary_dir
            )

        temporary_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        unwrapped = (
            accelerator.unwrap_model(
                model
            )
        )

        torch.save(
            unwrapped.stage1_head.state_dict(),
            temporary_dir
            / "stage1_head.pt",
        )

        torch.save(
            unwrapped
            .stage2_domain_head
            .state_dict(),
            temporary_dir
            / "stage2_domain_head.pt",
        )

        adapter_state = (
            get_peft_model_state_dict(
                unwrapped.encoder,
                adapter_name=(
                    JOINT_ADAPTER_NAME
                ),
            )
        )

        torch.save(
            adapter_state,
            temporary_dir
            / "shared_lora_adapter.pt",
        )

        torch.save(
            optimizer.state_dict(),
            temporary_dir
            / "optimizer.pt",
        )

        torch.save(
            scheduler.state_dict(),
            temporary_dir
            / "scheduler.pt",
        )

        torch.save(
            rng_states,
            temporary_dir
            / "rng_states.pt",
        )

        (
            temporary_dir
            / "trainer_state.json"
        ).write_text(
            json.dumps(
                trainer_state,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            temporary_dir
            / "configuration.json"
        ).write_text(
            json.dumps(
                configuration,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if final_dir.exists():
            shutil.rmtree(
                final_dir
            )

        os.rename(
            temporary_dir,
            final_dir,
        )

    accelerator.wait_for_everyone()


def load_lora_adapter_compat(
    *,
    encoder,
    adapter_state: dict[str, torch.Tensor],
    adapter_name: str,
) -> None:
    """
    Load a PEFT LoRA state.
    PEFT 0.19 expects a newer Transformers tensor-parallel module
    during set_peft_model_state_dict. The saved state removes the
    adapter-name component from LoRA keys, so restore that component
    and load the tensors directly through torch.nn.Module.
    """
    if not adapter_state:
        raise RuntimeError(
            "LoRA checkpoint state is empty"
        )

    current_state = encoder.state_dict()

    expected_adapter_keys = {
        key
        for key in current_state
        if (
            f".lora_A.{adapter_name}." in key
            or f".lora_B.{adapter_name}." in key
        )
    }

    if not expected_adapter_keys:
        raise RuntimeError(
            f"No active LoRA tensors found for adapter "
            f"{adapter_name!r}"
        )

    mapped_state: dict[
        str,
        torch.Tensor,
    ] = {}

    for saved_key, tensor in adapter_state.items():
        candidates = [
            saved_key,
        ]

        for module_name in (
            "lora_A",
            "lora_B",
        ):
            marker = f".{module_name}."

            if marker in saved_key:
                candidates.append(
                    saved_key.replace(
                        marker,
                        (
                            f".{module_name}."
                            f"{adapter_name}."
                        ),
                        1,
                    )
                )

        matching = [
            candidate
            for candidate in candidates
            if candidate in current_state
        ]

        if len(matching) != 1:
            raise RuntimeError(
                "Could not uniquely map saved LoRA key "
                f"{saved_key!r}; candidates={candidates}, "
                f"matches={matching}"
            )

        target_key = matching[0]

        if target_key in mapped_state:
            raise RuntimeError(
                "Multiple checkpoint tensors mapped to "
                f"{target_key!r}"
            )

        expected_shape = tuple(
            current_state[target_key].shape
        )

        observed_shape = tuple(
            tensor.shape
        )

        if observed_shape != expected_shape:
            raise RuntimeError(
                f"LoRA shape mismatch for {target_key}: "
                f"checkpoint={observed_shape}, "
                f"model={expected_shape}"
            )

        mapped_state[target_key] = tensor

    mapped_keys = set(
        mapped_state
    )

    missing_adapter_keys = (
        expected_adapter_keys
        - mapped_keys
    )

    unexpected_adapter_keys = (
        mapped_keys
        - expected_adapter_keys
    )

    if (
        missing_adapter_keys
        or unexpected_adapter_keys
    ):
        raise RuntimeError(
            "LoRA checkpoint coverage mismatch: "
            f"expected={len(expected_adapter_keys)}, "
            f"mapped={len(mapped_keys)}, "
            f"missing={sorted(missing_adapter_keys)[:10]}, "
            f"unexpected={sorted(unexpected_adapter_keys)[:10]}"
        )

    load_result = encoder.load_state_dict(
        mapped_state,
        strict=False,
    )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected directly loaded LoRA keys: "
            + ", ".join(
                load_result.unexpected_keys
            )
        )

    loaded_state = encoder.state_dict()

    for key, expected_tensor in mapped_state.items():
        observed_tensor = loaded_state[
            key
        ]

        if not torch.equal(
            observed_tensor.detach().cpu(),
            expected_tensor.detach().cpu(),
        ):
            raise RuntimeError(
                f"Reloaded LoRA tensor differs for {key}"
            )


def load_model_checkpoint(
    *,
    model,
    checkpoint_dir: Path,
) -> dict:
    trainer_state = json.loads(
        (
            checkpoint_dir
            / "trainer_state.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    model.stage1_head.load_state_dict(
        torch.load(
            checkpoint_dir
            / "stage1_head.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    (
        model.stage2_domain_head
        .load_state_dict(
            torch.load(
                checkpoint_dir
                / "stage2_domain_head.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    )

    adapter_state = torch.load(
        checkpoint_dir
        / "shared_lora_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )

    load_lora_adapter_compat(
        encoder=model.encoder,
        adapter_state=adapter_state,
        adapter_name=(
            JOINT_ADAPTER_NAME
        ),
    )

    return trainer_state


def load_training_state(
    *,
    accelerator: Accelerator,
    optimizer,
    scheduler,
    checkpoint_dir: Path,
) -> None:
    optimizer.load_state_dict(
        torch.load(
            checkpoint_dir
            / "optimizer.pt",
            map_location="cpu",
            weights_only=False,
        )
    )

    scheduler.load_state_dict(
        torch.load(
            checkpoint_dir
            / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
        )
    )

    rng_states = torch.load(
        checkpoint_dir
        / "rng_states.pt",
        map_location="cpu",
        weights_only=False,
    )

    if len(rng_states) != (
        accelerator.num_processes
    ):
        raise RuntimeError(
            "Checkpoint RNG world size "
            f"{len(rng_states)} != "
            f"{accelerator.num_processes}"
        )

    restore_rng_state(
        rng_states[
            accelerator.process_index
        ],
        accelerator.device,
    )


@torch.no_grad()
def validate_joint(
    *,
    accelerator: Accelerator,
    model,
    stage1_loader,
    stage2_loader,
    output_dir: Path,
    global_optimizer_step: int,
    expected_stage1_labels: int,
    expected_stage2_examples: int,
    class_count: int,
) -> dict:
    model.eval()

    stage1_loss_sum = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    stage1_count = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    stage2_loss_sum = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    stage2_count = torch.zeros(
        1,
        device=accelerator.device,
        dtype=torch.float64,
    )

    stage1_score_chunks: list[
        np.ndarray
    ] = []

    stage1_label_chunks: list[
        np.ndarray
    ] = []

    stage2_prediction_chunks: list[
        np.ndarray
    ] = []

    stage2_top5_chunks: list[
        np.ndarray
    ] = []

    stage2_label_chunks: list[
        np.ndarray
    ] = []

    for raw_batch in stage1_loader:
        batch = move_stage1_batch(
            raw_batch,
            accelerator.device,
        )

        residue_length = int(
            batch["labels"].shape[1]
        )

        logits = model(
            task="stage1",
            sequence_tokens=(
                batch["sequence_tokens"]
            ),
            residue_length=residue_length,
        )

        (
            loss_sum,
            valid_count,
        ) = stage1_loss_components(
            logits=logits,
            labels=batch["labels"],
            residue_mask=(
                batch["residue_mask"]
            ),
        )

        stage1_loss_sum += (
            loss_sum.detach().double()
        )

        stage1_count += (
            valid_count.detach().double()
        )

        mask = batch[
            "residue_mask"
        ]

        probabilities = (
            torch.sigmoid(
                logits.float()
            )
            .squeeze(-1)
        )

        labels = (
            batch["labels"]
            .squeeze(-1)
        )

        stage1_score_chunks.append(
            probabilities[
                mask
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        stage1_label_chunks.append(
            labels[
                mask
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.uint8,
                copy=False,
            )
        )

    for raw_batch in stage2_loader:
        batch = move_stage2_batch(
            raw_batch,
            accelerator.device,
        )

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

        labels = batch[
            "class_labels"
        ]

        (
            loss_sum,
            example_count,
        ) = stage2_loss_components(
            logits=logits,
            labels=labels,
        )

        stage2_loss_sum += (
            loss_sum.detach().double()
        )

        stage2_count += (
            example_count.detach().double()
        )

        predictions = logits.argmax(
            dim=-1
        )

        top5 = torch.topk(
            logits,
            k=min(
                5,
                logits.shape[-1],
            ),
            dim=-1,
        ).indices

        stage2_prediction_chunks.append(
            predictions
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int16,
                copy=False,
            )
        )

        stage2_top5_chunks.append(
            top5
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int16,
                copy=False,
            )
        )

        stage2_label_chunks.append(
            labels
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int16,
                copy=False,
            )
        )

    local_stage1_scores = (
        np.concatenate(
            stage1_score_chunks
        )
    )

    local_stage1_labels = (
        np.concatenate(
            stage1_label_chunks
        )
    )

    local_stage2_predictions = (
        np.concatenate(
            stage2_prediction_chunks
        )
    )

    local_stage2_top5 = (
        np.concatenate(
            stage2_top5_chunks
        )
    )

    local_stage2_labels = (
        np.concatenate(
            stage2_label_chunks
        )
    )

    cache_dir = (
        output_dir
        / ".validation_cache"
        / (
            "step_"
            f"{global_optimizer_step:08d}"
        )
    )

    if accelerator.is_main_process:
        if cache_dir.exists():
            shutil.rmtree(
                cache_dir
            )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    accelerator.wait_for_everyone()

    np.savez(
        cache_dir
        / (
            "rank_"
            f"{accelerator.process_index}.npz"
        ),
        stage1_scores=(
            local_stage1_scores
        ),
        stage1_labels=(
            local_stage1_labels
        ),
        stage2_predictions=(
            local_stage2_predictions
        ),
        stage2_top5=(
            local_stage2_top5
        ),
        stage2_labels=(
            local_stage2_labels
        ),
    )

    accelerator.wait_for_everyone()

    totals = accelerator.reduce(
        torch.cat(
            [
                stage1_loss_sum,
                stage1_count,
                stage2_loss_sum,
                stage2_count,
            ]
        ),
        reduction="sum",
    )

    metrics = None

    if accelerator.is_main_process:
        all_stage1_scores = []
        all_stage1_labels = []
        all_stage2_predictions = []
        all_stage2_top5 = []
        all_stage2_labels = []

        for rank in range(
            accelerator.num_processes
        ):
            rank_data = np.load(
                cache_dir
                / f"rank_{rank}.npz"
            )

            all_stage1_scores.append(
                rank_data[
                    "stage1_scores"
                ]
            )

            all_stage1_labels.append(
                rank_data[
                    "stage1_labels"
                ]
            )

            all_stage2_predictions.append(
                rank_data[
                    "stage2_predictions"
                ]
            )

            all_stage2_top5.append(
                rank_data[
                    "stage2_top5"
                ]
            )

            all_stage2_labels.append(
                rank_data[
                    "stage2_labels"
                ]
            )

        stage1_scores = np.concatenate(
            all_stage1_scores
        )

        stage1_labels = np.concatenate(
            all_stage1_labels
        )

        stage2_predictions = (
            np.concatenate(
                all_stage2_predictions
            )
        )

        stage2_top5 = np.concatenate(
            all_stage2_top5
        )

        stage2_labels = np.concatenate(
            all_stage2_labels
        )

        if stage1_scores.size != (
            expected_stage1_labels
        ):
            raise RuntimeError(
                "Stage 1 validation count "
                f"{stage1_scores.size:,} != "
                f"{expected_stage1_labels:,}"
            )

        if stage2_labels.size != (
            expected_stage2_examples
        ):
            raise RuntimeError(
                "Stage 2 validation count "
                f"{stage2_labels.size:,} != "
                f"{expected_stage2_examples:,}"
            )

        if not np.isfinite(
            stage1_scores
        ).all():
            raise RuntimeError(
                "Non-finite Stage 1 scores"
            )

        labels_array = np.arange(
            class_count
        )

        (
            macro_precision,
            macro_recall,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            stage2_labels,
            stage2_predictions,
            labels=labels_array,
            average="macro",
            zero_division=0,
        )

        stage1_auprc = float(
            average_precision_score(
                stage1_labels,
                stage1_scores,
            )
        )

        stage2_accuracy = float(
            np.mean(
                stage2_predictions
                == stage2_labels
            )
        )

        stage2_top5_accuracy = float(
            np.mean(
                np.any(
                    stage2_top5
                    == stage2_labels[
                        :,
                        None,
                    ],
                    axis=1,
                )
            )
        )

        stage1_bce = float(
            totals[0].item()
            / totals[1].item()
        )

        stage2_ce = float(
            totals[2].item()
            / totals[3].item()
        )

        joint_score = math.sqrt(
            max(
                stage1_auprc,
                0.0,
            )
            * max(
                float(macro_f1),
                0.0,
            )
        )

        metrics = {
            "global_optimizer_step": (
                global_optimizer_step
            ),
            "stage1_validation_bce": (
                stage1_bce
            ),
            "stage1_macro_auprc": (
                stage1_auprc
            ),
            "stage1_validation_labels": (
                int(stage1_scores.size)
            ),
            "stage2_validation_ce": (
                stage2_ce
            ),
            "stage2_accuracy": (
                stage2_accuracy
            ),
            "stage2_top5_accuracy": (
                stage2_top5_accuracy
            ),
            "stage2_macro_precision": (
                float(macro_precision)
            ),
            "stage2_macro_recall": (
                float(macro_recall)
            ),
            "stage2_macro_f1": (
                float(macro_f1)
            ),
            "stage2_validation_examples": (
                int(stage2_labels.size)
            ),
            "joint_score": (
                float(joint_score)
            ),
        }

        for name, value in (
            metrics.items()
        ):
            if (
                isinstance(
                    value,
                    float,
                )
                and not math.isfinite(
                    value
                )
            ):
                raise RuntimeError(
                    "Non-finite validation "
                    f"metric: {name}={value}"
                )

        shutil.rmtree(
            cache_dir
        )

    accelerator.wait_for_everyone()

    metrics = broadcast_object(
        accelerator,
        metrics,
    )

    model.train()

    return metrics
