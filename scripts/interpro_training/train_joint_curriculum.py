#!/usr/bin/env python3
"""Resumable four-GPU joint InterPro curriculum trainer."""

from __future__ import annotations
import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
)
from esm.models.esmc import ESMC
from torch.optim.lr_scheduler import (
    LambdaLR,
)
from torch.utils.data import (
    DataLoader,
    Subset,
)


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from interpro_joint.curriculum import (  # noqa: E402
    JointCurriculum,
)
from interpro_joint.data import (  # noqa: E402
    JointStage1Collator,
    JointStage1Dataset,
    JointStage2Collator,
    JointStage2Dataset,
    read_selected_domains,
)
from interpro_joint.engine import (  # noqa: E402
    RestartableDistributedLoader,
    run_optimizer_step,
)
from interpro_joint.model import (  # noqa: E402
    build_joint_model,
)
from interpro_joint.runtime import (  # noqa: E402
    append_jsonl,
    broadcast_object,
    build_optimizer,
    cosine_multiplier,
    load_model_checkpoint,
    load_training_state,
    save_checkpoint,
    seed_everything,
    update_early_stopping,
    validate_joint,
)
from interpro_stage1.batching import (  # noqa: E402
    DistributedTokenBatchSampler,
)
from interpro_stage1.config import (  # noqa: E402
    ESMC_MODEL_NAME,
)


DEFAULT_SPLIT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "top100_domain_pipeline_v1"
    / "splits_top64_length_filtered_v1"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--total-optimizer-steps",
        type=int,
        default=40_000,
    )

    parser.add_argument(
        "--validate-every",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--train-token-budget",
        type=int,
        default=4_096,
    )

    parser.add_argument(
        "--validation-token-budget",
        type=int,
        default=4_096,
    )

    parser.add_argument(
        "--bucket-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--lora-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--stage1-head",
        choices=[
            "linear",
            "mlp",
        ],
        default="mlp",
    )

    parser.add_argument(
        "--classifier-dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=47,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--resume",
        choices=[
            "auto",
            "none",
        ],
        default="auto",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--max-train-stage1",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-train-stage2",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-validation-stage1",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-validation-stage2",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--runtime-buffer-minutes",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    positive = {
        "total_optimizer_steps": (
            args.total_optimizer_steps
        ),
        "validate_every": (
            args.validate_every
        ),
        "log_every": args.log_every,
        "gradient_accumulation_steps": (
            args
            .gradient_accumulation_steps
        ),
        "train_token_budget": (
            args.train_token_budget
        ),
        "validation_token_budget": (
            args
            .validation_token_budget
        ),
        "bucket_size": args.bucket_size,
        "expected_world_size": (
            args.expected_world_size
        ),
    }

    for name, value in positive.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be positive"
            )

    if (
        args.total_optimizer_steps
        % 40
        != 0
    ):
        raise ValueError(
            "total_optimizer_steps must "
            "be divisible by 40"
        )

    if not (
        0.0
        <= args.warmup_fraction
        < 1.0
    ):
        raise ValueError(
            "warmup_fraction must be "
            "in [0, 1)"
        )

    if args.max_epochs < 0:
        raise ValueError(
            "max_epochs cannot be negative"
        )

    if args.patience <= 0:
        raise ValueError(
            "patience must be positive"
        )

    if args.min_delta < 0:
        raise ValueError(
            "min_delta cannot be negative"
        )

    if args.max_epochs > 0:
        expected_steps = (
            args.max_epochs
            * args.validate_every
        )

        if (
            args.total_optimizer_steps
            != expected_steps
        ):
            raise ValueError(
                "total_optimizer_steps must equal "
                "max_epochs * validate_every; "
                f"observed "
                f"{args.total_optimizer_steps} != "
                f"{expected_steps}"
            )

    if args.runtime_buffer_minutes < 0:
        raise ValueError(
            "runtime_buffer_minutes "
            "cannot be negative"
        )


def subset_dataset(
    *,
    dataset,
    maximum: int,
    seed: int,
):
    lengths = list(
        dataset.sequence_lengths
    )

    if maximum <= 0 or maximum >= len(
        dataset
    ):
        return dataset, lengths

    rng = random.Random(seed)

    indices = rng.sample(
        range(len(dataset)),
        maximum,
    )

    indices.sort()

    subset = Subset(
        dataset,
        indices,
    )

    subset_lengths = [
        lengths[index]
        for index in indices
    ]

    return subset, subset_lengths


def build_sampler(
    *,
    lengths: list[int],
    token_budget: int,
    accelerator: Accelerator,
    seed: int,
    bucket_size: int,
    shuffle: bool,
    gradient_accumulation_steps: int,
):
    multiple = (
        accelerator.num_processes
        * gradient_accumulation_steps
        if shuffle
        else accelerator.num_processes
    )

    return DistributedTokenBatchSampler(
        sequence_lengths=lengths,
        token_budget=token_budget,
        rank=(
            accelerator.process_index
        ),
        world_size=(
            accelerator.num_processes
        ),
        seed=seed,
        bucket_size=bucket_size,
        shuffle=shuffle,
        allow_oversize_singletons=True,
        batch_count_multiple=multiple,
    )


def trainer_state(
    *,
    completed_steps: int,
    stage1_loader,
    stage2_loader,
    best_stage1_auprc: float,
    best_stage2_macro_f1: float,
    best_joint_score: float,
    validation_index: int,
    latest_metrics: dict | None,
    completed_epochs: int,
    patience_counter: int,
    best_patience_joint_score: float,
    stopped_early: bool,
) -> dict:
    return {
        "global_optimizer_step": (
            completed_steps
        ),
        "stage1_loader_state": (
            stage1_loader.state_dict()
        ),
        "stage2_loader_state": (
            stage2_loader.state_dict()
        ),
        "best_stage1_macro_auprc": (
            best_stage1_auprc
        ),
        "best_stage2_macro_f1": (
            best_stage2_macro_f1
        ),
        "best_joint_score": (
            best_joint_score
        ),
        "validation_index": (
            validation_index
        ),
        "latest_validation_metrics": (
            latest_metrics
        ),
        "completed_epochs": (
            completed_epochs
        ),
        "patience_counter": (
            patience_counter
        ),
        "best_patience_joint_score": (
            best_patience_joint_score
        ),
        "stopped_early": (
            stopped_early
        ),
        "saved_at_utc": utc_now(),
    }


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    ddp_kwargs = (
        DistributedDataParallelKwargs(
            find_unused_parameters=True,
        )
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=(
            args
            .gradient_accumulation_steps
        ),
        kwargs_handlers=[
            ddp_kwargs
        ],
    )

    if accelerator.num_processes != (
        args.expected_world_size
    ):
        raise RuntimeError(
            "Expected "
            f"{args.expected_world_size} "
            "processes, observed "
            f"{accelerator.num_processes}"
        )

    args.split_dir = (
        args.split_dir.resolve()
    )

    args.output_dir = (
        args.output_dir.resolve()
    )

    if (
        args.overwrite
        and accelerator.is_main_process
        and args.output_dir.exists()
    ):
        shutil.rmtree(
            args.output_dir
        )

    accelerator.wait_for_everyone()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_dir = (
        args.output_dir
        / "latest"
    )

    if (
        args.resume == "none"
        and not args.overwrite
        and any(
            args.output_dir.iterdir()
        )
    ):
        raise RuntimeError(
            "Output directory is not empty. "
            "Use --resume auto or --overwrite."
        )

    seed_everything(
        args.seed
    )

    split_dir = args.split_dir

    selected_domains_path = (
        split_dir
        / "selected_domains.txt"
    )

    (
        selected_domains,
        _class_by_domain,
    ) = read_selected_domains(
        selected_domains_path
    )

    domain_class_count = len(
        selected_domains
    )

    if domain_class_count != 64:
        raise RuntimeError(
            "Expected 64 retained Domains, "
            f"observed {domain_class_count}"
        )

    accelerator.print(
        "Loading joint datasets"
    )

    train_stage1_full = (
        JointStage1Dataset(
            fasta_path=(
                split_dir
                / "train.fasta"
            ),
            annotations_path=(
                split_dir
                / "train_annotations.tsv.gz"
            ),
            selected_domains_path=(
                selected_domains_path
            ),
        )
    )

    train_stage2_full = (
        JointStage2Dataset(
            fasta_path=(
                split_dir
                / "train.fasta"
            ),
            targets_path=(
                split_dir
                / (
                    "train_balanced_"
                    "domain_targets.tsv.gz"
                )
            ),
            target_annotations_path=(
                split_dir
                / (
                    "train_balanced_domain_"
                    "target_annotations.tsv.gz"
                )
            ),
            selected_domains_path=(
                selected_domains_path
            ),
        )
    )

    validation_stage1_full = (
        JointStage1Dataset(
            fasta_path=(
                split_dir
                / "validation.fasta"
            ),
            annotations_path=(
                split_dir
                / (
                    "validation_"
                    "annotations.tsv.gz"
                )
            ),
            selected_domains_path=(
                selected_domains_path
            ),
        )
    )

    validation_stage2_full = (
        JointStage2Dataset(
            fasta_path=(
                split_dir
                / "validation.fasta"
            ),
            targets_path=(
                split_dir
                / (
                    "validation_balanced_"
                    "domain_targets.tsv.gz"
                )
            ),
            target_annotations_path=(
                split_dir
                / (
                    "validation_balanced_"
                    "domain_target_"
                    "annotations.tsv.gz"
                )
            ),
            selected_domains_path=(
                selected_domains_path
            ),
        )
    )

    (
        train_stage1,
        train_stage1_lengths,
    ) = subset_dataset(
        dataset=train_stage1_full,
        maximum=args.max_train_stage1,
        seed=args.seed + 101,
    )

    (
        train_stage2,
        train_stage2_lengths,
    ) = subset_dataset(
        dataset=train_stage2_full,
        maximum=args.max_train_stage2,
        seed=args.seed + 202,
    )

    (
        validation_stage1,
        validation_stage1_lengths,
    ) = subset_dataset(
        dataset=(
            validation_stage1_full
        ),
        maximum=(
            args.max_validation_stage1
        ),
        seed=args.seed + 303,
    )

    (
        validation_stage2,
        validation_stage2_lengths,
    ) = subset_dataset(
        dataset=(
            validation_stage2_full
        ),
        maximum=(
            args.max_validation_stage2
        ),
        seed=args.seed + 404,
    )

    training_batch_multiple = (
        accelerator.num_processes
        * args
        .gradient_accumulation_steps
    )

    train_stage1_sampler = (
        DistributedTokenBatchSampler(
            sequence_lengths=(
                train_stage1_lengths
            ),
            token_budget=(
                args.train_token_budget
            ),
            rank=(
                accelerator.process_index
            ),
            world_size=(
                accelerator.num_processes
            ),
            seed=args.seed + 1_000,
            bucket_size=args.bucket_size,
            shuffle=True,
            allow_oversize_singletons=True,
            batch_count_multiple=(
                training_batch_multiple
            ),
        )
    )

    train_stage2_sampler = (
        DistributedTokenBatchSampler(
            sequence_lengths=(
                train_stage2_lengths
            ),
            token_budget=(
                args.train_token_budget
            ),
            rank=(
                accelerator.process_index
            ),
            world_size=(
                accelerator.num_processes
            ),
            seed=args.seed + 2_000,
            bucket_size=args.bucket_size,
            shuffle=True,
            allow_oversize_singletons=True,
            batch_count_multiple=(
                training_batch_multiple
            ),
        )
    )

    validation_stage1_sampler = (
        DistributedTokenBatchSampler(
            sequence_lengths=(
                validation_stage1_lengths
            ),
            token_budget=(
                args
                .validation_token_budget
            ),
            rank=(
                accelerator.process_index
            ),
            world_size=(
                accelerator.num_processes
            ),
            seed=args.seed + 3_000,
            bucket_size=args.bucket_size,
            shuffle=False,
            allow_oversize_singletons=True,
            batch_count_multiple=(
                accelerator.num_processes
            ),
        )
    )

    validation_stage2_sampler = (
        DistributedTokenBatchSampler(
            sequence_lengths=(
                validation_stage2_lengths
            ),
            token_budget=(
                args
                .validation_token_budget
            ),
            rank=(
                accelerator.process_index
            ),
            world_size=(
                accelerator.num_processes
            ),
            seed=args.seed + 4_000,
            bucket_size=args.bucket_size,
            shuffle=False,
            allow_oversize_singletons=True,
            batch_count_multiple=(
                accelerator.num_processes
            ),
        )
    )

    train_stage1_loader = DataLoader(
        train_stage1,
        batch_sampler=(
            train_stage1_sampler
        ),
        collate_fn=(
            JointStage1Collator()
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_stage2_loader = DataLoader(
        train_stage2,
        batch_sampler=(
            train_stage2_sampler
        ),
        collate_fn=(
            JointStage2Collator()
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    validation_stage1_loader = (
        DataLoader(
            validation_stage1,
            batch_sampler=(
                validation_stage1_sampler
            ),
            collate_fn=(
                JointStage1Collator()
            ),
            num_workers=args.num_workers,
            pin_memory=True,
        )
    )

    validation_stage2_loader = (
        DataLoader(
            validation_stage2,
            batch_sampler=(
                validation_stage2_sampler
            ),
            collate_fn=(
                JointStage2Collator()
            ),
            num_workers=args.num_workers,
            pin_memory=True,
        )
    )

    expected_stage1_labels = sum(
        validation_stage1_lengths
    )

    expected_stage2_examples = len(
        validation_stage2
    )

    accelerator.print(
        "Loading ESMC encoder"
    )

    encoder = ESMC.from_pretrained(
        ESMC_MODEL_NAME,
        device=accelerator.device,
    )

    model = build_joint_model(
        encoder=encoder,
        stage1_head_type=(
            args.stage1_head
        ),
        stage1_dropout=(
            args.classifier_dropout
        ),
        stage2_dropout=(
            args.classifier_dropout
        ),
        domain_class_count=domain_class_count,
    )

    parameter_summary = (
        model.parameter_summary()
        .as_dict()
    )

    resumed_state = None

    if (
        args.resume == "auto"
        and latest_dir.exists()
    ):
        resumed_state = (
            load_model_checkpoint(
                model=model,
                checkpoint_dir=(
                    latest_dir
                ),
            )
        )

        accelerator.print(
            "Found resumable checkpoint: "
            f"{latest_dir}"
        )

    optimizer = build_optimizer(
        model=model,
        classifier_lr=(
            args.classifier_lr
        ),
        lora_lr=args.lora_lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    warmup_steps = int(
        round(
            args.total_optimizer_steps
            * args.warmup_fraction
        )
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            cosine_multiplier(
                step=step,
                warmup_steps=(
                    warmup_steps
                ),
                total_steps=(
                    args
                    .total_optimizer_steps
                ),
            )
        ),
    )

    model = accelerator.prepare(
        model
    )

    if resumed_state is not None:
        load_training_state(
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=latest_dir,
        )

    stage1_saved_loader_state = (
        resumed_state.get(
            "stage1_loader_state",
            {},
        )
        if resumed_state
        else {}
    )

    stage2_saved_loader_state = (
        resumed_state.get(
            "stage2_loader_state",
            {},
        )
        if resumed_state
        else {}
    )

    stage1_cycle = (
        RestartableDistributedLoader(
            loader=(
                train_stage1_loader
            ),
            sampler=(
                train_stage1_sampler
            ),
            initial_epoch=int(
                stage1_saved_loader_state
                .get(
                    "epoch",
                    0,
                )
            ),
            initial_batches_consumed=int(
                stage1_saved_loader_state
                .get(
                    "batches_consumed",
                    0,
                )
            ),
        )
    )

    stage2_cycle = (
        RestartableDistributedLoader(
            loader=(
                train_stage2_loader
            ),
            sampler=(
                train_stage2_sampler
            ),
            initial_epoch=int(
                stage2_saved_loader_state
                .get(
                    "epoch",
                    0,
                )
            ),
            initial_batches_consumed=int(
                stage2_saved_loader_state
                .get(
                    "batches_consumed",
                    0,
                )
            ),
        )
    )

    curriculum = JointCurriculum(
        total_optimizer_steps=(
            args.total_optimizer_steps
        ),
        seed=args.seed,
    )

    stage2_start_step = (
        3
        * args.total_optimizer_steps
        // 20
    )

    completed_steps = int(
        resumed_state.get(
            "global_optimizer_step",
            0,
        )
        if resumed_state
        else 0
    )

    best_stage1_auprc = float(
        resumed_state.get(
            "best_stage1_macro_auprc",
            -1.0,
        )
        if resumed_state
        else -1.0
    )

    best_stage2_macro_f1 = float(
        resumed_state.get(
            "best_stage2_macro_f1",
            -1.0,
        )
        if resumed_state
        else -1.0
    )

    best_joint_score = float(
        resumed_state.get(
            "best_joint_score",
            -1.0,
        )
        if resumed_state
        else -1.0
    )

    validation_index = int(
        resumed_state.get(
            "validation_index",
            0,
        )
        if resumed_state
        else 0
    )

    latest_metrics = (
        resumed_state.get(
            "latest_validation_metrics"
        )
        if resumed_state
        else None
    )

    completed_epochs = int(
        resumed_state.get(
            "completed_epochs",
            validation_index,
        )
        if resumed_state
        else 0
    )

    patience_counter = int(
        resumed_state.get(
            "patience_counter",
            0,
        )
        if resumed_state
        else 0
    )

    best_patience_joint_score = float(
        resumed_state.get(
            "best_patience_joint_score",
            -1.0,
        )
        if resumed_state
        else -1.0
    )

    stopped_early = bool(
        resumed_state.get(
            "stopped_early",
            False,
        )
        if resumed_state
        else False
    )

    if scheduler.last_epoch != (
        completed_steps
    ):
        raise RuntimeError(
            "Scheduler/global-step mismatch "
            f"{scheduler.last_epoch} != "
            f"{completed_steps}"
        )

    configuration = {
        "model_name": ESMC_MODEL_NAME,
        "shared_adapter": True,
        "stage1_output_channels": 1,
        "stage2_domain_classes": domain_class_count,
        "stage2_input_mode": (
            "visible unchanged sequence"
        ),
        "stage1_head": (
            args.stage1_head
        ),
        "classifier_dropout": (
            args.classifier_dropout
        ),
        "total_optimizer_steps": (
            args.total_optimizer_steps
        ),
        "max_epochs": (
            args.max_epochs
        ),
        "patience": (
            args.patience
        ),
        "min_delta": (
            args.min_delta
        ),
        "early_stopping_metric": (
            "joint_score"
        ),
        "patience_starts_after_step": (
            stage2_start_step
        ),
        "curriculum": (
            curriculum.summarize()
            .as_dict()
        ),
        "gradient_accumulation_steps": (
            args
            .gradient_accumulation_steps
        ),
        "train_token_budget": (
            args.train_token_budget
        ),
        "validation_token_budget": (
            args
            .validation_token_budget
        ),
        "classifier_lr": (
            args.classifier_lr
        ),
        "lora_lr": args.lora_lr,
        "weight_decay": (
            args.weight_decay
        ),
        "warmup_fraction": (
            args.warmup_fraction
        ),
        "warmup_steps": warmup_steps,
        "max_grad_norm": (
            args.max_grad_norm
        ),
        "seed": args.seed,
        "world_size": (
            accelerator.num_processes
        ),
        "find_unused_parameters": True,
        "train_stage1_examples": len(
            train_stage1
        ),
        "train_stage2_examples": len(
            train_stage2
        ),
        "validation_stage1_examples": (
            len(validation_stage1)
        ),
        "validation_stage2_examples": (
            len(validation_stage2)
        ),
        "validation_stage1_labels": (
            expected_stage1_labels
        ),
        "maximum_train_stage1_length": (
            max(train_stage1_lengths)
        ),
        "maximum_train_stage2_length": (
            max(train_stage2_lengths)
        ),
        "maximum_validation_stage1_length": (
            max(
                validation_stage1_lengths
            )
        ),
        "maximum_validation_stage2_length": (
            max(
                validation_stage2_lengths
            )
        ),
        "length_filter_applied": True,
        "parameter_summary": (
            parameter_summary
        ),
    }

    if accelerator.is_main_process:
        (
            args.output_dir
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

    accelerator.print(
        json.dumps(
            {
                "status": (
                    "trainer initialized"
                ),
                "resume_step": (
                    completed_steps
                ),
                "configuration": (
                    configuration
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    model.train()

    recent = {
        "stage1": {
            "loss_sum": 0.0,
            "count": 0.0,
            "steps": 0,
        },
        "stage2": {
            "loss_sum": 0.0,
            "count": 0.0,
            "steps": 0,
        },
    }

    run_start = time.perf_counter()
    stopped_for_allocation = False

    training_step_range = (
        range(
            completed_steps,
            args.total_optimizer_steps,
        )
        if not stopped_early
        else range(0)
    )

    for step_index in training_step_range:
        result = run_optimizer_step(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            stage1_loader=stage1_cycle,
            stage2_loader=stage2_cycle,
            curriculum=curriculum,
            global_optimizer_step=(
                step_index
            ),
            gradient_accumulation_steps=(
                args
                .gradient_accumulation_steps
            ),
            max_grad_norm=(
                args.max_grad_norm
            ),
        )

        completed_steps = (
            step_index + 1
        )

        recent[result.task][
            "loss_sum"
        ] += result.local_loss_sum

        recent[result.task][
            "count"
        ] += result.local_example_count

        recent[result.task][
            "steps"
        ] += 1

        if scheduler.last_epoch != (
            completed_steps
        ):
            raise RuntimeError(
                "Scheduler advanced an "
                "unexpected number of steps"
            )

        if (
            completed_steps
            % args.log_every
            == 0
        ):
            local_totals = torch.tensor(
                [
                    recent["stage1"][
                        "loss_sum"
                    ],
                    recent["stage1"][
                        "count"
                    ],
                    recent["stage2"][
                        "loss_sum"
                    ],
                    recent["stage2"][
                        "count"
                    ],
                ],
                device=(
                    accelerator.device
                ),
                dtype=torch.float64,
            )

            global_totals = (
                accelerator.reduce(
                    local_totals,
                    reduction="sum",
                )
            )

            stage1_loss = (
                float(
                    global_totals[0]
                    .item()
                    / global_totals[1]
                    .item()
                )
                if global_totals[1]
                .item()
                > 0
                else None
            )

            stage2_loss = (
                float(
                    global_totals[2]
                    .item()
                    / global_totals[3]
                    .item()
                )
                if global_totals[3]
                .item()
                > 0
                else None
            )

            accelerator.print(
                json.dumps(
                    {
                        "optimizer_step": (
                            completed_steps
                        ),
                        "phase": (
                            result.phase
                        ),
                        "last_task": (
                            result.task
                        ),
                        "recent_stage1_loss": (
                            stage1_loss
                        ),
                        "recent_stage2_loss": (
                            stage2_loss
                        ),
                        "recent_stage1_steps": (
                            recent["stage1"][
                                "steps"
                            ]
                        ),
                        "recent_stage2_steps": (
                            recent["stage2"][
                                "steps"
                            ]
                        ),
                        "head_lr": (
                            optimizer
                            .param_groups[0][
                                "lr"
                            ]
                        ),
                        "lora_lr": (
                            optimizer
                            .param_groups[1][
                                "lr"
                            ]
                        ),
                    },
                    sort_keys=True,
                )
            )

            for task in recent:
                recent[task] = {
                    "loss_sum": 0.0,
                    "count": 0.0,
                    "steps": 0,
                }

        validation_due = (
            completed_steps
            % args.validate_every
            == 0
            or completed_steps
            == args.total_optimizer_steps
        )

        if validation_due:
            validation_index += 1

            metrics = validate_joint(
                accelerator=accelerator,
                model=model,
                stage1_loader=(
                    validation_stage1_loader
                ),
                stage2_loader=(
                    validation_stage2_loader
                ),
                output_dir=(
                    args.output_dir
                ),
                global_optimizer_step=(
                    completed_steps
                ),
                expected_stage1_labels=(
                    expected_stage1_labels
                ),
                expected_stage2_examples=(
                    expected_stage2_examples
                ),
                class_count=domain_class_count,
            )

            latest_metrics = metrics

            stage1_improved = (
                metrics[
                    "stage1_macro_auprc"
                ]
                > best_stage1_auprc
            )

            stage2_improved = (
                metrics[
                    "stage2_macro_f1"
                ]
                > best_stage2_macro_f1
            )

            joint_improved = (
                metrics[
                    "joint_score"
                ]
                > best_joint_score
            )

            patience_active = (
                completed_steps
                > stage2_start_step
            )

            (
                best_patience_joint_score,
                patience_counter,
                patience_improved,
            ) = update_early_stopping(
                current_score=(
                    metrics["joint_score"]
                ),
                best_score=(
                    best_patience_joint_score
                ),
                patience_counter=(
                    patience_counter
                ),
                active=patience_active,
                min_delta=args.min_delta,
            )

            completed_epochs = (
                validation_index
            )

            stopped_early = (
                patience_active
                and patience_counter
                >= args.patience
            )

            if stage1_improved:
                best_stage1_auprc = (
                    metrics[
                        "stage1_macro_auprc"
                    ]
                )

            if stage2_improved:
                best_stage2_macro_f1 = (
                    metrics[
                        "stage2_macro_f1"
                    ]
                )

            if joint_improved:
                best_joint_score = (
                    metrics[
                        "joint_score"
                    ]
                )

            state = trainer_state(
                completed_steps=(
                    completed_steps
                ),
                stage1_loader=(
                    stage1_cycle
                ),
                stage2_loader=(
                    stage2_cycle
                ),
                best_stage1_auprc=(
                    best_stage1_auprc
                ),
                best_stage2_macro_f1=(
                    best_stage2_macro_f1
                ),
                best_joint_score=(
                    best_joint_score
                ),
                validation_index=(
                    validation_index
                ),
                latest_metrics=(
                    latest_metrics
                ),
                completed_epochs=completed_epochs,
                patience_counter=patience_counter,
                best_patience_joint_score=best_patience_joint_score,
                stopped_early=stopped_early,
            )

            if stage1_improved:
                save_checkpoint(
                    accelerator=(
                        accelerator
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=(
                        args.output_dir
                    ),
                    checkpoint_name=(
                        "best_by_stage1_auprc"
                    ),
                    trainer_state=state,
                    configuration=(
                        configuration
                    ),
                )

            if stage2_improved:
                save_checkpoint(
                    accelerator=(
                        accelerator
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=(
                        args.output_dir
                    ),
                    checkpoint_name=(
                        "best_by_stage2_macro_f1"
                    ),
                    trainer_state=state,
                    configuration=(
                        configuration
                    ),
                )

            if joint_improved:
                save_checkpoint(
                    accelerator=(
                        accelerator
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=(
                        args.output_dir
                    ),
                    checkpoint_name=(
                        "best_by_joint_score"
                    ),
                    trainer_state=state,
                    configuration=(
                        configuration
                    ),
                )

            save_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=(
                    args.output_dir
                ),
                checkpoint_name="latest",
                trainer_state=state,
                configuration=configuration,
            )

            history_record = {
                **metrics,
                "validation_index": (
                    validation_index
                ),
                "stage1_improved": (
                    stage1_improved
                ),
                "stage2_improved": (
                    stage2_improved
                ),
                "joint_improved": (
                    joint_improved
                ),
                "epoch": (
                    completed_epochs
                ),
                "patience_active": (
                    patience_active
                ),
                "patience_improved": (
                    patience_improved
                ),
                "patience_counter": (
                    patience_counter
                ),
                "best_patience_joint_score": (
                    best_patience_joint_score
                ),
                "stopped_early": (
                    stopped_early
                ),
                "best_stage1_macro_auprc": (
                    best_stage1_auprc
                ),
                "best_stage2_macro_f1": (
                    best_stage2_macro_f1
                ),
                "best_joint_score": (
                    best_joint_score
                ),
                "finished_at_utc": (
                    utc_now()
                ),
            }

            if accelerator.is_main_process:
                append_jsonl(
                    args.output_dir
                    / "history.jsonl",
                    history_record,
                )

            accelerator.print(
                json.dumps(
                    history_record,
                    indent=2,
                    sort_keys=True,
                )
            )

            if stopped_early:
                accelerator.print(
                    "Early stopping triggered: "
                    f"joint score failed to improve "
                    f"by {args.min_delta} for "
                    f"{args.patience} nominal epochs"
                )
                break

        stop_for_allocation = False

        if args.max_runtime_minutes > 0:
            if accelerator.is_main_process:
                elapsed_minutes = (
                    time.perf_counter()
                    - run_start
                ) / 60.0

                stop_for_allocation = (
                    elapsed_minutes
                    >= (
                        args
                        .max_runtime_minutes
                        - args
                        .runtime_buffer_minutes
                    )
                )

            stop_for_allocation = (
                broadcast_object(
                    accelerator,
                    stop_for_allocation,
                )
            )

        if stop_for_allocation:
            if not validation_due:
                state = trainer_state(
                    completed_steps=(
                        completed_steps
                    ),
                    stage1_loader=(
                        stage1_cycle
                    ),
                    stage2_loader=(
                        stage2_cycle
                    ),
                    best_stage1_auprc=(
                        best_stage1_auprc
                    ),
                    best_stage2_macro_f1=(
                        best_stage2_macro_f1
                    ),
                    best_joint_score=(
                        best_joint_score
                    ),
                    validation_index=(
                        validation_index
                    ),
                    latest_metrics=(
                        latest_metrics
                    ),
                    completed_epochs=completed_epochs,
                    patience_counter=patience_counter,
                    best_patience_joint_score=best_patience_joint_score,
                    stopped_early=stopped_early,
                )

                save_checkpoint(
                    accelerator=(
                        accelerator
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=(
                        args.output_dir
                    ),
                    checkpoint_name=(
                        "latest"
                    ),
                    trainer_state=state,
                    configuration=(
                        configuration
                    ),
                )

            stopped_for_allocation = True
            break

    completed = (
        completed_steps
        >= args.total_optimizer_steps
    )

    final_summary = {
        "status": (
            "EARLY_STOPPED"
            if stopped_early
            else (
                "COMPLETE"
                if completed
                else (
                    "ALLOCATION_STOP"
                    if stopped_for_allocation
                    else "STOPPED"
                )
            )
        ),
        "global_optimizer_step": (
            completed_steps
        ),
        "total_optimizer_steps": (
            args.total_optimizer_steps
        ),
        "best_stage1_macro_auprc": (
            best_stage1_auprc
        ),
        "best_stage2_macro_f1": (
            best_stage2_macro_f1
        ),
        "best_joint_score": (
            best_joint_score
        ),
        "latest_validation_metrics": (
            latest_metrics
        ),
        "completed_epochs": (
            completed_epochs
        ),
        "maximum_epochs": (
            args.max_epochs
        ),
        "patience": (
            args.patience
        ),
        "patience_counter": (
            patience_counter
        ),
        "best_patience_joint_score": (
            best_patience_joint_score
        ),
        "stopped_early": (
            stopped_early
        ),
        "stage1_loader_state": (
            stage1_cycle.state_dict()
        ),
        "stage2_loader_state": (
            stage2_cycle.state_dict()
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - run_start
        ),
        "finished_at_utc": utc_now(),
    }

    if accelerator.is_main_process:
        (
            args.output_dir
            / "run_summary.json"
        ).write_text(
            json.dumps(
                final_summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    accelerator.print(
        json.dumps(
            final_summary,
            indent=2,
            sort_keys=True,
        )
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
