"""Auxiliary masked-language-model utilities for joint InterPro training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


MASK_TOKEN_ID = 32

# Native ESMC non-residue/special token IDs in the local tokenizer:
# <cls>=0, <pad>=1, <eos>=2, <unk>=3, |=31, <mask>=32.
SPECIAL_TOKEN_IDS = (
    0,
    1,
    2,
    3,
    31,
    32,
)


@dataclass(frozen=True)
class GradientPairStats:
    task_norm: float
    mlm_norm: float
    cosine_similarity: float
    lambda_candidate: float

    def as_dict(
        self,
    ) -> dict[str, float]:
        return asdict(self)


def make_mlm_copy(
    *,
    sequence_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_fraction: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create a dynamically masked copy of a task input.

    The original tensor is never modified. Exactly approximately
    `mask_fraction` of eligible residue tokens in each protein are
    replaced by the native ESMC <mask> token.
    """
    if sequence_tokens.ndim != 2:
        raise ValueError(
            "sequence_tokens must be "
            "[batch, tokens]"
        )

    if (
        attention_mask.shape
        != sequence_tokens.shape
    ):
        raise ValueError(
            "attention_mask must match "
            "sequence_tokens"
        )

    if attention_mask.dtype != torch.bool:
        raise TypeError(
            "attention_mask must be boolean"
        )

    if not 0.0 < mask_fraction < 1.0:
        raise ValueError(
            "mask_fraction must be in (0, 1)"
        )

    eligible = attention_mask.clone()

    for token_id in SPECIAL_TOKEN_IDS:
        eligible &= sequence_tokens.ne(
            token_id
        )

    masked_tokens = (
        sequence_tokens.clone()
    )

    prediction_mask = torch.zeros_like(
        attention_mask,
        dtype=torch.bool,
    )

    for row in range(
        sequence_tokens.shape[0]
    ):
        positions = torch.nonzero(
            eligible[row],
            as_tuple=False,
        ).flatten()

        eligible_count = int(
            positions.numel()
        )

        if eligible_count <= 0:
            raise RuntimeError(
                "MLM row contains no eligible "
                f"residue tokens: row={row}"
            )

        # Round 15% to the nearest integer,
        # with at least one masked residue.
        mask_count = max(
            1,
            min(
                eligible_count,
                int(
                    math.floor(
                        mask_fraction
                        * eligible_count
                        + 0.5
                    )
                ),
            ),
        )

        selected = positions[
            torch.randperm(
                eligible_count,
                device=positions.device,
            )[
                :mask_count
            ]
        ]

        masked_tokens[
            row,
            selected,
        ] = MASK_TOKEN_ID

        prediction_mask[
            row,
            selected,
        ] = True

    return (
        masked_tokens,
        prediction_mask,
    )


def mlm_loss_components(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute ESMC-style equal-protein MLM loss components.

    Each protein's masked-token cross-entropy is first averaged
    within that protein. Those per-protein means are then summed.
    Dividing sequence_loss_sum by sequence_count therefore gives
    each protein equal weight regardless of sequence length.

    Token-weighted sums/counts are also returned for logging.
    """
    if logits.ndim != 3:
        raise ValueError(
            "MLM logits must be "
            "[batch, tokens, vocabulary]"
        )

    if (
        targets.shape
        != logits.shape[:2]
    ):
        raise ValueError(
            "MLM target shape mismatch"
        )

    if (
        prediction_mask.shape
        != targets.shape
    ):
        raise ValueError(
            "MLM prediction-mask shape "
            "mismatch"
        )

    if prediction_mask.dtype != torch.bool:
        raise TypeError(
            "MLM prediction_mask must "
            "be boolean"
        )

    if (
        int(
            targets.max().item()
        )
        >= logits.shape[-1]
    ):
        raise RuntimeError(
            "MLM target token ID exceeds "
            "sequence-logit dimension"
        )

    token_losses = (
        F.cross_entropy(
            logits.float().reshape(
                -1,
                logits.shape[-1],
            ),
            targets.long().reshape(-1),
            reduction="none",
        )
        .reshape(
            targets.shape
        )
    )

    counts = prediction_mask.sum(
        dim=1
    )

    if torch.any(
        counts <= 0
    ):
        raise RuntimeError(
            "Every MLM sequence must "
            "contain a masked token"
        )

    masked_losses = (
        token_losses
        * prediction_mask.to(
            token_losses.dtype
        )
    )

    per_sequence_mean = (
        masked_losses.sum(
            dim=1
        )
        / counts.to(
            token_losses.dtype
        )
    )

    sequence_loss_sum = (
        per_sequence_mean.sum()
    )

    sequence_count = torch.tensor(
        float(
            targets.shape[0]
        ),
        device=logits.device,
        dtype=(
            sequence_loss_sum.dtype
        ),
    )

    token_loss_sum = (
        masked_losses.sum()
    )

    token_count = (
        prediction_mask
        .sum()
        .to(
            token_loss_sum.dtype
        )
    )

    return (
        sequence_loss_sum,
        sequence_count,
        token_loss_sum,
        token_count,
    )


def capture_lora_gradients(
    model: nn.Module,
) -> dict[
    str,
    torch.Tensor,
]:
    """
    Copy the current gradients of every trainable LoRA tensor.
    """
    gradients: dict[
        str,
        torch.Tensor,
    ] = {}

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if (
            parameter.requires_grad
            and "lora_" in name
        ):
            if parameter.grad is None:
                raise RuntimeError(
                    "Trainable LoRA parameter "
                    "has no gradient: "
                    f"{name}"
                )

            gradients[name] = (
                parameter
                .grad
                .detach()
                .float()
                .clone()
            )

    if not gradients:
        raise RuntimeError(
            "No LoRA gradients were "
            "captured"
        )

    return gradients


def compare_gradient_sets(
    *,
    task_gradients: dict[
        str,
        torch.Tensor,
    ],
    mlm_gradients: dict[
        str,
        torch.Tensor,
    ],
    target_ratio: float,
) -> GradientPairStats:
    """
    Compare task and MLM gradients over the shared LoRA tensors.

    lambda_candidate is chosen so that:

        ||lambda * g_MLM||
            ~= target_ratio * ||g_task||
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(
            "target_ratio must be "
            "in (0, 1)"
        )

    if (
        set(task_gradients)
        != set(mlm_gradients)
    ):
        raise RuntimeError(
            "Task and MLM LoRA "
            "gradient keys differ"
        )

    device = next(
        iter(
            task_gradients.values()
        )
    ).device

    task_sq = torch.zeros(
        (),
        device=device,
    )

    mlm_sq = torch.zeros(
        (),
        device=device,
    )

    dot = torch.zeros(
        (),
        device=device,
    )

    for name in task_gradients:
        task = task_gradients[
            name
        ]

        mlm = mlm_gradients[
            name
        ]

        task_sq += torch.sum(
            task * task
        )

        mlm_sq += torch.sum(
            mlm * mlm
        )

        dot += torch.sum(
            task * mlm
        )

    task_norm = torch.sqrt(
        task_sq
    )

    mlm_norm = torch.sqrt(
        mlm_sq
    )

    if task_norm.item() <= 0.0:
        raise RuntimeError(
            "Task LoRA gradient norm "
            "is zero"
        )

    if mlm_norm.item() <= 0.0:
        raise RuntimeError(
            "MLM LoRA gradient norm "
            "is zero"
        )

    cosine = (
        dot
        / (
            task_norm
            * mlm_norm
        )
    )

    lambda_candidate = (
        target_ratio
        * task_norm
        / mlm_norm
    )

    return GradientPairStats(
        task_norm=float(
            task_norm.item()
        ),
        mlm_norm=float(
            mlm_norm.item()
        ),
        cosine_similarity=float(
            cosine.item()
        ),
        lambda_candidate=float(
            lambda_candidate.item()
        ),
    )
