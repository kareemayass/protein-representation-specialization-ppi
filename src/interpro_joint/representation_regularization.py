"""Feature-space preservation regularization for joint InterPro training."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from peft.tuners.tuners_utils import BaseTunerLayer


# ESMC-300M has 30 transformer blocks.
# Preserve five evenly spaced post-block representations:
# blocks 6, 12, 18, 24, 30.
REP_PRESERVE_LAYER_INDICES = (
    5,
    11,
    17,
    23,
    29,
)


@contextmanager
def adapters_enabled(
    module,
    enabled: bool,
):
    """Temporarily enable or disable all PEFT LoRA layers."""
    layers = [
        child
        for child in module.modules()
        if isinstance(
            child,
            BaseTunerLayer,
        )
    ]

    if not layers:
        raise RuntimeError(
            "No PEFT tuner layers found"
        )

    previous_disabled = [
        bool(layer.disable_adapters)
        for layer in layers
    ]

    try:
        for layer in layers:
            layer.enable_adapters(
                enabled
            )

        yield

    finally:
        for layer, was_disabled in zip(
            layers,
            previous_disabled,
        ):
            layer.enable_adapters(
                not was_disabled
            )


def select_representation_hidden_states(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Select the five ESMC hidden states used for
    representation preservation.

    Input:
        [layers, batch, tokens, hidden]
    """
    if hidden_states.ndim != 4:
        raise ValueError(
            "hidden_states must have shape "
            "[layers, batch, tokens, hidden]"
        )

    if (
        max(REP_PRESERVE_LAYER_INDICES)
        >= hidden_states.shape[0]
    ):
        raise ValueError(
            "Requested representation layer "
            "does not exist"
        )

    indices = torch.tensor(
        REP_PRESERVE_LAYER_INDICES,
        device=hidden_states.device,
        dtype=torch.long,
    )

    # Materialize only the five selected layers.
    selected = torch.index_select(
        hidden_states,
        dim=0,
        index=indices,
    )

    return tuple(
        selected.unbind(dim=0)
    )


def reference_hidden_states(
    *,
    encoder,
    sequence_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Produce the Base-ESMC reference representation by
    temporarily disabling LoRA on the same encoder.

    No gradient graph is created for the reference pass.
    """
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

    with adapters_enabled(
        encoder,
        False,
    ):
        with torch.no_grad():
            with torch.autocast(
                device_type=(
                    sequence_tokens.device.type
                ),
                dtype=torch.bfloat16,
                enabled=(
                    sequence_tokens.is_cuda
                ),
            ):
                output = encoder.forward(
                    sequence_tokens=(
                        sequence_tokens
                    ),
                    sequence_id=(
                        attention_mask
                    ),
                )

    if output.hidden_states is None:
        raise RuntimeError(
            "ESMC returned no hidden states"
        )

    return tuple(
        feature.detach()
        for feature
        in select_representation_hidden_states(
            output.hidden_states
        )
    )


def representation_regularization_components(
    *,
    adapted_features: tuple[
        torch.Tensor,
        ...
    ],
    reference_features: tuple[
        torch.Tensor,
        ...
    ],
    attention_mask: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Padding-aware normalized squared-L2 feature distance.

    Returns:
        loss_sum:
            sum of per-protein losses

        protein_count:
            number of proteins represented in loss_sum
    """
    if (
        len(adapted_features)
        != len(reference_features)
    ):
        raise ValueError(
            "Adapted/reference layer mismatch"
        )

    if not adapted_features:
        raise ValueError(
            "No representation features"
        )

    if attention_mask.dtype != torch.bool:
        raise TypeError(
            "attention_mask must be boolean"
        )

    valid_positions = (
        attention_mask
        .sum(dim=1)
        .float()
    )

    if torch.any(
        valid_positions <= 0
    ):
        raise RuntimeError(
            "Empty protein in batch"
        )

    mask = (
        attention_mask
        .unsqueeze(-1)
        .float()
    )

    layer_losses = []

    for adapted, reference in zip(
        adapted_features,
        reference_features,
    ):
        if (
            adapted.shape
            != reference.shape
        ):
            raise ValueError(
                "Adapted/reference "
                "feature shape mismatch"
            )

        if (
            tuple(adapted.shape[:2])
            != tuple(
                attention_mask.shape
            )
        ):
            raise ValueError(
                "Feature/token mask mismatch"
            )

        adapted = (
            adapted.float()
            * mask
        )

        reference = (
            reference.float()
            * mask
        )

        # Match the normalized feature-space
        # comparison used by LDIFS-style methods:
        # normalize along token/position dimension.
        adapted_norm = torch.sqrt(
            adapted.square()
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(eps)
        )

        reference_norm = torch.sqrt(
            reference.square()
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(eps)
        )

        adapted = (
            adapted
            / adapted_norm
        )

        reference = (
            reference
            / reference_norm
        )

        squared_difference = (
            adapted
            - reference
        ).square() * mask

        # Equal weight per protein regardless
        # of sequence length.
        per_protein = (
            squared_difference.sum(
                dim=(1, 2)
            )
            /
            (
                valid_positions
                * adapted.shape[-1]
            )
        )

        layer_losses.append(
            per_protein
        )

    per_protein_loss = (
        torch.stack(
            layer_losses,
            dim=0,
        )
        .mean(dim=0)
    )

    loss_sum = (
        per_protein_loss.sum()
    )

    protein_count = torch.tensor(
        float(
            per_protein_loss.numel()
        ),
        device=loss_sum.device,
        dtype=loss_sum.dtype,
    )

    return (
        loss_sum,
        protein_count,
    )
