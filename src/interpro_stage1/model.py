"""ESMC residue-level Family/Domain prediction models."""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
from esm.models.esmc import ESMC
from peft import LoraConfig, inject_adapter_in_model


Stage1Mode = Literal["frozen", "lora_r8"]
HeadType = Literal["linear", "mlp"]
HIDDEN_SIZE = 960
MLP_HIDDEN_SIZE = 256
NUMBER_OF_LABELS = 2
NUMBER_OF_BLOCKS = 30
PAD_TOKEN_ID = 1
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_ADAPTER_NAME = "stage1"


def exact_lora_targets() -> list[str]:
    targets: list[str] = []

    for block_index in range(NUMBER_OF_BLOCKS):
        prefix = f"transformer.blocks.{block_index}"

        targets.extend(
            [
                f"{prefix}.attn.layernorm_qkv.1",
                f"{prefix}.attn.out_proj",
                f"{prefix}.ffn.1",
                f"{prefix}.ffn.3",
            ]
        )

    return targets


class Stage1ResidueHead(nn.Module):
    """Predict independent Family and Domain logits per residue."""

    def __init__(
        self,
        head_type: HeadType = "linear",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if head_type not in {"linear", "mlp"}:
            raise ValueError(
                f"Unsupported head type: {head_type}"
            )

        self.head_type = head_type
        self.normalization = nn.LayerNorm(HIDDEN_SIZE)

        if head_type == "linear":
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    HIDDEN_SIZE,
                    NUMBER_OF_LABELS,
                ),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(
                    HIDDEN_SIZE,
                    MLP_HIDDEN_SIZE,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    MLP_HIDDEN_SIZE,
                    NUMBER_OF_LABELS,
                ),
            )

    def forward(
        self,
        residue_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = residue_embeddings.float()
        embeddings = self.normalization(embeddings)

        return self.network(embeddings)


class InterProStage1Model(nn.Module):
    def __init__(
        self,
        encoder: ESMC,
        mode: Stage1Mode,
        head_type: HeadType,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if mode not in {"frozen", "lora_r8"}:
            raise ValueError(
                f"Unsupported Stage 1 mode: {mode}"
            )

        self.mode = mode
        self.head_type = head_type
        self.encoder = encoder
        self.head = Stage1ResidueHead(
            head_type=head_type,
            dropout=classifier_dropout,
        )

    def train(
        self,
        mode: bool = True,
    ) -> "InterProStage1Model":
        super().train(mode)

        if self.mode == "frozen":
            self.encoder.eval()

        return self

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        residue_length: int,
    ) -> torch.Tensor:
        if sequence_tokens.ndim != 2:
            raise ValueError(
                "sequence_tokens must have shape [batch, tokens]"
            )

        if residue_length <= 0:
            raise ValueError(
                "residue_length must be positive"
            )

        sequence_id = sequence_tokens.ne(
            PAD_TOKEN_ID
        )

        autocast_enabled = sequence_tokens.is_cuda

        if self.mode == "frozen":
            with torch.no_grad():
                with torch.autocast(
                    device_type=sequence_tokens.device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    output = self.encoder.forward(
                        sequence_tokens=sequence_tokens,
                        sequence_id=sequence_id,
                    )
        else:
            with torch.autocast(
                device_type=sequence_tokens.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                output = self.encoder.forward(
                    sequence_tokens=sequence_tokens,
                    sequence_id=sequence_id,
                )

        if output.embeddings is None:
            raise RuntimeError(
                "ESMC returned no embeddings"
            )

        residue_embeddings = output.embeddings[
            :,
            1 : 1 + residue_length,
            :,
        ]

        expected = (
            sequence_tokens.shape[0],
            residue_length,
            HIDDEN_SIZE,
        )

        if tuple(residue_embeddings.shape) != expected:
            raise RuntimeError(
                "Unexpected residue embedding shape: "
                f"expected={expected}, "
                f"observed={tuple(residue_embeddings.shape)}"
            )

        return self.head(residue_embeddings)


def freeze_encoder(
    encoder: ESMC,
) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False


def inject_stage1_lora(
    encoder: ESMC,
) -> ESMC:
    freeze_encoder(encoder)

    targets = exact_lora_targets()
    available_modules = dict(
        encoder.named_modules()
    )

    missing = [
        target
        for target in targets
        if target not in available_modules
    ]

    if missing:
        raise RuntimeError(
            "Missing expected LoRA modules: "
            + ", ".join(missing)
        )

    configuration = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets,
        bias="none",
        init_lora_weights=True,
    )

    encoder = inject_adapter_in_model(
        configuration,
        encoder,
        adapter_name=LORA_ADAPTER_NAME,
    )

    wrapped = 0

    for target in targets:
        module = dict(
            encoder.named_modules()
        ).get(target)

        if module is None or not hasattr(module, "lora_A"):
            raise RuntimeError(
                f"LoRA was not injected into {target}"
            )

        wrapped += 1

    if wrapped != 120:
        raise RuntimeError(
            f"Expected 120 LoRA modules, observed {wrapped}"
        )

    unexpected_trainable = [
        name
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
    ]

    if unexpected_trainable:
        raise RuntimeError(
            "Unexpected trainable base parameters: "
            + ", ".join(unexpected_trainable[:20])
        )

    return encoder


def build_stage1_model(
    encoder: ESMC,
    mode: Stage1Mode,
    head_type: HeadType = "linear",
    classifier_dropout: float = 0.1,
) -> InterProStage1Model:
    if mode == "frozen":
        freeze_encoder(encoder)
    elif mode == "lora_r8":
        encoder = inject_stage1_lora(encoder)
    else:
        raise ValueError(
            f"Unsupported Stage 1 mode: {mode}"
        )

    return InterProStage1Model(
        encoder=encoder,
        mode=mode,
        head_type=head_type,
        classifier_dropout=classifier_dropout,
    )


def count_parameters(
    model: nn.Module,
) -> dict[str, int]:
    return {
        "total": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "head_total": sum(
            parameter.numel()
            for parameter in model.head.parameters()
        ),
        "head_trainable": sum(
            parameter.numel()
            for parameter in model.head.parameters()
            if parameter.requires_grad
        ),
    }
