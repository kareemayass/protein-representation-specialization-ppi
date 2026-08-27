"""Stage 2 exact InterPro-ID prediction model."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from esm.models.esmc import ESMC


InputMode = Literal["visible", "sentinel"]
EntryType = Literal["Family", "Domain"]

HIDDEN_SIZE = 960
MLP_HIDDEN_SIZE = 256
FAMILY_CLASS_COUNT = 4118
DOMAIN_CLASS_COUNT = 3595
class Stage2ClassificationHead(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, MLP_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_HIDDEN_SIZE, number_of_classes),
        )

    def forward(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        if representation.ndim != 2:
            raise ValueError(
                "representation must have shape [batch, hidden]"
            )

        return self.network(representation.float())


class InterProStage2Model(nn.Module):
    def __init__(
        self,
        encoder: ESMC,
        encoder_gradients_enabled: bool,
        classifier_dropout: float = 0.1,
        family_class_count: int = FAMILY_CLASS_COUNT,
        domain_class_count: int = DOMAIN_CLASS_COUNT,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.encoder_gradients_enabled = encoder_gradients_enabled

        self.family_head = Stage2ClassificationHead(
            family_class_count,
            classifier_dropout,
        )
        self.domain_head = Stage2ClassificationHead(
            domain_class_count,
            classifier_dropout,
        )

    def train(
        self,
        mode: bool = True,
    ) -> "InterProStage2Model":
        super().train(mode)

        if not self.encoder_gradients_enabled:
            self.encoder.eval()

        return self

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, tokens]"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids"
            )

        autocast_enabled = input_ids.is_cuda

        if self.encoder_gradients_enabled:
            with torch.autocast(
                device_type=input_ids.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                output = self.encoder.forward(
                    sequence_tokens=input_ids,
                    sequence_id=attention_mask.bool(),
                )
        else:
            with torch.no_grad():
                with torch.autocast(
                    device_type=input_ids.device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    output = self.encoder.forward(
                        sequence_tokens=input_ids,
                        sequence_id=attention_mask.bool(),
                    )

        if output.embeddings is None:
            raise RuntimeError("ESMC returned no embeddings")

        embeddings = output.embeddings

        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            HIDDEN_SIZE,
        )

        if tuple(embeddings.shape) != expected_shape:
            raise RuntimeError(
                f"Expected embeddings {expected_shape}, "
                f"observed {tuple(embeddings.shape)}"
            )

        return embeddings

    @staticmethod
    def pool_visible(
        embeddings: torch.Tensor,
        target_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_token_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "target_token_mask shape does not match embeddings"
            )

        mask = target_token_mask.unsqueeze(-1).to(
            embeddings.dtype
        )
        counts = mask.sum(dim=1)

        if torch.any(counts <= 0):
            raise RuntimeError(
                "Visible examples must select at least one token"
            )

        return (embeddings * mask).sum(dim=1) / counts

    @staticmethod
    def pool_sentinel(
        embeddings: torch.Tensor,
        sentinel_positions: torch.Tensor,
    ) -> torch.Tensor:
        if sentinel_positions.ndim != 1:
            raise ValueError(
                "sentinel_positions must have shape [batch]"
            )

        if sentinel_positions.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "sentinel_positions batch size mismatch"
            )

        if torch.any(sentinel_positions < 0):
            raise RuntimeError(
                "Invalid negative sentinel position"
            )

        if torch.any(
            sentinel_positions >= embeddings.shape[1]
        ):
            raise RuntimeError(
                "Sentinel position exceeds sequence length"
            )

        batch_indices = torch.arange(
            embeddings.shape[0],
            device=embeddings.device,
        )

        return embeddings[
            batch_indices,
            sentinel_positions,
            :,
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_mask: torch.Tensor,
        sentinel_positions: torch.Tensor,
        input_mode: InputMode,
        entry_type: EntryType,
    ) -> torch.Tensor:
        embeddings = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if input_mode == "visible":
            representation = self.pool_visible(
                embeddings,
                target_token_mask,
            )
        elif input_mode == "sentinel":
            representation = self.pool_sentinel(
                embeddings,
                sentinel_positions,
            )
        else:
            raise ValueError(
                f"Unsupported input mode: {input_mode}"
            )

        if entry_type == "Family":
            return self.family_head(representation)

        if entry_type == "Domain":
            return self.domain_head(representation)

        raise ValueError(
            f"Unsupported entry type: {entry_type}"
        )


def freeze_encoder(
    encoder: ESMC,
) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False


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
        "family_head": sum(
            parameter.numel()
            for parameter in model.family_head.parameters()
        ),
        "domain_head": sum(
            parameter.numel()
            for parameter in model.domain_head.parameters()
        ),
    }
