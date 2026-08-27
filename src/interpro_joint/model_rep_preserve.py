"""One shared ESMC-LoRA encoder with Stage 1 and Stage 2 heads."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from esm.models.esmc import ESMC
from torch import nn
from interpro_joint.data import (
    PAD_TOKEN_ID,
)
from interpro_joint.representation_regularization import (
    select_representation_hidden_states,
)
from interpro_stage1.model import (
    HIDDEN_SIZE,
    LORA_ADAPTER_NAME,
    MLP_HIDDEN_SIZE,
    HeadType,
    inject_stage1_lora,
)
from interpro_stage2.model import (
    Stage2ClassificationHead,
)

DOMAIN_CLASS_COUNT = 64
JOINT_ADAPTER_NAME = LORA_ADAPTER_NAME

@dataclass(frozen=True)
class JointParameterSummary:
    total: int
    trainable: int
    encoder_total: int
    encoder_trainable: int
    stage1_head_total: int
    stage1_head_trainable: int
    stage2_head_total: int
    stage2_head_trainable: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def parameter_count(
    module: nn.Module,
    *,
    trainable_only: bool,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if (
            not trainable_only
            or parameter.requires_grad
        )
    )


class JointStage1ResidueHead(nn.Module):
    """Binary residue head for retained Domain-span prediction."""

    def __init__(
        self,
        *,
        head_type: HeadType = "linear",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.normalization = nn.LayerNorm(
            HIDDEN_SIZE
        )

        if head_type == "linear":
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    HIDDEN_SIZE,
                    1,
                ),
            )
        elif head_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(
                    HIDDEN_SIZE,
                    MLP_HIDDEN_SIZE,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    MLP_HIDDEN_SIZE,
                    1,
                ),
            )
        else:
            raise ValueError(
                f"Unsupported head type: "
                f"{head_type!r}"
            )

    def forward(
        self,
        residue_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = (
            residue_embeddings.float()
        )

        embeddings = self.normalization(
            embeddings
        )

        return self.network(
            embeddings
        )


class JointInterProModel(nn.Module):
    """
    Joint visible-region InterPro model.

    Stage 1: Original protein sequence -> residue-level span logits.

    Stage 2: Original protein sequence plus a boolean target mask -> one of the configured Domain IDs.

    The target mask is used only for pooling embeddings.
    """

    def __init__(
        self,
        *,
        encoder: ESMC,
        stage1_head_type: HeadType = "linear",
        stage1_dropout: float = 0.1,
        stage2_dropout: float = 0.1,
        domain_class_count: int = DOMAIN_CLASS_COUNT,
    ) -> None:
        super().__init__()

        if domain_class_count <= 1:
            raise ValueError(
                "domain_class_count must exceed one"
            )

        self.encoder = encoder

        self.stage1_head = JointStage1ResidueHead(
            head_type=stage1_head_type,
            dropout=stage1_dropout,
        )

        self.stage2_domain_head = (
            Stage2ClassificationHead(
                number_of_classes=domain_class_count,
                dropout=stage2_dropout,
            )
        )

        self.domain_class_count = int(
            domain_class_count
        )

    def encode(
        self,
        *,
        sequence_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        return_representation_features: bool = False,
    ):
        if sequence_tokens.ndim != 2:
            raise ValueError(
                "sequence_tokens must have shape "
                "[batch, tokens]"
            )

        if attention_mask.shape != sequence_tokens.shape:
            raise ValueError(
                "attention_mask must match "
                "sequence_tokens"
            )

        if attention_mask.dtype != torch.bool:
            raise TypeError(
                "attention_mask must be boolean"
            )

        autocast_enabled = sequence_tokens.is_cuda

        with torch.autocast(
            device_type=sequence_tokens.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = self.encoder.forward(
                sequence_tokens=sequence_tokens,
                sequence_id=attention_mask,
            )

        if output.embeddings is None:
            raise RuntimeError(
                "ESMC returned no embeddings"
            )

        embeddings = output.embeddings

        expected_shape = (
            sequence_tokens.shape[0],
            sequence_tokens.shape[1],
            HIDDEN_SIZE,
        )

        if tuple(embeddings.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected encoder embedding shape: "
                f"expected={expected_shape}, "
                f"observed={tuple(embeddings.shape)}"
            )

        if not return_representation_features:
            return embeddings

        if output.hidden_states is None:
            raise RuntimeError(
                "ESMC returned no hidden states"
            )

        features = (
            select_representation_hidden_states(
                output.hidden_states
            )
        )

        return embeddings, features

    def forward_stage1(
        self,
        *,
        sequence_tokens: torch.Tensor,
        residue_length: int,
        return_representation_features: bool = False,
    ):
        """
        Return Stage 1 residue logits. `residue_length` is the padded residue dimension in the
        corresponding Stage 1 labels tensor.

        BOS and EOS embeddings are excluded before applying the residue head.
        """
        if residue_length <= 0:
            raise ValueError(
                "residue_length must be positive"
            )

        if sequence_tokens.shape[1] < (
            residue_length + 2
        ):
            raise ValueError(
                "sequence_tokens are too short for "
                f"residue_length={residue_length}"
            )

        attention_mask = sequence_tokens.ne(
            PAD_TOKEN_ID
        )

        encoded = self.encode(
            sequence_tokens=sequence_tokens,
            attention_mask=attention_mask,
            return_representation_features=(
                return_representation_features
            ),
        )

        if return_representation_features:
            embeddings, features = encoded
        else:
            embeddings = encoded
            features = None

        residue_embeddings = embeddings[
            :,
            1 : 1 + residue_length,
            :,
        ]

        logits = self.stage1_head(
            residue_embeddings
        )

        if return_representation_features:
            assert features is not None
            return logits, features

        return logits

    @staticmethod
    def pool_visible_regions(
        *,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool visible amino-acid embeddings selected by the
        target-region mask.
        """
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape "
                "[batch, tokens, hidden]"
            )

        expected_mask_shape = embeddings.shape[:2]

        if tuple(attention_mask.shape) != tuple(
            expected_mask_shape
        ):
            raise ValueError(
                "attention_mask does not match embeddings"
            )

        if tuple(target_token_mask.shape) != tuple(
            expected_mask_shape
        ):
            raise ValueError(
                "target_token_mask does not match embeddings"
            )

        if attention_mask.dtype != torch.bool:
            raise TypeError(
                "attention_mask must be boolean"
            )

        if target_token_mask.dtype != torch.bool:
            raise TypeError(
                "target_token_mask must be boolean"
            )

        effective_mask = (
            attention_mask
            & target_token_mask
        )

        target_counts = effective_mask.sum(
            dim=1
        )

        if torch.any(target_counts <= 0):
            bad_rows = (
                torch.nonzero(
                    target_counts <= 0,
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )

            raise ValueError(
                "Every Stage 2 example must contain "
                "at least one visible target token; "
                f"invalid rows={bad_rows}"
            )

        weights = effective_mask.unsqueeze(
            -1
        ).to(dtype=embeddings.dtype)

        pooled = (
            embeddings * weights
        ).sum(dim=1)

        pooled = pooled / target_counts.to(
            dtype=embeddings.dtype
        ).unsqueeze(-1)

        return pooled

    def forward_stage2(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_mask: torch.Tensor,
        return_representation_features: bool = False,
    ):
        """
        Classify a fully visible target Domain region.
        """
        encoded = self.encode(
            sequence_tokens=input_ids,
            attention_mask=attention_mask,
            return_representation_features=(
                return_representation_features
            ),
        )

        if return_representation_features:
            embeddings, features = encoded
        else:
            embeddings = encoded
            features = None

        representation = (
            self.pool_visible_regions(
                embeddings=embeddings,
                attention_mask=attention_mask,
                target_token_mask=(
                    target_token_mask
                ),
            )
        )

        logits = self.stage2_domain_head(
            representation
        )

        expected_shape = (
            input_ids.shape[0],
            self.domain_class_count,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected Stage 2 logit shape: "
                f"expected={expected_shape}, "
                f"observed={tuple(logits.shape)}"
            )

        if return_representation_features:
            assert features is not None
            return logits, features

        return logits

    def forward(
        self,
        *,
        task: str,
        sequence_tokens: torch.Tensor | None = None,
        residue_length: int | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
        return_representation_features: bool = False,
    ):
        """
        The joint trainer must call the wrapped model through this
        method rather than calling forward_stage1 or forward_stage2
        directly.
        """
        if task == "stage1":
            if sequence_tokens is None:
                raise ValueError(
                    "Stage 1 requires sequence_tokens"
                )

            if residue_length is None:
                raise ValueError(
                    "Stage 1 requires residue_length"
                )

            return self.forward_stage1(
                sequence_tokens=sequence_tokens,
                residue_length=residue_length,
                return_representation_features=(
                    return_representation_features
                ),
            )

        if task == "mlm":
            if sequence_tokens is None:
                raise ValueError(
                    "MLM requires sequence_tokens"
                )

            if attention_mask is None:
                raise ValueError(
                    "MLM requires attention_mask"
                )

            autocast_enabled = (
                sequence_tokens.is_cuda
            )

            with torch.autocast(
                device_type=(
                    sequence_tokens.device.type
                ),
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                output = (
                    self.encoder.forward(
                        sequence_tokens=(
                            sequence_tokens
                        ),
                        sequence_id=(
                            attention_mask
                        ),
                    )
                )

            if (
                output.sequence_logits
                is None
            ):
                raise RuntimeError(
                    "ESMC returned no "
                    "sequence logits"
                )

            return (
                output.sequence_logits
            )

        if task == "stage2":
            if input_ids is None:
                raise ValueError(
                    "Stage 2 requires input_ids"
                )

            if attention_mask is None:
                raise ValueError(
                    "Stage 2 requires attention_mask"
                )

            if target_token_mask is None:
                raise ValueError(
                    "Stage 2 requires target_token_mask"
                )

            return self.forward_stage2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_token_mask=target_token_mask,
                return_representation_features=(
                    return_representation_features
                ),
            )

        raise ValueError(
            'task must be "stage1", "stage2", or "mlm"; '
            f"observed {task!r}"
        )

    def parameter_summary(
        self,
    ) -> JointParameterSummary:
        return JointParameterSummary(
            total=parameter_count(
                self,
                trainable_only=False,
            ),
            trainable=parameter_count(
                self,
                trainable_only=True,
            ),
            encoder_total=parameter_count(
                self.encoder,
                trainable_only=False,
            ),
            encoder_trainable=parameter_count(
                self.encoder,
                trainable_only=True,
            ),
            stage1_head_total=parameter_count(
                self.stage1_head,
                trainable_only=False,
            ),
            stage1_head_trainable=parameter_count(
                self.stage1_head,
                trainable_only=True,
            ),
            stage2_head_total=parameter_count(
                self.stage2_domain_head,
                trainable_only=False,
            ),
            stage2_head_trainable=parameter_count(
                self.stage2_domain_head,
                trainable_only=True,
            ),
        )


def build_joint_model(
    *,
    encoder: ESMC,
    stage1_head_type: HeadType = "linear",
    stage1_dropout: float = 0.1,
    stage2_dropout: float = 0.1,
    domain_class_count: int = DOMAIN_CLASS_COUNT,
) -> JointInterProModel:
    """
    Inject one fresh LoRA adapter and attach both task heads.

    The same adapter is updated by Stage 1 and Stage 2 steps.
    """
    encoder = inject_stage1_lora(
        encoder
    )

    model = JointInterProModel(
        encoder=encoder,
        stage1_head_type=stage1_head_type,
        stage1_dropout=stage1_dropout,
        stage2_dropout=stage2_dropout,
        domain_class_count=domain_class_count,
    )

    if not any(
        parameter.requires_grad
        for parameter in model.encoder.parameters()
    ):
        raise RuntimeError(
            "Joint encoder has no trainable parameters"
        )

    return model
