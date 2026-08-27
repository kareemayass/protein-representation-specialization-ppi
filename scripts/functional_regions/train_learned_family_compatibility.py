#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import hashlib
import math
import random
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "INTERPRO_PROJECT_ROOT",
        REPO_ROOT,
    )
).resolve()

TUNA = Path(
    os.environ.get(
        "TUNA_ROOT",
        REPO_ROOT / "external" / "TUnA-R",
    )
).resolve()

OUTDIR = (
    ROOT
    / "results/family_compatibility_overnight"
)

PREP = (
    OUTDIR
    / "base_family_data.pt"
)

TRAIN_PAIRS = (
    TUNA
    / "data/gold_standard"
    / "Intra1_interaction_1500_or_less.tsv"
)

VAL_PAIRS = (
    TUNA
    / "data/gold_standard"
    / "Intra0_interaction_1500_or_less.tsv"
)

SUMMARY_OUT = (
    OUTDIR
    / "learned_compatibility_summary.tsv"
)

SEEDS = (
    47,
    1047,
    2047,
)

EPOCHS = 6
BATCH_SIZE = 192
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_DOMAINS_PER_PROTEIN = 16
INTERNAL_VAL_FRACTION = 0.10


class CompatibilityMLP(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                960 * 3,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                0.10
            ),
            nn.Linear(
                256,
                64,
            ),
            nn.GELU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        features,
    ):

        return (
            self.network(
                features
            )
            .squeeze(-1)
        )


def read_pairs(path):

    pairs = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, raw in enumerate(
            handle,
            start=1,
        ):

            cols = (
                raw.rstrip("\n")
                .split("\t")
            )

            if len(cols) != 3:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    "expected 3 columns"
                )

            A, B, label = cols

            if label not in {
                "0",
                "1",
            }:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"bad label {label}"
                )

            pairs.append(
                (
                    A.strip(),
                    B.strip(),
                    int(label),
                )
            )

    return pairs


def auc_score(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    pos = (
        labels == 1
    )

    neg = (
        labels == 0
    )

    n_pos = int(
        pos.sum()
    )

    n_neg = int(
        neg.sum()
    )

    ranks = rankdata(
        scores,
        method="average",
    )

    return float(
        (
            ranks[pos].sum()
            -
            n_pos
            * (n_pos + 1)
            / 2.0
        )
        /
        (
            n_pos
            * n_neg
        )
    )


def average_precision(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    y = labels[
        order
    ]

    n_pos = int(
        y.sum()
    )

    if n_pos == 0:
        return float("nan")

    cumulative = np.cumsum(
        y
    )

    precision = (
        cumulative
        /
        np.arange(
            1,
            len(y) + 1,
        )
    )

    return float(
        precision[
            y == 1
        ].sum()
        /
        n_pos
    )


def deterministic_cap(
    *,
    protein,
    indices,
    ipr_ids,
):

    if (
        len(indices)
        <= MAX_DOMAINS_PER_PROTEIN
    ):
        return list(
            indices
        )

    ranked = []

    for index in indices:

        token = (
            f"{protein}|"
            f"{ipr_ids[index]}"
        )

        digest = hashlib.sha1(
            token.encode(
                "utf-8"
            )
        ).hexdigest()

        ranked.append(
            (
                digest,
                index,
            )
        )

    ranked.sort()

    return [
        index
        for _, index
        in ranked[
            :MAX_DOMAINS_PER_PROTEIN
        ]
    ]


def prepare_split(
    split,
    device,
):

    vectors = (
        split[
            "vectors"
        ]
        .float()
        .to(
            device
        )
    )

    mapping = {}

    truncated = 0

    for protein, indices in (
        split[
            "protein_to_indices"
        ].items()
    ):

        capped = deterministic_cap(
            protein=protein,
            indices=indices,
            ipr_ids=(
                split[
                    "ipr_ids"
                ]
            ),
        )

        if (
            len(capped)
            < len(indices)
        ):
            truncated += 1

        mapping[
            protein
        ] = torch.tensor(
            capped,
            dtype=torch.long,
            device=device,
        )

    print(
        f"{split['split']}: "
        f"annotated proteins="
        f"{len(mapping):,} | "
        f"protein-domain vectors="
        f"{vectors.shape[0]:,} | "
        f"truncated proteins="
        f"{truncated:,}",
        flush=True,
    )

    return (
        vectors,
        mapping,
    )


def annotated_pairs(
    pairs,
    mapping,
):

    return [
        (
            index,
            A,
            B,
            label,
        )
        for index, (
            A,
            B,
            label,
        ) in enumerate(
            pairs
        )
        if (
            A in mapping
            and B in mapping
            and len(
                mapping[A]
            ) > 0
            and len(
                mapping[B]
            ) > 0
        )
    ]


def stratified_split(
    pairs,
    *,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    positive = [
        i
        for i, item
        in enumerate(
            pairs
        )
        if item[
            3
        ] == 1
    ]

    negative = [
        i
        for i, item
        in enumerate(
            pairs
        )
        if item[
            3
        ] == 0
    ]

    rng.shuffle(
        positive
    )

    rng.shuffle(
        negative
    )

    n_pos_val = max(
        1,
        int(
            round(
                len(positive)
                * INTERNAL_VAL_FRACTION
            )
        ),
    )

    n_neg_val = max(
        1,
        int(
            round(
                len(negative)
                * INTERNAL_VAL_FRACTION
            )
        ),
    )

    val_indices = set(
        positive[
            :n_pos_val
        ]
        +
        negative[
            :n_neg_val
        ]
    )

    train = [
        item
        for i, item
        in enumerate(
            pairs
        )
        if i not in val_indices
    ]

    val = [
        item
        for i, item
        in enumerate(
            pairs
        )
        if i in val_indices
    ]

    return (
        train,
        val,
    )


def pair_logits(
    *,
    model,
    vectors,
    mapping,
    batch,
):

    feature_chunks = []
    lengths = []

    for (
        _pair_index,
        protein_A,
        protein_B,
        _label,
    ) in batch:

        A = vectors[
            mapping[
                protein_A
            ]
        ]

        B = vectors[
            mapping[
                protein_B
            ]
        ]

        A_expand = (
            A[
                :,
                None,
                :
            ]
        )

        B_expand = (
            B[
                None,
                :,
                :
            ]
        )

        sum_feature = (
            A_expand
            + B_expand
        )

        difference_feature = (
            torch.abs(
                A_expand
                - B_expand
            )
        )

        product_feature = (
            A_expand
            * B_expand
        )

        features = torch.cat(
            (
                sum_feature,
                difference_feature,
                product_feature,
            ),
            dim=-1,
        ).reshape(
            -1,
            960 * 3,
        )

        feature_chunks.append(
            features
        )

        lengths.append(
            int(
                features.shape[0]
            )
        )

    all_features = torch.cat(
        feature_chunks,
        dim=0,
    )

    domain_logits = model(
        all_features
    )

    outputs = []

    offset = 0

    for length in lengths:

        segment = (
            domain_logits[
                offset:
                offset + length
            ]
        )

        #
        # Normalized log-sum-exp:
        # a smooth MIL aggregator that does
        # not automatically increase merely
        # because a protein has more domains.
        #
        pooled = (
            torch.logsumexp(
                segment,
                dim=0,
            )
            -
            math.log(
                length
            )
        )

        outputs.append(
            pooled
        )

        offset += length

    return torch.stack(
        outputs
    )


def evaluate_pairs(
    *,
    model,
    vectors,
    mapping,
    pairs,
    batch_size,
    device,
):

    model.eval()

    labels = []
    logits = []
    pair_indices = []

    with torch.inference_mode():

        for start in range(
            0,
            len(pairs),
            batch_size,
        ):

            batch = pairs[
                start:
                start + batch_size
            ]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):

                output = pair_logits(
                    model=model,
                    vectors=vectors,
                    mapping=mapping,
                    batch=batch,
                )

            logits.extend(
                output
                .float()
                .cpu()
                .tolist()
            )

            labels.extend(
                [
                    item[3]
                    for item
                    in batch
                ]
            )

            pair_indices.extend(
                [
                    item[0]
                    for item
                    in batch
                ]
            )

    probabilities = (
        torch.sigmoid(
            torch.tensor(
                logits,
                dtype=torch.float32,
            )
        )
        .numpy()
    )

    return {
        "labels":
            np.asarray(
                labels,
                dtype=np.int64,
            ),

        "logits":
            np.asarray(
                logits,
                dtype=np.float64,
            ),

        "probabilities":
            probabilities,

        "pair_indices":
            np.asarray(
                pair_indices,
                dtype=np.int64,
            ),

        "AUROC":
            auc_score(
                labels,
                logits,
            ),

        "AUPRC":
            average_precision(
                labels,
                logits,
            ),
    }


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
        flush=True,
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Learned compatibility job "
            "expected a GPU compute node"
        )

    payload = torch.load(
        PREP,
        map_location="cpu",
        weights_only=False,
    )

    intra1_vectors, intra1_mapping = (
        prepare_split(
            payload[
                "intra1"
            ],
            device,
        )
    )

    intra0_vectors, intra0_mapping = (
        prepare_split(
            payload[
                "intra0"
            ],
            device,
        )
    )

    raw_train_pairs = read_pairs(
        TRAIN_PAIRS
    )

    raw_val_pairs = read_pairs(
        VAL_PAIRS
    )

    if len(raw_val_pairs) != 53665:
        raise RuntimeError(
            "Expected 53,665 Intra0 pairs; "
            f"observed {len(raw_val_pairs):,}"
        )

    train_annotated = annotated_pairs(
        raw_train_pairs,
        intra1_mapping,
    )

    val_annotated = annotated_pairs(
        raw_val_pairs,
        intra0_mapping,
    )

    print(
        "Intra1 both-domain-annotated "
        "PPI pairs:",
        f"{len(train_annotated):,}",
        flush=True,
    )

    print(
        "Intra0 both-domain-annotated "
        "PPI pairs:",
        f"{len(val_annotated):,}",
        flush=True,
    )

    summaries = []

    for seed in SEEDS:

        print()
        print("=" * 80)
        print(
            "SEED",
            seed,
            flush=True,
        )

        random.seed(
            seed
        )

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

        (
            train_pairs,
            internal_val_pairs,
        ) = stratified_split(
            train_annotated,
            seed=seed,
        )

        print(
            "training pairs:",
            f"{len(train_pairs):,}",
            "| internal validation:",
            f"{len(internal_val_pairs):,}",
            flush=True,
        )

        model = (
            CompatibilityMLP()
            .to(
                device
            )
        )

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=(
                    WEIGHT_DECAY
                ),
            )
        )

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        best_state = None
        best_epoch = None
        best_internal_key = None
        best_internal_auc = None
        best_internal_ap = None

        rng = np.random.default_rng(
            seed
        )

        for epoch in range(
            1,
            EPOCHS + 1,
        ):

            model.train()

            order = rng.permutation(
                len(
                    train_pairs
                )
            )

            running_loss = 0.0
            seen = 0

            for start in range(
                0,
                len(order),
                BATCH_SIZE,
            ):

                batch_indices = order[
                    start:
                    start + BATCH_SIZE
                ]

                batch = [
                    train_pairs[
                        int(index)
                    ]
                    for index
                    in batch_indices
                ]

                targets = torch.tensor(
                    [
                        item[3]
                        for item
                        in batch
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    logits = pair_logits(
                        model=model,
                        vectors=(
                            intra1_vectors
                        ),
                        mapping=(
                            intra1_mapping
                        ),
                        batch=batch,
                    )

                    loss = criterion(
                        logits,
                        targets,
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

                batch_n = len(
                    batch
                )

                running_loss += (
                    float(
                        loss.item()
                    )
                    * batch_n
                )

                seen += batch_n

            internal = evaluate_pairs(
                model=model,
                vectors=(
                    intra1_vectors
                ),
                mapping=(
                    intra1_mapping
                ),
                pairs=(
                    internal_val_pairs
                ),
                batch_size=(
                    EVAL_BATCH_SIZE
                ),
                device=device,
            )

            train_loss = (
                running_loss
                / seen
            )

            print(
                f"epoch={epoch} "
                f"train_loss="
                f"{train_loss:.6f} "
                f"internal_AUROC="
                f"{internal['AUROC']:.6f} "
                f"internal_AUPRC="
                f"{internal['AUPRC']:.6f}",
                flush=True,
            )

            #
            # Checkpoint selection uses only an
            # Intra1 internal pair holdout.
            #
            internal_key = (
                internal[
                    "AUPRC"
                ],
                internal[
                    "AUROC"
                ],
            )

            if (
                best_internal_key is None
                or internal_key
                > best_internal_key
            ):

                best_internal_key = (
                    internal_key
                )

                best_epoch = epoch

                best_internal_auc = (
                    internal[
                        "AUROC"
                    ]
                )

                best_internal_ap = (
                    internal[
                        "AUPRC"
                    ]
                )

                best_state = {
                    key:
                        value
                        .detach()
                        .cpu()
                        .clone()
                    for key, value
                    in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError(
                f"seed {seed}: "
                "no checkpoint selected"
            )

        model.load_state_dict(
            best_state
        )

        validation = evaluate_pairs(
            model=model,
            vectors=intra0_vectors,
            mapping=intra0_mapping,
            pairs=val_annotated,
            batch_size=(
                EVAL_BATCH_SIZE
            ),
            device=device,
        )

        all_probabilities = np.full(
            len(
                raw_val_pairs
            ),
            0.5,
            dtype=np.float64,
        )

        annotated_mask = np.zeros(
            len(
                raw_val_pairs
            ),
            dtype=bool,
        )

        for pair_index, probability in zip(
            validation[
                "pair_indices"
            ],
            validation[
                "probabilities"
            ],
        ):

            all_probabilities[
                pair_index
            ] = float(
                probability
            )

            annotated_mask[
                pair_index
            ] = True

        all_labels = np.asarray(
            [
                label
                for _, _, label
                in raw_val_pairs
            ],
            dtype=np.int64,
        )

        all_auc = auc_score(
            all_labels,
            all_probabilities,
        )

        all_ap = average_precision(
            all_labels,
            all_probabilities,
        )

        score_path = (
            OUTDIR
            / (
                "learned_seed"
                f"{seed}_intra0_pair_scores.tsv.gz"
            )
        )

        with gzip.open(
            score_path,
            "wt",
            encoding="utf-8",
        ) as handle:

            writer = csv.writer(
                handle,
                delimiter="\t",
                lineterminator="\n",
            )

            writer.writerow(
                [
                    "pair_index",
                    "protein_A",
                    "protein_B",
                    "label",
                    "both_domain_annotated",
                    "learned_domain_probability",
                    "seed",
                    "best_epoch",
                ]
            )

            for pair_index, (
                protein_A,
                protein_B,
                label,
            ) in enumerate(
                raw_val_pairs
            ):

                writer.writerow(
                    [
                        pair_index,
                        protein_A,
                        protein_B,
                        label,
                        int(
                            annotated_mask[
                                pair_index
                            ]
                        ),
                        float(
                            all_probabilities[
                                pair_index
                            ]
                        ),
                        seed,
                        best_epoch,
                    ]
                )

        checkpoint_path = (
            OUTDIR
            / (
                "learned_seed"
                f"{seed}_best.pt"
            )
        )

        torch.save(
            {
                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "internal_Intra1_AUROC":
                    best_internal_auc,

                "internal_Intra1_AUPRC":
                    best_internal_ap,

                "state_dict":
                    best_state,

                "architecture":
                    (
                        "symmetric [zA+zB, "
                        "|zA-zB|, zA*zB] "
                        "MLP with normalized "
                        "log-sum-exp MIL pooling"
                    ),

                "max_domains_per_protein":
                    MAX_DOMAINS_PER_PROTEIN,
            },
            checkpoint_path,
        )

        summaries.append(
            {
                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "Intra1_internal_AUROC":
                    best_internal_auc,

                "Intra1_internal_AUPRC":
                    best_internal_ap,

                "Intra0_annotated_pairs":
                    int(
                        annotated_mask.sum()
                    ),

                "Intra0_annotated_fraction_all":
                    float(
                        annotated_mask.mean()
                    ),

                "Intra0_annotated_AUROC":
                    validation[
                        "AUROC"
                    ],

                "Intra0_annotated_AUPRC":
                    validation[
                        "AUPRC"
                    ],

                "Intra0_all_AUROC":
                    all_auc,

                "Intra0_all_AUPRC":
                    all_ap,

                "checkpoint":
                    str(
                        checkpoint_path
                    ),

                "pair_scores":
                    str(
                        score_path
                    ),
            }
        )

        print(
            "Intra0 annotated:",
            f"AUROC="
            f"{validation['AUROC']:.6f} "
            f"AUPRC="
            f"{validation['AUPRC']:.6f}",
            flush=True,
        )

        del model

        torch.cuda.empty_cache()

    with SUMMARY_OUT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summaries[
                    0
                ].keys()
            ),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    best = max(
        summaries,
        key=lambda row: (
            row[
                "Intra0_annotated_AUROC"
            ],
            row[
                "Intra0_annotated_AUPRC"
            ],
        ),
    )

    print()
    print(
        "BEST Intra0 SEED "
        "(validation/model-selection only):",
        best[
            "seed"
        ],
        f"AUROC="
        f"{best['Intra0_annotated_AUROC']:.6f}",
        f"AUPRC="
        f"{best['Intra0_annotated_AUPRC']:.6f}",
    )

    print()
    print("WROTE:")
    print(SUMMARY_OUT)

    print()
    print(
        "LEARNED_DOMAIN_COMPATIBILITY_COMPLETE"
    )


if __name__ == "__main__":
    main()
