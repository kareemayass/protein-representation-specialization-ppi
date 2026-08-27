import os
from pathlib import Path
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "TUNA_ROOT",
        REPO_ROOT / "external" / "TUnA-R",
    )
).resolve()

OUT = (
    ROOT
    / "results"
    / "representation_geometry"
    / "standard_mlm_rep10"
)

CACHE = OUT / "pooled_cache"

CACHE_FILES = {
    "Base":
        CACHE / "base_mean_pooled.pt",

    "Standard":
        CACHE / "standard_mean_pooled.pt",

    "MLM-preserved":
        CACHE / "mlm-preserved_mean_pooled.pt",

    "Rep-preserved alpha=10":
        CACHE / "rep-preserved_alpha10_mean_pooled.pt",
}

PAIR_FILES = {
    "Intra1":
        ROOT
        / "data/gold_standard"
        / "Intra1_interaction_1500_or_less.tsv",

    "Intra0":
        ROOT
        / "data/gold_standard"
        / "Intra0_interaction_1500_or_less.tsv",

    "Intra2":
        ROOT
        / "data/gold_standard"
        / "Intra2_interaction.tsv",
}

CONDITIONS = (
    "Base",
    "Standard",
    "MLM-preserved",
    "Rep-preserved alpha=10",
)

BLOCK = 100_000


def load_cache(label, expected_keys=None):
    path = CACHE_FILES[label]

    if not path.is_file():
        raise FileNotFoundError(
            f"{label}: missing pooled cache\n{path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    keys = payload["keys"]
    x = payload["pooled"].float()

    if expected_keys is not None and keys != expected_keys:
        raise RuntimeError(
            f"{label}: protein ordering differs from Base"
        )

    if x.ndim != 2 or x.shape[1] != 960:
        raise RuntimeError(
            f"{label}: unexpected pooled shape {tuple(x.shape)}"
        )

    if len(keys) != x.shape[0]:
        raise RuntimeError(
            f"{label}: key/matrix length mismatch"
        )

    return keys, F.normalize(x, p=2, dim=1)


def load_pairs(path, protein_to_idx):
    ii = []
    jj = []
    labels = []

    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()

            if len(fields) != 3:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"expected 3 fields, got {len(fields)}"
                )

            protein_a, protein_b, label = fields

            if label not in {"0", "1"}:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"invalid label {label!r}"
                )

            if protein_a not in protein_to_idx:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"missing embedding for {protein_a}"
                )

            if protein_b not in protein_to_idx:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"missing embedding for {protein_b}"
                )

            ii.append(protein_to_idx[protein_a])
            jj.append(protein_to_idx[protein_b])
            labels.append(int(label))

    i = np.asarray(ii, dtype=np.int64)
    j = np.asarray(jj, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)

    n = len(protein_to_idx)

    lo = np.minimum(i, j)
    hi = np.maximum(i, j)

    pair_codes = lo * n + hi
    unique_pairs = np.unique(pair_codes).size

    metadata = {
        "rows": int(len(i)),
        "unique_unordered_pairs": int(unique_pairs),
        "duplicate_rows_by_unordered_pair": int(
            len(i) - unique_pairs
        ),
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
        "self_pairs": int((i == j).sum()),
    }

    return i, j, metadata


def pair_cosines(x, i, j):
    result = np.empty(
        len(i),
        dtype=np.float32,
    )

    for start in range(0, len(i), BLOCK):
        end = min(
            start + BLOCK,
            len(i),
        )

        ti = torch.from_numpy(i[start:end])
        tj = torch.from_numpy(j[start:end])

        result[start:end] = (
            x[ti] * x[tj]
        ).sum(dim=1).numpy()

    return result


def main():
    print(
        "========================================"
    )
    print(
        "BERNETT PPI-PAIR GEOMETRY"
    )
    print(
        "========================================"
    )

    keys, base = load_cache("Base")

    matrices = {
        "Base": base,
    }

    for label in CONDITIONS[1:]:
        _, matrix = load_cache(
            label,
            expected_keys=keys,
        )
        matrices[label] = matrix

    protein_to_idx = {
        protein: idx
        for idx, protein in enumerate(keys)
    }

    results = {}
    split_metadata = {}

    for split, path in PAIR_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        i, j, metadata = load_pairs(
            path,
            protein_to_idx,
        )

        split_metadata[split] = metadata

        print()
        print(
            f"{split}: "
            f"{metadata['rows']:,} rows | "
            f"{metadata['positive_rows']:,} positive | "
            f"{metadata['negative_rows']:,} negative | "
            f"{metadata['unique_unordered_pairs']:,} "
            f"unique unordered pairs"
        )

        base_cos = pair_cosines(
            matrices["Base"],
            i,
            j,
        )

        results[split] = {
            "Base": 1.0,
        }

        for label in CONDITIONS[1:]:
            adapted_cos = pair_cosines(
                matrices[label],
                i,
                j,
            )

            rho = float(
                spearmanr(
                    base_cos,
                    adapted_cos,
                ).statistic
            )

            if not np.isfinite(rho):
                raise RuntimeError(
                    f"{split}/{label}: "
                    "non-finite Spearman rho"
                )

            results[split][label] = rho

            print(
                f"  {label:24s} "
                f"rho={rho:.6f}"
            )

    # --------------------------------------------------------
    # Merge into the existing global-geometry table
    # --------------------------------------------------------

    original = OUT / "geometry_summary.tsv"

    if not original.is_file():
        raise FileNotFoundError(
            f"Missing global summary: {original}"
        )

    with original.open() as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )
        rows = list(reader)
        original_fields = list(
            reader.fieldnames or []
        )

    expected_conditions = set(CONDITIONS)
    observed_conditions = {
        row["condition"]
        for row in rows
    }

    if observed_conditions != expected_conditions:
        raise RuntimeError(
            "Global summary conditions differ "
            f"from expected: {observed_conditions}"
        )

    new_fields = [
        "intra1_pair_cosine_spearman_rho",
        "intra0_pair_cosine_spearman_rho",
        "intra2_pair_cosine_spearman_rho",
    ]

    for row in rows:
        label = row["condition"]

        row[
            "intra1_pair_cosine_spearman_rho"
        ] = results["Intra1"][label]

        row[
            "intra0_pair_cosine_spearman_rho"
        ] = results["Intra0"][label]

        row[
            "intra2_pair_cosine_spearman_rho"
        ] = results["Intra2"][label]

    combined = (
        OUT
        / "geometry_summary_with_ppi.tsv"
    )

    with combined.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                original_fields
                + new_fields
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = (
        OUT
        / "ppi_pair_geometry.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "pair_files": {
                    split: str(path)
                    for split, path
                    in PAIR_FILES.items()
                },
                "split_metadata":
                    split_metadata,
                "spearman":
                    results,
                "definition": (
                    "Spearman correlation between "
                    "Base-ESMC and adapted-model "
                    "cosine similarities for the "
                    "actual Bernett PPI pairs in "
                    "each split."
                ),
            },
            indent=2,
        )
    )

    print()
    print(
        "========================================"
    )
    print(
        "PPI-PAIR GEOMETRY SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"{'Condition':24s} "
        f"{'Intra1':>10s} "
        f"{'Intra0':>10s} "
        f"{'Intra2':>10s}"
    )

    for label in CONDITIONS:
        print(
            f"{label:24s} "
            f"{results['Intra1'][label]:10.4f} "
            f"{results['Intra0'][label]:10.4f} "
            f"{results['Intra2'][label]:10.4f}"
        )

    print(
        "========================================"
    )

    print(
        f"Combined summary: {combined}"
    )

    print(
        "PPI_PAIR_GEOMETRY_COMPLETE"
    )


if __name__ == "__main__":
    main()
