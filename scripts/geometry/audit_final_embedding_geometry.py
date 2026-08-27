from pathlib import Path
import csv
import gc
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "TUNA_ROOT",
        REPO_ROOT / "external" / "TUnA-R",
    )
).resolve()

EMB = ROOT / "data/gold_standard/embeddings"

FILES = {
    "Base": (
        EMB
        / "esmc_300m_unadapted_embeddings.pt"
    ),
    "Standard": (
        EMB
        / "esmc_300m_interpro_joint64_best_joint_embeddings.pt"
    ),
    "MLM-preserved": (
        EMB
        / (
            "esmc_300m_interpro_joint64_"
            "mlm_preserved_best_joint_embeddings.pt"
        )
    ),
    "Rep-preserved alpha=10": (
        EMB
        / (
            "esmc_300m_interpro_joint64_"
            "rep10_best_joint_embeddings.pt"
        )
    ),
}

OUT = (
    ROOT
    / "results"
    / "representation_geometry"
    / "standard_mlm_rep10"
)

CACHE = OUT / "pooled_cache"

HIDDEN = 960

KS = (10, 25, 50)
MAX_K = max(KS)

KNN_BLOCK = 512

PAIR_SAMPLE_SIZE = 1_000_000
PAIR_SEED = 47
PAIR_BLOCK = 100_000


threads = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        "16",
    )
)

torch.set_num_threads(threads)


# ============================================================
# UTILITIES
# ============================================================

def source_signature(path):
    stat = path.stat()

    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def key_digest(keys):
    h = hashlib.sha256()

    for key in keys:
        h.update(
            str(key).encode(
                "utf-8",
                errors="surrogatepass",
            )
        )
        h.update(b"\0")

    return h.hexdigest()


# ============================================================
# LOAD + MEAN-POOL RESIDUE EMBEDDINGS
# ============================================================

def load_or_pool(
    label,
    path,
    expected_keys=None,
):
    safe = (
        label.lower()
        .replace(" ", "_")
        .replace("=", "")
    )

    cache_path = (
        CACHE
        / f"{safe}_mean_pooled.pt"
    )

    sig = source_signature(path)

    # --------------------------------------------------------
    # Reuse validated cache if available
    # --------------------------------------------------------

    if cache_path.is_file():
        print(
            f"\nChecking pooled cache: {cache_path}",
            flush=True,
        )

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            payload.get("source_signature")
            == sig
        ):
            keys = payload["keys"]
            pooled = payload["pooled"]

            if (
                expected_keys is not None
                and keys != expected_keys
            ):
                raise RuntimeError(
                    f"{label}: cached protein "
                    "ordering differs from Base"
                )

            if pooled.shape != (
                len(keys),
                HIDDEN,
            ):
                raise RuntimeError(
                    f"{label}: invalid cached "
                    f"shape {tuple(pooled.shape)}"
                )

            print(
                f"{label}: using validated "
                f"cache, proteins={len(keys):,}",
                flush=True,
            )

            return keys, pooled

        print(
            f"{label}: cache source signature "
            "does not match; rebuilding",
            flush=True,
        )

    # --------------------------------------------------------
    # Load residue embeddings
    # --------------------------------------------------------

    print(
        f"\nLoading residue embeddings: "
        f"{label}\n{path}",
        flush=True,
    )

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"{label}: expected embedding dict; "
            f"got {type(obj)}"
        )

    keys = sorted(obj.keys())

    if expected_keys is not None:
        if keys != expected_keys:
            base_set = set(expected_keys)
            this_set = set(keys)

            missing = (
                base_set - this_set
            )
            extra = (
                this_set - base_set
            )

            raise RuntimeError(
                f"{label}: protein set mismatch; "
                f"missing={len(missing):,}, "
                f"extra={len(extra):,}"
            )

    pooled = torch.empty(
        (len(keys), HIDDEN),
        dtype=torch.float32,
    )

    for idx, key in enumerate(keys):
        residue = obj[key]

        if (
            not torch.is_tensor(residue)
            or residue.ndim != 2
            or residue.shape[1] != HIDDEN
            or residue.shape[0] < 1
        ):
            raise RuntimeError(
                f"{label}: unexpected tensor "
                f"for key index {idx}: "
                f"{getattr(residue, 'shape', None)}"
            )

        # Stored exports contain residue-level embeddings.
        # Protein vector = mean over all stored residues.
        pooled[idx] = (
            residue
            .float()
            .mean(dim=0)
        )

        if (
            (idx + 1) % 2000 == 0
            or idx + 1 == len(keys)
        ):
            print(
                f"{label}: pooled "
                f"{idx + 1:,}/{len(keys):,}",
                flush=True,
            )

    if not torch.isfinite(pooled).all():
        raise RuntimeError(
            f"{label}: non-finite pooled values"
        )

    # Release the ~10 GB mmap before moving on.
    del obj
    gc.collect()

    CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "label": label,
        "source_signature": sig,
        "keys": keys,
        "key_sha256": key_digest(keys),
        "pooled": pooled,
    }

    tmp = cache_path.with_suffix(
        ".pt.tmp"
    )

    torch.save(
        payload,
        tmp,
    )

    tmp.replace(cache_path)

    print(
        f"{label}: pooled cache saved to "
        f"{cache_path}",
        flush=True,
    )

    return keys, pooled


# ============================================================
# EXACT COSINE KNN
# ============================================================

def exact_cosine_knn(
    normalized,
    k=MAX_K,
):
    n = normalized.shape[0]

    if k >= n:
        raise RuntimeError(
            f"k={k} invalid for n={n}"
        )

    neighbors = torch.empty(
        (n, k),
        dtype=torch.int32,
    )

    target = (
        normalized
        .T
        .contiguous()
    )

    print(
        f"Exact cosine kNN: "
        f"n={n:,}, k={k}",
        flush=True,
    )

    for start in range(
        0,
        n,
        KNN_BLOCK,
    ):
        end = min(
            start + KNN_BLOCK,
            n,
        )

        similarities = (
            normalized[start:end]
            @ target
        )

        local_rows = torch.arange(
            end - start
        )

        global_rows = torch.arange(
            start,
            end,
        )

        similarities[
            local_rows,
            global_rows,
        ] = -float("inf")

        topk = torch.topk(
            similarities,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        neighbors[start:end] = (
            topk.to(torch.int32)
        )

        del similarities
        del topk

        print(
            f"  neighbours "
            f"{end:,}/{n:,}",
            flush=True,
        )

    return neighbors


def knn_overlap(
    reference,
    comparison,
    k,
):
    ref = (
        reference[:, :k]
        .to(torch.int64)
    )

    comp = (
        comparison[:, :k]
        .to(torch.int64)
    )

    # For every Base neighbour, determine whether it
    # appears among the adapted model's top-k neighbours.
    matched = (
        ref[:, :, None]
        == comp[:, None, :]
    ).any(dim=2)

    per_protein = (
        matched
        .float()
        .mean(dim=1)
    )

    q = torch.quantile(
        per_protein,
        torch.tensor(
            [0.25, 0.50, 0.75]
        ),
    )

    return per_protein, {
        "mean": float(
            per_protein.mean()
        ),
        "std": float(
            per_protein.std(
                unbiased=False
            )
        ),
        "q25": float(q[0]),
        "median": float(q[1]),
        "q75": float(q[2]),
    }


# ============================================================
# FIXED UNIQUE PAIR SAMPLE
# ============================================================

def sample_unique_pairs(
    n,
    sample_size,
    seed,
):
    total_possible = (
        n * (n - 1) // 2
    )

    if sample_size > total_possible:
        raise RuntimeError(
            "Requested more unique pairs "
            "than exist"
        )

    rng = np.random.default_rng(
        seed
    )

    codes = np.empty(
        0,
        dtype=np.int64,
    )

    while codes.size < sample_size:
        need = (
            sample_size
            - codes.size
        )

        draw = max(
            50_000,
            int(need * 1.10),
        )

        a = rng.integers(
            0,
            n,
            size=draw,
            dtype=np.int64,
        )

        b = rng.integers(
            0,
            n,
            size=draw,
            dtype=np.int64,
        )

        lo = np.minimum(a, b)
        hi = np.maximum(a, b)

        valid = lo != hi

        new_codes = (
            lo[valid] * n
            + hi[valid]
        )

        codes = np.unique(
            np.concatenate(
                [codes, new_codes]
            )
        )

        print(
            f"Unique sampled pairs: "
            f"{min(codes.size, sample_size):,}"
            f"/{sample_size:,}",
            flush=True,
        )

    if codes.size > sample_size:
        chosen = rng.choice(
            codes.size,
            size=sample_size,
            replace=False,
        )

        codes = codes[chosen]

    i = (
        codes // n
    ).astype(
        np.int64,
        copy=False,
    )

    j = (
        codes % n
    ).astype(
        np.int64,
        copy=False,
    )

    if np.any(i >= j):
        raise RuntimeError(
            "Pair sampling invariant failed"
        )

    return i, j


def pairwise_cosines(
    normalized,
    i,
    j,
):
    m = len(i)

    result = np.empty(
        m,
        dtype=np.float32,
    )

    for start in range(
        0,
        m,
        PAIR_BLOCK,
    ):
        end = min(
            start + PAIR_BLOCK,
            m,
        )

        ii = torch.from_numpy(
            i[start:end]
        )

        jj = torch.from_numpy(
            j[start:end]
        )

        result[start:end] = (
            normalized[ii]
            * normalized[jj]
        ).sum(
            dim=1
        ).numpy()

    return result


# ============================================================
# LINEAR CKA
# ============================================================

def centered_double(x):
    x = x.to(
        dtype=torch.float64
    )

    return (
        x
        - x.mean(
            dim=0,
            keepdim=True,
        )
    )


def linear_cka(
    base_centered,
    adapted_centered,
    base_gram_norm,
):
    cross = (
        base_centered.T
        @ adapted_centered
    )

    numerator = (
        cross
        .square()
        .sum()
    )

    adapted_gram = (
        adapted_centered.T
        @ adapted_centered
    )

    adapted_norm = (
        torch.linalg.matrix_norm(
            adapted_gram,
            ord="fro",
        )
    )

    denominator = (
        base_gram_norm
        * adapted_norm
    )

    value = (
        numerator
        / denominator
    )

    return float(value)


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========================================"
    )
    print(
        "FINAL STATIC EMBEDDING GEOMETRY AUDIT"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Strict file preflight
    # --------------------------------------------------------

    for label, path in FILES.items():
        if (
            not path.is_file()
            or path.stat().st_size == 0
        ):
            raise FileNotFoundError(
                f"{label}: missing or empty\n"
                f"{path}"
            )

        print(
            f"{label:24s} "
            f"{path.stat().st_size / 1e9:.2f} GB "
            f"{path.name}",
            flush=True,
        )

    # --------------------------------------------------------
    # Pool all four encoders in identical protein order
    # --------------------------------------------------------

    keys, base_raw = load_or_pool(
        "Base",
        FILES["Base"],
    )

    pooled = {
        "Base": base_raw,
    }

    for label in (
        "Standard",
        "MLM-preserved",
        "Rep-preserved alpha=10",
    ):
        _, matrix = load_or_pool(
            label,
            FILES[label],
            expected_keys=keys,
        )

        pooled[label] = matrix

    n = len(keys)

    digest = key_digest(keys)

    print(
        "\nMatched protein set:",
        f"{n:,}",
        flush=True,
    )
    print(
        "Protein-order SHA256:",
        digest,
        flush=True,
    )

    # --------------------------------------------------------
    # Cosine-normalized matrices
    # --------------------------------------------------------

    normalized = {}

    for label, matrix in pooled.items():
        z = F.normalize(
            matrix,
            p=2,
            dim=1,
        )

        if not torch.isfinite(z).all():
            raise RuntimeError(
                f"{label}: non-finite "
                "normalized embeddings"
            )

        normalized[label] = z

    # --------------------------------------------------------
    # 1. Exact nearest-neighbour preservation
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "1. EXACT KNN PRESERVATION"
    )
    print(
        "========================================"
    )

    base_nn = exact_cosine_knn(
        normalized["Base"]
    )

    knn_results = {}

    per_protein_results = {}

    for label in (
        "Standard",
        "MLM-preserved",
        "Rep-preserved alpha=10",
    ):
        print(
            f"\nCondition: {label}",
            flush=True,
        )

        adapted_nn = exact_cosine_knn(
            normalized[label]
        )

        knn_results[label] = {}
        per_protein_results[label] = {}

        for k in KS:
            per_protein, stats = (
                knn_overlap(
                    base_nn,
                    adapted_nn,
                    k,
                )
            )

            random_expectation = (
                k / (n - 1)
            )

            stats[
                "random_expectation"
            ] = float(
                random_expectation
            )

            stats[
                "fold_over_random"
            ] = float(
                stats["mean"]
                / random_expectation
            )

            knn_results[label][
                f"k{k}"
            ] = stats

            per_protein_results[label][
                f"k{k}"
            ] = per_protein

            print(
                f"{label:24s} "
                f"k={k:2d}: "
                f"{100 * stats['mean']:.3f}% "
                f"(random "
                f"{100 * random_expectation:.3f}%)",
                flush=True,
            )

        del adapted_nn
        gc.collect()

    torch.save(
        {
            "protein_order_sha256": digest,
            "protein_count": n,
            "overlap": (
                per_protein_results
            ),
        },
        (
            OUT
            / "knn_per_protein_overlap.pt"
        ),
    )

    # --------------------------------------------------------
    # 2. Pairwise cosine-geometry Spearman correlation
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "2. PAIRWISE COSINE GEOMETRY SPEARMAN"
    )
    print(
        "========================================"
    )

    pair_i, pair_j = (
        sample_unique_pairs(
            n=n,
            sample_size=(
                PAIR_SAMPLE_SIZE
            ),
            seed=PAIR_SEED,
        )
    )

    base_pair_cos = (
        pairwise_cosines(
            normalized["Base"],
            pair_i,
            pair_j,
        )
    )

    spearman_results = {
        "Base": 1.0,
    }

    for label in (
        "Standard",
        "MLM-preserved",
        "Rep-preserved alpha=10",
    ):
        adapted_pair_cos = (
            pairwise_cosines(
                normalized[label],
                pair_i,
                pair_j,
            )
        )

        result = spearmanr(
            base_pair_cos,
            adapted_pair_cos,
        )

        rho = float(
            result.statistic
        )

        if not np.isfinite(rho):
            raise RuntimeError(
                f"{label}: invalid Spearman rho"
            )

        spearman_results[label] = rho

        print(
            f"{label:24s} "
            f"rho={rho:.6f}",
            flush=True,
        )

        del adapted_pair_cos

    # --------------------------------------------------------
    # 3. Linear CKA
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "3. LINEAR CKA"
    )
    print(
        "========================================"
    )

    base_centered = (
        centered_double(
            pooled["Base"]
        )
    )

    base_gram = (
        base_centered.T
        @ base_centered
    )

    base_gram_norm = (
        torch.linalg.matrix_norm(
            base_gram,
            ord="fro",
        )
    )

    cka_results = {
        "Base": 1.0,
    }

    for label in (
        "Standard",
        "MLM-preserved",
        "Rep-preserved alpha=10",
    ):
        adapted_centered = (
            centered_double(
                pooled[label]
            )
        )

        value = linear_cka(
            base_centered=base_centered,
            adapted_centered=(
                adapted_centered
            ),
            base_gram_norm=(
                base_gram_norm
            ),
        )

        cka_results[label] = value

        print(
            f"{label:24s} "
            f"CKA={value:.6f}",
            flush=True,
        )

        del adapted_centered
        gc.collect()

    # --------------------------------------------------------
    # Assemble paper-ready summary
    # --------------------------------------------------------

    rows = []

    rows.append(
        {
            "condition": "Base",
            "knn10_preservation": 1.0,
            "knn25_preservation": 1.0,
            "knn50_preservation": 1.0,
            "knn50_median": 1.0,
            "pairwise_cosine_spearman_rho": 1.0,
            "linear_cka": 1.0,
        }
    )

    for label in (
        "Standard",
        "MLM-preserved",
        "Rep-preserved alpha=10",
    ):
        rows.append(
            {
                "condition": label,
                "knn10_preservation": (
                    knn_results[label][
                        "k10"
                    ]["mean"]
                ),
                "knn25_preservation": (
                    knn_results[label][
                        "k25"
                    ]["mean"]
                ),
                "knn50_preservation": (
                    knn_results[label][
                        "k50"
                    ]["mean"]
                ),
                "knn50_median": (
                    knn_results[label][
                        "k50"
                    ]["median"]
                ),
                "pairwise_cosine_spearman_rho": (
                    spearman_results[label]
                ),
                "linear_cka": (
                    cka_results[label]
                ),
            }
        )

    tsv_path = (
        OUT
        / "geometry_summary.tsv"
    )

    with tsv_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
            delimiter="\t",
        )

        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protein_count": n,
        "hidden_size": HIDDEN,
        "protein_order_sha256": digest,
        "reference": "Base",
        "source_files": {
            label: (
                source_signature(path)
            )
            for label, path
            in FILES.items()
        },
        "methods": {
            "protein_representation": (
                "mean over all stored "
                "residue embeddings"
            ),
            "knn": {
                "similarity": "cosine",
                "exact": True,
                "k_values": list(KS),
                "primary_k": 50,
                "random_overlap_fraction": {
                    str(k): float(
                        k / (n - 1)
                    )
                    for k in KS
                },
            },
            "pairwise_geometry": {
                "metric": (
                    "Spearman correlation "
                    "between Base and adapted "
                    "pairwise cosine similarities"
                ),
                "unique_unordered_pairs": (
                    PAIR_SAMPLE_SIZE
                ),
                "seed": PAIR_SEED,
            },
            "cka": {
                "metric": "linear CKA",
                "input": (
                    "raw mean-pooled protein "
                    "representations"
                ),
                "centering": (
                    "feature-wise across proteins"
                ),
            },
        },
        "knn": knn_results,
        "pairwise_cosine_spearman": (
            spearman_results
        ),
        "linear_cka": cka_results,
        "summary_rows": rows,
    }

    json_path = (
        OUT
        / "geometry_summary.json"
    )

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Final console table
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "FINAL PAPER GEOMETRY SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"{'Condition':24s} "
        f"{'NN@10':>9s} "
        f"{'NN@25':>9s} "
        f"{'NN@50':>9s} "
        f"{'Spearman':>10s} "
        f"{'CKA':>10s}"
    )

    for row in rows:
        print(
            f"{row['condition']:24s} "
            f"{row['knn10_preservation']:9.4f} "
            f"{row['knn25_preservation']:9.4f} "
            f"{row['knn50_preservation']:9.4f} "
            f"{row['pairwise_cosine_spearman_rho']:10.4f} "
            f"{row['linear_cka']:10.4f}"
        )

    print(
        "========================================"
    )

    print(
        f"TSV : {tsv_path}"
    )

    print(
        f"JSON: {json_path}"
    )

    print(
        "FINAL_EMBEDDING_GEOMETRY_AUDIT_COMPLETE"
    )


if __name__ == "__main__":
    main()
