from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


TOKEN_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
    "T": 3,
    "N": 4,
}
PAD_ID = 4


def read_sequence_order(path: Path) -> list[str]:
    rows: list[tuple[int, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                rows.append((int(row[0]), row[1].strip().upper()))
    rows.sort(key=lambda item: item[0])
    expected = list(range(len(rows)))
    actual = [item[0] for item in rows]
    if actual != expected:
        raise ValueError(f"Non-contiguous node ids in {path}")
    return [item[1] for item in rows]


def read_pair_names(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0] not in mapping:
                mapping[row[0]] = len(mapping)
    return mapping


def read_pairs_mir_first(path: Path) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                pairs.append((int(row[0]), int(row[1])))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def read_pairs(
    path: Path,
    mir_name_to_id: dict[str, int] | None = None,
    circ_id_offset: int = 0,
) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                circ_id = int(row[0]) + circ_id_offset
                if row[1].lstrip("-").isdigit():
                    mir_id = int(row[1])
                else:
                    if mir_name_to_id is None or row[1] not in mir_name_to_id:
                        raise ValueError(
                            f"Unknown miRNA name {row[1]!r} in {path}"
                        )
                    mir_id = mir_name_to_id[row[1]]
                pairs.append((mir_id, circ_id))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _encode_circular(
    sequence: str,
    segment_length: int = 64,
    segment_count: int = 4,
) -> np.ndarray:
    sequence = sequence or ""
    half = segment_length // 2
    center_start = max(0, (len(sequence) - segment_length) // 2)
    prefix = sequence[:segment_length]
    center = sequence[center_start : center_start + segment_length]
    suffix = sequence[-segment_length:]
    junction = sequence[-half:] + sequence[:half]
    if segment_count == 1:
        segments = [prefix]
    elif segment_count == 2:
        segments = [prefix, junction]
    elif segment_count == 3:
        segments = [prefix, center, junction]
    elif segment_count == 4:
        segments = [prefix, center, suffix, junction]
    else:
        raise ValueError("circRNA segment_count must be between 1 and 4")

    encoded = np.full((len(segments), segment_length), PAD_ID, dtype=np.int64)
    for row, segment in enumerate(segments):
        ids = [TOKEN_TO_ID.get(base, PAD_ID) for base in segment[:segment_length]]
        encoded[row, : len(ids)] = ids
    return encoded


def _encode_linear(sequence: str, max_length: int = 96) -> np.ndarray:
    ids = [TOKEN_TO_ID.get(base, PAD_ID) for base in sequence[:max_length]]
    encoded = np.full(max_length, PAD_ID, dtype=np.int64)
    encoded[: len(ids)] = ids
    return encoded


def encode_sequences(
    mir_sequences: list[str],
    circ_sequences: list[str],
    device: torch.device,
    circ_segment_length: int = 64,
    circ_segment_count: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mir_tokens = np.stack([_encode_linear(sequence) for sequence in mir_sequences])
    circ_tokens = np.stack(
        [
            _encode_circular(
                sequence,
                segment_length=circ_segment_length,
                segment_count=circ_segment_count,
            )
            for sequence in circ_sequences
        ]
    )

    mir_mask = mir_tokens != PAD_ID
    circ_mask = circ_tokens != PAD_ID

    return (
        torch.as_tensor(mir_tokens, dtype=torch.long, device=device),
        torch.as_tensor(mir_mask, dtype=torch.bool, device=device),
        torch.as_tensor(circ_tokens, dtype=torch.long, device=device),
        torch.as_tensor(circ_mask, dtype=torch.bool, device=device),
    )


def _kmer_counts(sequence: str, k: int, circular: bool) -> np.ndarray:
    base_to_index = {
        "A": 0,
        "C": 1,
        "G": 2,
        "U": 3,
        "T": 3,
    }
    size = 4**k
    counts = np.zeros(size, dtype=np.float32)
    if not sequence:
        return counts
    extended = sequence + sequence[: k - 1] if circular and k > 1 else sequence
    values = [base_to_index.get(base, -1) for base in extended]
    for start in range(0, len(values) - k + 1):
        index = 0
        valid = True
        for base in values[start : start + k]:
            if base < 0:
                valid = False
                break
            index = index * 4 + base
        if valid:
            counts[index] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _sequence_feature(sequence: str, circular: bool) -> np.ndarray:
    sequence = sequence or ""
    blocks = [_kmer_counts(sequence, k, circular) for k in range(1, 5)]
    length = float(len(sequence))
    upper = sequence.upper()
    gc = (
        (upper.count("G") + upper.count("C")) / length
        if length
        else 0.0
    )
    au = (
        (upper.count("A") + upper.count("U") + upper.count("T")) / length
        if length
        else 0.0
    )
    entropy = 0.0
    if length:
        counts = np.asarray(
            [
                upper.count(base) / length
                for base in ("A", "C", "G", "U", "T")
            ],
            dtype=np.float32,
        )
        nonzero = counts[counts > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(5.0))
    stats = np.asarray(
        [
            np.log1p(length) / 10.0,
            gc,
            au,
            entropy,
        ],
        dtype=np.float32,
    )
    return np.concatenate([*blocks, stats]).astype(np.float32)


def encode_kmer_features(
    mir_sequences: list[str],
    circ_sequences: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mir_features = np.stack(
        [_sequence_feature(sequence, circular=False) for sequence in mir_sequences]
    )
    circ_features = np.stack(
        [_sequence_feature(sequence, circular=True) for sequence in circ_sequences]
    )
    return (
        torch.as_tensor(mir_features, dtype=torch.float32, device=device),
        torch.as_tensor(circ_features, dtype=torch.float32, device=device),
    )


def load_pooled_word2vec(
    dataset: str,
    root: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = root / "Dataset" / dataset / "EmbeddingFeature.npz"
    with np.load(path) as data:
        mir = np.asarray(data["miRNA"], dtype=np.float32)
        circ = np.asarray(data["circRNA"], dtype=np.float32)

    def pool(array: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                array.mean(axis=1),
                array.max(axis=1),
                array.std(axis=1),
            ],
            axis=1,
        ).astype(np.float32)

    return (
        torch.as_tensor(pool(mir), dtype=torch.float32, device=device),
        torch.as_tensor(pool(circ), dtype=torch.float32, device=device),
    )


def build_norm_adj(
    n_mir: int,
    n_circ: int,
    train_positive_pairs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    n_nodes = n_mir + n_circ
    rows = train_positive_pairs[:, 0].astype(np.int64)
    cols = n_mir + train_positive_pairs[:, 1].astype(np.int64)

    interaction = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_nodes, n_nodes),
    )
    adjacency = interaction + interaction.T
    adjacency = adjacency + sp.eye(n_nodes, dtype=np.float32, format="coo")
    adjacency = adjacency.tocsr()

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    np.power(degree, -0.5, out=degree_inv_sqrt, where=degree > 0)
    degree_inv_sqrt[~np.isfinite(degree_inv_sqrt)] = 0.0
    degree_matrix = sp.diags(degree_inv_sqrt)
    normalized = (degree_matrix @ adjacency @ degree_matrix).tocoo()

    indices = torch.as_tensor(
        np.vstack([normalized.row, normalized.col]),
        dtype=torch.long,
    )
    values = torch.as_tensor(normalized.data, dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(
        indices,
        values,
        size=normalized.shape,
        dtype=torch.float32,
        device=device,
    )
    return sparse.coalesce()


def build_structure_features(
    n_mir: int,
    n_circ: int,
    train_positive_pairs: np.ndarray,
    query_pairs: np.ndarray,
) -> np.ndarray:
    interaction = sp.coo_matrix(
        (
            np.ones(train_positive_pairs.shape[0], dtype=np.float32),
            (train_positive_pairs[:, 0], train_positive_pairs[:, 1]),
        ),
        shape=(n_mir, n_circ),
    ).tocsr()
    mir_degree = np.asarray(interaction.sum(axis=1)).ravel()
    circ_degree = np.asarray(interaction.sum(axis=0)).ravel()

    path_three = (interaction @ interaction.T @ interaction).tocsr()
    if query_pairs.size:
        query_mir = query_pairs[:, 0]
        query_circ = query_pairs[:, 1]
        path_count = np.asarray(path_three[query_mir, query_circ]).ravel()
        mir_values = mir_degree[query_mir]
        circ_values = circ_degree[query_circ]
    else:
        path_count = np.empty(0, dtype=np.float32)
        mir_values = np.empty(0, dtype=np.float32)
        circ_values = np.empty(0, dtype=np.float32)

    degree_product = mir_values * circ_values
    normalized_path = path_count / np.sqrt(np.maximum(degree_product, 1.0))
    return np.column_stack(
        [
            np.log1p(path_count),
            np.log1p(mir_values),
            np.log1p(circ_values),
            np.log1p(degree_product),
            normalized_path,
        ]
    ).astype(np.float32)


def validate_pair_splits(n_mir: int, n_circ: int, splits: dict[str, np.ndarray]) -> None:
    """Reject invalid IDs, repeated pairs and label/split leakage before building graphs."""
    seen: dict[tuple[int, int], str] = {}
    for name, pairs in splits.items():
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
            raise ValueError(f"{name}: expected a nonempty (N, 2) pair array")
        if np.any(pairs < 0) or np.any(pairs[:, 0] >= n_mir) or np.any(pairs[:, 1] >= n_circ):
            raise ValueError(f"{name}: pair IDs outside sequence table bounds")
        for pair in map(tuple, pairs.tolist()):
            if pair in seen:
                raise ValueError(f"Repeated pair {pair} in {seen[pair]} and {name}; duplicate, label conflict or split overlap")
            seen[pair] = name


@dataclass
class FoldData:
    dataset: str
    fold: int
    mir_sequences: list[str]
    circ_sequences: list[str]
    train_positive: np.ndarray
    train_negative: np.ndarray
    validation_positive: np.ndarray
    validation_negative: np.ndarray
    test_positive: np.ndarray
    test_negative: np.ndarray
    train_structure: np.ndarray
    validation_structure: np.ndarray
    test_structure: np.ndarray
    n_mir: int
    n_circ: int
    positive_prior: float
    norm_adj: torch.Tensor


def load_fold(dataset: str, fold: int, root: Path, device: torch.device) -> FoldData:
    dataset_root = root / "Dataset" / dataset
    fold_root = dataset_root / "5fold_CV"

    mir_sequences = read_sequence_order(dataset_root / "miRNA_order.csv")
    circ_sequences = read_sequence_order(dataset_root / "circRNA_order.csv")
    n_mir = len(mir_sequences)
    n_circ = len(circ_sequences)
    mir_name_to_id = read_pair_names(dataset_root / "miRNA.csv")

    train_positive = read_pairs(
        fold_root / f"Positive_Sample_Train{fold}.csv",
        mir_name_to_id,
        circ_id_offset=-1,
    )
    train_negative = read_pairs(
        fold_root / f"Negative_Sample_Train{fold}.csv",
        mir_name_to_id,
        circ_id_offset=-1,
    )
    validation_positive = read_pairs(
        fold_root / f"Positive_Sample_Validation{fold}.csv",
        mir_name_to_id,
        circ_id_offset=-1,
    )
    validation_negative = read_pairs(
        fold_root / f"Negative_Sample_Validation{fold}.csv",
        mir_name_to_id,
        circ_id_offset=-1,
    )
    test_positive = read_pairs(
        fold_root / f"Positive_Sample_Test{fold}.csv",
        mir_name_to_id,
        circ_id_offset=-1,
    )
    test_negative = read_pairs(
        fold_root / f"Negative_Sample_Test{fold}.csv",
        mir_name_to_id,
        circ_id_offset=-1,
    )

    validate_pair_splits(n_mir, n_circ, {
        "train_positive": train_positive, "train_negative": train_negative,
        "validation_positive": validation_positive, "validation_negative": validation_negative,
        "test_positive": test_positive, "test_negative": test_negative,
    })

    positive_prior = float(
        train_positive.shape[0] / max(1, n_mir * n_circ)
    )
    norm_adj = build_norm_adj(n_mir, n_circ, train_positive, device)
    train_structure = build_structure_features(
        n_mir,
        n_circ,
        train_positive,
        np.concatenate([train_positive, train_negative], axis=0),
    )
    validation_structure = build_structure_features(
        n_mir,
        n_circ,
        train_positive,
        np.concatenate([validation_positive, validation_negative], axis=0),
    )
    test_structure = build_structure_features(
        n_mir,
        n_circ,
        train_positive,
        np.concatenate([test_positive, test_negative], axis=0),
    )

    return FoldData(
        dataset=dataset,
        fold=fold,
        mir_sequences=mir_sequences,
        circ_sequences=circ_sequences,
        train_positive=train_positive,
        train_negative=train_negative,
        validation_positive=validation_positive,
        validation_negative=validation_negative,
        test_positive=test_positive,
        test_negative=test_negative,
        train_structure=train_structure,
        validation_structure=validation_structure,
        test_structure=test_structure,
        n_mir=n_mir,
        n_circ=n_circ,
        positive_prior=positive_prior,
        norm_adj=norm_adj,
    )


def load_root_fold(
    dataset: str,
    fold: int,
    root: Path,
    device: torch.device,
    seed: int,
    validation_fraction: float = 0.1,
) -> FoldData:
    dataset_root = root / "Dataset" / dataset

    mir_sequences = read_sequence_order(dataset_root / "miRNA_order.csv")
    circ_sequences = read_sequence_order(dataset_root / "circRNA_order.csv")
    n_mir = len(mir_sequences)
    n_circ = len(circ_sequences)

    train_positive_all = read_pairs_mir_first(
        dataset_root / f"Positive_Sample_Train{fold}.csv"
    )
    train_negative_all = read_pairs_mir_first(
        dataset_root / f"Negative_Sample_Train{fold}.csv"
    )
    test_positive = read_pairs_mir_first(
        dataset_root / f"Positive_Sample_Test{fold}.csv"
    )
    test_negative = read_pairs_mir_first(
        dataset_root / f"Negative_Sample_Test{fold}.csv"
    )

    rng = np.random.default_rng(seed + fold)
    positive_permutation = rng.permutation(train_positive_all.shape[0])
    negative_permutation = rng.permutation(train_negative_all.shape[0])
    positive_validation_size = max(
        1,
        int(train_positive_all.shape[0] * validation_fraction),
    )
    negative_validation_size = max(
        1,
        int(train_negative_all.shape[0] * validation_fraction),
    )
    positive_validation_index = positive_permutation[:positive_validation_size]
    positive_train_index = positive_permutation[positive_validation_size:]
    negative_validation_index = negative_permutation[:negative_validation_size]
    negative_train_index = negative_permutation[negative_validation_size:]

    train_positive = train_positive_all[positive_train_index]
    train_negative = train_negative_all[negative_train_index]
    validation_positive = train_positive_all[positive_validation_index]
    validation_negative = train_negative_all[negative_validation_index]

    validate_pair_splits(n_mir, n_circ, {
        "train_positive": train_positive, "train_negative": train_negative,
        "validation_positive": validation_positive, "validation_negative": validation_negative,
        "test_positive": test_positive, "test_negative": test_negative,
    })

    positive_prior = float(
        train_positive.shape[0] / max(1, n_mir * n_circ)
    )
    norm_adj = build_norm_adj(n_mir, n_circ, train_positive, device)
    train_structure = build_structure_features(
        n_mir,
        n_circ,
        train_positive,
        np.concatenate([train_positive, train_negative], axis=0),
    )
    validation_structure = build_structure_features(
        n_mir,
        n_circ,
        train_positive,
        np.concatenate([validation_positive, validation_negative], axis=0),
    )
    test_structure = build_structure_features(
        n_mir,
        n_circ,
        train_positive,
        np.concatenate([test_positive, test_negative], axis=0),
    )

    return FoldData(
        dataset=dataset,
        fold=fold,
        mir_sequences=mir_sequences,
        circ_sequences=circ_sequences,
        train_positive=train_positive,
        train_negative=train_negative,
        validation_positive=validation_positive,
        validation_negative=validation_negative,
        test_positive=test_positive,
        test_negative=test_negative,
        train_structure=train_structure,
        validation_structure=validation_structure,
        test_structure=test_structure,
        n_mir=n_mir,
        n_circ=n_circ,
        positive_prior=positive_prior,
        norm_adj=norm_adj,
    )
