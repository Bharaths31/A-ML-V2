"""Layer B - typo-tolerant blocking: MinHash LSH (B1/B2) and TF-IDF top-k (B3)."""
from __future__ import annotations

import zlib
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ..blocking import LAYER_BITS

_UINT64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)


def _char3_shingles(text: str) -> Set[str]:
    text = str(text or "")
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _token_shingles(text: str) -> Set[str]:
    return {t for t in str(text or "").split() if t}


def _permutations(num_perm: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, (1 << 31) - 1, size=num_perm).astype(np.uint64)
    b = rng.randint(0, (1 << 31) - 1, size=num_perm).astype(np.uint64)
    return a, b


def _signature(shingles: Set[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    best = np.full(len(a), _UINT64_MAX, dtype=np.uint64)
    if not shingles:
        return best
    hashed = np.fromiter(
        (zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF for s in sorted(shingles)),
        dtype=np.uint64,
        count=len(shingles),
    )
    chunk = 4096
    for start in range(0, len(hashed), chunk):
        xs = hashed[start : start + chunk]
        values = (a[:, None] * xs[None, :] + b[:, None]) & _UINT64_MAX
        best = np.minimum(best, values.min(axis=1))
    return best


def _signature_matrix(texts, shingle_fn, num_perm: int, seed: int) -> np.ndarray:
    a, b = _permutations(num_perm, seed)
    sigs = [_signature(shingle_fn(t), a, b) for t in texts]
    return np.vstack(sigs) if sigs else np.zeros((0, num_perm), dtype=np.uint64)


def _lsh_index_pairs(
    s1_sigs: np.ndarray, s23_sigs: np.ndarray, bands: int, seed: int
) -> List[Tuple[int, int]]:
    if s1_sigs.shape[0] == 0 or s23_sigs.shape[0] == 0:
        return []
    num_perm = s1_sigs.shape[1]
    rows = max(1, num_perm // bands)
    effective_bands = num_perm // rows
    buckets: Dict[bytes, List[int]] = {}
    for idx in range(s23_sigs.shape[0]):
        sig = s23_sigs[idx]
        for band in range(effective_bands):
            key = sig[band * rows : (band + 1) * rows].tobytes()
            buckets.setdefault(key, []).append(idx)
    seen: Set[Tuple[int, int]] = set()
    for idx in range(s1_sigs.shape[0]):
        sig = s1_sigs[idx]
        for band in range(effective_bands):
            key = sig[band * rows : (band + 1) * rows].tobytes()
            for match in buckets.get(key, ()):  # pragma: no branch
                seen.add((idx, match))
    return list(seen)


def _pairs_from_index_pairs(
    index_pairs, s1_df: pd.DataFrame, s23_df: pd.DataFrame, bit: int
) -> pd.DataFrame:
    if not index_pairs:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    s1_ids = s1_df["entity_id"].to_numpy()
    s23_ids = s23_df["entity_id"].to_numpy()
    rows = [(s1_ids[i], s23_ids[j], bit) for i, j in index_pairs]
    return pd.DataFrame(rows, columns=["s1_id", "s23_id", "layer_mask"])


def minhash_layer_pairs(
    s1_df: pd.DataFrame, s23_df: pd.DataFrame, enabled, cfg: dict, seed: int
) -> List[pd.DataFrame]:
    """Return [B1 pairs, B2 pairs] using MinHash LSH banding."""
    num_perm = int(cfg["blocking"]["minhash"]["num_perm"])
    bands = int(cfg["blocking"]["minhash"]["bands"])
    frames: List[pd.DataFrame] = []

    if enabled.get("B1", True):
        # Identical permutations must be used on both sides for signatures to be comparable.
        s1_sigs = _signature_matrix(s1_df["name_canon"], _char3_shingles, num_perm, seed)
        s23_sigs = _signature_matrix(s23_df["name_canon"], _char3_shingles, num_perm, seed)
        pairs = _lsh_index_pairs(s1_sigs, s23_sigs, bands, seed)
        frames.append(_pairs_from_index_pairs(pairs, s1_df, s23_df, LAYER_BITS["B1"]))

    if enabled.get("B2", True):
        s1_sigs = _signature_matrix(
            s1_df["addr_street_tokens"], _token_shingles, num_perm, seed + 2
        )
        s23_sigs = _signature_matrix(
            s23_df["addr_street_tokens"], _token_shingles, num_perm, seed + 2
        )
        pairs = _lsh_index_pairs(s1_sigs, s23_sigs, bands, seed)
        frames.append(_pairs_from_index_pairs(pairs, s1_df, s23_df, LAYER_BITS["B2"]))

    return frames


def tfidf_topk_layer_pairs(
    s1_df: pd.DataFrame, s23_df: pd.DataFrame, enabled, cfg: dict
) -> pd.DataFrame:
    """B3: top-k TF-IDF char-3-gram cosine neighbours per S1 entity.

    Uses a CSR row-slice top-k implementation (no sparse_dot_topn dependency),
    chunked over S1 rows to bound memory.
    """
    if not enabled.get("B3", True):
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    k = int(cfg["blocking"]["tfidf_topk"]["k"])
    chunk = int(cfg["blocking"]["tfidf_topk"]["chunk"])
    if s1_df.empty or s23_df.empty:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True)
    all_names = pd.concat([s1_df["name_canon"], s23_df["name_canon"]], ignore_index=True)
    matrix = vectorizer.fit_transform(all_names)
    n_s1 = len(s1_df)
    x_s1 = matrix[:n_s1]
    x_s23 = matrix[n_s1:]

    s1_ids = s1_df["entity_id"].to_numpy()
    s23_ids = s23_df["entity_id"].to_numpy()
    rows: List[Tuple[str, str, int]] = []
    k_eff = min(k, x_s23.shape[0])
    if k_eff <= 0:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    for start in range(0, n_s1, chunk):
        stop = min(n_s1, start + chunk)
        # Sparse product; densify one row at a time to bound peak memory.
        product = x_s1[start:stop] @ x_s23.T
        for offset in range(product.shape[0]):
            scores = np.asarray(product.getrow(offset).todense()).ravel()
            top = np.argpartition(-scores, k_eff - 1)[:k_eff]
            for j in top:
                if scores[j] > 0:
                    rows.append((s1_ids[start + offset], s23_ids[j], LAYER_BITS["B3"]))
    return pd.DataFrame(rows, columns=["s1_id", "s23_id", "layer_mask"])
