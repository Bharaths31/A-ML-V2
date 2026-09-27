"""Orchestrates the blocking cascade and computes cheap similarity scores."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..blocking import LAYER_BITS
from ..blocking.layers_exact import exact_layer_pairs
from ..blocking.layers_lsh import minhash_layer_pairs, tfidf_topk_layer_pairs
from ..blocking.layers_safety import house_number_pairs, rare_token_pairs
from ..blocking.mass_control import apply_mass_control


def _union_masks(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["s1_id", "s23_id"])
    combined = combined.sort_values(["s1_id", "s23_id"]).reset_index(drop=True)
    keys = combined["s1_id"].astype(str) + "\x00" + combined["s23_id"].astype(str)
    changes = keys.ne(keys.shift()).to_numpy()
    starts = np.flatnonzero(changes)
    masks = np.bitwise_or.reduceat(
        combined["layer_mask"].to_numpy(dtype=np.int64), starts
    )
    out = combined.iloc[starts][["s1_id", "s23_id"]].copy()
    out["layer_mask"] = masks.astype(int)
    return out.reset_index(drop=True)


def compute_cheap_scores(
    pairs: pd.DataFrame, all_canon: pd.DataFrame
) -> pd.DataFrame:
    """TF-IDF char-3-gram cosine on canonical names for every candidate pair."""
    if pairs.empty:
        pairs = pairs.copy()
        pairs["cheap_score"] = pd.Series(dtype=float)
        return pairs
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True)
    matrix = vectorizer.fit_transform(all_canon["name_canon"].astype(str).tolist())
    id_to_row = {eid: i for i, eid in enumerate(all_canon["entity_id"].tolist())}

    s1_rows = np.array([id_to_row.get(x, -1) for x in pairs["s1_id"]], dtype=int)
    s23_rows = np.array([id_to_row.get(x, -1) for x in pairs["s23_id"]], dtype=int)
    cheap = np.zeros(len(pairs), dtype=float)
    valid = (s1_rows >= 0) & (s23_rows >= 0)
    if valid.any():
        xs = matrix[s1_rows[valid]]
        ys = matrix[s23_rows[valid]]
        cheap[valid] = np.asarray(xs.multiply(ys).sum(axis=1)).ravel()
    out = pairs.copy()
    out["cheap_score"] = cheap
    return out


def build_candidate_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    all_canon: pd.DataFrame,
    cfg: dict,
    apply_mass: bool = True,
) -> pd.DataFrame:
    """Run all enabled blocking layers, union them, and mass-control the result."""
    enabled = cfg["blocking"]["layers"]
    seed = int(cfg.get("seed", 42))

    frames: List[pd.DataFrame] = []
    if s1_df.empty or s23_df.empty:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask", "cheap_score"])

    frames.append(exact_layer_pairs(s1_df, s23_df, enabled))
    frames.append(house_number_pairs(s1_df, s23_df, enabled))
    frames.append(rare_token_pairs(s1_df, s23_df, enabled, cfg))
    frames.extend(minhash_layer_pairs(s1_df, s23_df, enabled, cfg, seed))
    frames.append(tfidf_topk_layer_pairs(s1_df, s23_df, enabled, cfg))

    pairs = _union_masks(frames)
    pairs = compute_cheap_scores(pairs, all_canon)
    if apply_mass:
        pairs = apply_mass_control(pairs, cfg)
    return pairs
