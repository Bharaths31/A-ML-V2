"""Layer C - safety-net blockers: rare-token and house-number anchors."""
from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd

from ..blocking import LAYER_BITS


def _token_df(names) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for name in names:
        for token in set(str(name).split()):
            if token:
                df[token] = df.get(token, 0) + 1
    return df


def rare_token_pairs(
    s1_df: pd.DataFrame, s23_df: pd.DataFrame, enabled, cfg: dict
) -> pd.DataFrame:
    """C1: (country, rare name token) inverted index."""
    if not enabled.get("C1", True):
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    rare_max_df = int(cfg["blocking"]["rare_token"]["rare_max_df"])
    min_len = int(cfg["blocking"]["rare_token"]["min_token_len"])

    token_counts = _token_df(pd.concat([s1_df["name_canon"], s23_df["name_canon"]]))
    rare_tokens: Set[str] = {
        token
        for token, count in token_counts.items()
        if count <= rare_max_df and len(token) >= min_len
    }

    index: Dict[str, List[str]] = {}
    for country, tokens, entity in zip(
        s23_df["country"].astype(str),
        s23_df["name_canon"],
        s23_df["entity_id"],
    ):
        for token in set(str(tokens).split()):
            if token in rare_tokens:
                index.setdefault(f"{country}|{token}", []).append(entity)

    rows: List[tuple] = []
    for country, tokens, s1_id in zip(
        s1_df["country"].astype(str),
        s1_df["name_canon"],
        s1_df["entity_id"],
    ):
        for token in set(str(tokens).split()):
            if token in rare_tokens:
                for s23_id in index.get(f"{country}|{token}", ()):  # pragma: no branch
                    rows.append((s1_id, s23_id, LAYER_BITS["C1"]))
    return pd.DataFrame(rows, columns=["s1_id", "s23_id", "layer_mask"])


def house_number_pairs(
    s1_df: pd.DataFrame, s23_df: pd.DataFrame, enabled
) -> pd.DataFrame:
    """C2: (house number, first name token) exact anchor."""
    if not enabled.get("C2", True):
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    s1_mask = (s1_df["has_house"] == 1) & (s1_df["first_name_token"].str.len() > 0)
    s23_mask = (s23_df["has_house"] == 1) & (s23_df["first_name_token"].str.len() > 0)
    if not s1_mask.any() or not s23_mask.any():
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    left = pd.DataFrame(
        {
            "key": s1_df.loc[s1_mask, "addr_house_no"].astype(str)
            + "|"
            + s1_df.loc[s1_mask, "first_name_token"].astype(str),
            "s1_id": s1_df.loc[s1_mask, "entity_id"],
        }
    )
    right = pd.DataFrame(
        {
            "key": s23_df.loc[s23_mask, "addr_house_no"].astype(str)
            + "|"
            + s23_df.loc[s23_mask, "first_name_token"].astype(str),
            "s23_id": s23_df.loc[s23_mask, "entity_id"],
        }
    )
    merged = left.merge(right, on="key", how="inner")[["s1_id", "s23_id"]]
    merged["layer_mask"] = LAYER_BITS["C2"]
    return merged
