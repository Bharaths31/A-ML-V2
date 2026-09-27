"""Layer A - exact composite-key blocking (vectorized hash joins)."""
from __future__ import annotations

from typing import List

import pandas as pd

from ..blocking import LAYER_BITS


def _merge_on_key(s1_keys: pd.DataFrame, s23_keys: pd.DataFrame, bit: int) -> pd.DataFrame:
    if s1_keys.empty or s23_keys.empty:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    merged = s1_keys.merge(s23_keys, on="key", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    out = merged[["entity_id_x", "entity_id_y"]].rename(
        columns={"entity_id_x": "s1_id", "entity_id_y": "s23_id"}
    )
    out["layer_mask"] = bit
    return out


def _key_frame(df: pd.DataFrame, key_col) -> pd.DataFrame:
    cols = df[["entity_id"]].copy()
    cols["key"] = key_col
    cols = cols[cols["key"].astype(str).str.len() > 0]
    return cols


def exact_layer_pairs(s1_df: pd.DataFrame, s23_df: pd.DataFrame, enabled) -> pd.DataFrame:
    """Build candidate pairs from exact-key layers A1, A2, A3, A5, A6."""
    frames: List[pd.DataFrame] = []

    if enabled.get("A1", True):
        frames.append(
            _merge_on_key(
                _key_frame(s1_df, s1_df["name_canon"]),
                _key_frame(s23_df, s23_df["name_canon"]),
                LAYER_BITS["A1"],
            )
        )
    if enabled.get("A2", True):
        frames.append(
            _merge_on_key(
                _key_frame(s1_df, s1_df["name_nosuffix"]),
                _key_frame(s23_df, s23_df["name_nosuffix"]),
                LAYER_BITS["A2"],
            )
        )
    if enabled.get("A3", True):
        frames.append(
            _merge_on_key(
                _key_frame(s1_df, s1_df["name_sortedtok"]),
                _key_frame(s23_df, s23_df["name_sortedtok"]),
                LAYER_BITS["A3"],
            )
        )
    if enabled.get("A5", True):
        s1_key = (
            s1_df["addr_pin"].astype(str) + "|" + s1_df["first_name_token"].astype(str)
        )
        s23_key = (
            s23_df["addr_pin"].astype(str) + "|" + s23_df["first_name_token"].astype(str)
        )
        mask = (s1_df["has_pin"] == 1) & (s1_df["first_name_token"].astype(str).str.len() > 0)
        mask23 = (s23_df["has_pin"] == 1) & (s23_df["first_name_token"].astype(str).str.len() > 0)
        frames.append(
            _merge_on_key(
                _key_frame(s1_df[mask], s1_key[mask]),
                _key_frame(s23_df[mask23], s23_key[mask23]),
                LAYER_BITS["A5"],
            )
        )
    if enabled.get("A6", True):
        s1_key = (
            s1_df["addr_city"].astype(str) + "|" + s1_df["first_name_token"].astype(str)
        )
        s23_key = (
            s23_df["addr_city"].astype(str) + "|" + s23_df["first_name_token"].astype(str)
        )
        mask = (s1_df["has_city"] == 1) & (s1_df["first_name_token"].astype(str).str.len() > 0)
        mask23 = (s23_df["has_city"] == 1) & (s23_df["first_name_token"].astype(str).str.len() > 0)
        frames.append(
            _merge_on_key(
                _key_frame(s1_df[mask], s1_key[mask]),
                _key_frame(s23_df[mask23], s23_key[mask23]),
                LAYER_BITS["A6"],
            )
        )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask"])
    return pd.concat(frames, ignore_index=True)
