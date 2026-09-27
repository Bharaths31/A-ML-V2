"""Candidate mass control: bounded per-layer top-k, floor, and per-entity cap.

This is what the 'smaller candidate set ranks higher' tie-breaker rewards, so
every knob here is tuned against measured blocking recall, never by feel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..blocking import EXACT_BITS, LAYER_BITS

NON_EXACT_LAYERS = [name for name, bit in LAYER_BITS.items() if not (bit & EXACT_BITS)]


def apply_mass_control(pairs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=["s1_id", "s23_id", "layer_mask", "cheap_score"])

    mass = cfg["blocking"]["mass"]
    per_layer_topk = int(mass["per_layer_topk"])
    per_entity_cap = int(mass["per_entity_cap"])
    keep_exact = bool(mass.get("keep_exact_keys", True))
    floor = float(cfg["blocking"]["tfidf_topk"].get("floor", 0.0) or 0.0)

    frame = pairs.drop_duplicates(["s1_id", "s23_id"]).reset_index(drop=True)
    frame["is_exact"] = (frame["layer_mask"] & EXACT_BITS) != 0

    keep = frame["is_exact"].copy()
    if not keep_exact:
        keep = pd.Series(False, index=frame.index)

    for name in NON_EXACT_LAYERS:
        bit = LAYER_BITS[name]
        sub = frame[(frame["layer_mask"] & bit) != 0]
        if sub.empty:
            continue
        sub = sub.sort_values(["s1_id", "cheap_score"], ascending=[True, False])
        rank = sub.groupby("s1_id").cumcount()
        selected = sub.index[rank < per_layer_topk]
        keep.loc[selected] = True

    if floor > 0.0:
        keep = keep & (frame["is_exact"] | (frame["cheap_score"] >= floor))

    frame = frame[keep].copy()
    if frame.empty:
        frame = frame.drop(columns=["is_exact"])
        return frame.reset_index(drop=True)

    frame = frame.sort_values(
        ["s1_id", "is_exact", "cheap_score"], ascending=[True, False, False]
    )
    frame["_rank"] = frame.groupby("s1_id").cumcount()
    frame = frame[frame["_rank"] < per_entity_cap]
    frame = frame.drop(columns=["is_exact", "_rank"])
    return frame.reset_index(drop=True)
