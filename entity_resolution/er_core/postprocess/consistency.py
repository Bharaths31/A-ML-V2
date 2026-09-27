"""Consistency post-processing: near-duplicate probability pooling."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..decisions.expected_f05 import decide_frame


def pool_near_duplicates(
    pair_frame: pd.DataFrame,
    canon_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Max-pool probabilities of (near-)identical S2/S3 candidates within an S1 list.

    Two candidate records from the same source with identical canonical
    name+address are almost certainly the same real-world business. If one is a
    likely match, propagate the confidence to its twin.
    """
    if pair_frame.empty:
        return pair_frame
    canon_lookup = canon_df.set_index("entity_id")
    frame = pair_frame.copy()

    def canon_key(entity_id: str) -> str:
        if entity_id not in canon_lookup.index:
            return entity_id
        row = canon_lookup.loc[entity_id]
        return f"{row['name_canon']}||{row['addr_canon']}"

    frame["dup_key"] = frame["s23_id"].map(canon_key)
    pooled = frame.groupby(["s1_id", "dup_key"])["p"].transform("max")
    frame["p_pooled"] = np.maximum(frame["p"], pooled)
    # Never let pooling exceed the strongest evidence by more than a hair; clamp.
    frame["p_pooled"] = np.clip(frame["p_pooled"], 0.0, 1.0)
    return frame


def transitive_guard(pair_frame: pd.DataFrame) -> Dict[str, object]:
    """Report (do not hard-constrain) S2/S3 records claimed by multiple S1s."""
    if pair_frame.empty:
        return {"multi_parent_records": 0, "sample": []}
    chosen = pair_frame[pair_frame["p"] >= 0.5]
    counts = chosen.groupby("s23_id")["s1_id"].nunique()
    multi = counts[counts > 1]
    return {
        "multi_parent_records": int(len(multi)),
        "sample": multi.index.tolist()[:10],
    }


def apply_decisions(
    pair_frame: pd.DataFrame,
    q_by_entity: Dict[str, float],
    cfg: dict,
    prob_column: str = "p",
) -> Dict[str, List[str]]:
    entity_pairs: Dict[str, List[tuple]] = {}
    for s1_id, sub in pair_frame.groupby("s1_id"):
        entity_pairs[s1_id] = list(zip(sub["s23_id"], sub[prob_column]))
    return decide_frame(entity_pairs, q_by_entity, cfg)
