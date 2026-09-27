"""Global-context features computed over a full candidate-pair frame.

These are identical at training and inference time (both are computed from the
candidate structure of their own split), so there is no leakage by construction.
They deliberately exclude model probabilities (p_max/p_top1_gap/entity_median_p),
which are only available after inference and live in the decision stage.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..blocking import EXACT_BITS

GLOBAL_FEATURES = [
    "candidate_rank_by_cheap",
    "n_candidates",
    "entity_median_cheap",
    "s23_best_alt_score",
    "s23_n_s1_candidates",
    "name_idf_uniqueness",
    "layer_exact_hit_flag",
]


def compute(frame: pd.DataFrame, idf, index) -> pd.DataFrame:
    out = frame.copy()
    out = out.sort_values(["s1_id", "cheap_score"], ascending=[True, False])
    out["candidate_rank_by_cheap"] = out.groupby("s1_id").cumcount() + 1
    counts = out.groupby("s1_id")["s23_id"].transform("size")
    out["n_candidates"] = counts.astype(float)
    out["entity_median_cheap"] = out.groupby("s1_id")["cheap_score"].transform("median")
    out["layer_exact_hit_flag"] = ((out["layer_mask"] & EXACT_BITS) != 0).astype(float)

    # Soft exclusivity: how strongly this S2/S3 record is claimed by other S1 entities.
    by_s23 = out.groupby("s23_id")["cheap_score"]
    out["s23_best_alt_score"] = by_s23.transform("max")
    out["s23_n_s1_candidates"] = out.groupby("s23_id")["s1_id"].transform("nunique").astype(float)

    def idf_uniqueness(entity_id: str) -> float:
        record = index.get(entity_id)
        if record is None or not record.name_tokens:
            return 0.0
        return float(np.mean([idf.get(t, 0.0) for t in record.name_tokens]))

    out["name_idf_uniqueness"] = out["s1_id"].map(idf_uniqueness).astype(float)
    return out.reset_index(drop=True)
