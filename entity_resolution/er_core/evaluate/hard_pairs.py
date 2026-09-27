"""Hard-pair zoo: curated regression set of deceptive pairs."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..evaluate.scorer import macro_f05


def build_zoo(
    features: pd.DataFrame,
    truth: Dict[str, Sequence[str]],
    hard_positive_quantile: float = 0.1,
    hard_negative_quantile: float = 0.9,
) -> Dict[str, object]:
    """Select hard positives (weak-looking true pairs) and hard negatives.

    Hard positives: true matches with low name cheap score.
    Hard negatives: false pairs with high name cheap score.
    """
    frame = features.copy()
    frame["is_true"] = [
        1 if s23 in set(truth.get(s1, [])) else 0
        for s1, s23 in zip(frame["s1_id"], frame["s23_id"])
    ]
    positives = frame[frame["is_true"] == 1]
    negatives = frame[frame["is_true"] == 0]
    zoo_positive_ids: List[str] = []
    zoo_negative_ids: List[str] = []
    if not positives.empty:
        threshold = positives["cheap_score"].quantile(hard_positive_quantile)
        zoo_positive_ids = positives[positives["cheap_score"] <= threshold]["s1_id"].tolist()
    if not negatives.empty:
        threshold = negatives["cheap_score"].quantile(hard_negative_quantile)
        zoo_negative_ids = negatives[negatives["cheap_score"] >= threshold]["s1_id"].tolist()
    entities = sorted(set(zoo_positive_ids) | set(zoo_negative_ids))
    return {
        "entities": entities,
        "n_entities": len(entities),
        "n_hard_positives": len(zoo_positive_ids),
        "n_hard_negatives": len(zoo_negative_ids),
    }


def zoo_score(
    decisions: Dict[str, List[str]],
    truth: Dict[str, Sequence[str]],
    zoo_entities: Sequence[str],
) -> float:
    return macro_f05(decisions, truth, entities=list(zoo_entities))
