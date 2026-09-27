"""Blocking quality measurement: recall ceiling, per-layer attribution, mass."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..blocking import LAYER_BITS, mask_to_layers


def blocking_report(
    pairs: pd.DataFrame,
    truth: Dict[str, Sequence[str]],
    s1_ids: Sequence[str],
    n_s23: int,
) -> Dict[str, object]:
    """Compute recall ceiling, per-layer unique/marginal contribution, and mass."""
    total_truth = 0
    caught = 0
    per_layer_caught: Dict[str, int] = {name: 0 for name in LAYER_BITS}
    unique_single_layer: Dict[str, int] = {name: 0 for name in LAYER_BITS}
    caught_pairs = set()
    if not pairs.empty:
        caught_pairs = set(zip(pairs["s1_id"], pairs["s23_id"]))
    mask_lookup = (
        dict(zip(zip(pairs["s1_id"], pairs["s23_id"]), pairs["layer_mask"]))
        if not pairs.empty
        else {}
    )

    for s1_id in s1_ids:
        for s23_id in truth.get(s1_id, []):
            total_truth += 1
            key = (s1_id, s23_id)
            if key in caught_pairs:
                caught += 1
                layers = mask_to_layers(int(mask_lookup[key]))
                for layer in layers:
                    per_layer_caught[layer] += 1
                if len(layers) == 1:
                    unique_single_layer[layers[0]] += 1

    counts_per_s1 = (
        pairs.groupby("s1_id").size().reindex(list(s1_ids), fill_value=0)
        if not pairs.empty
        else pd.Series(0, index=list(s1_ids))
    )
    cross_product = max(1, len(s1_ids) * max(1, n_s23))
    report = {
        "n_s1_entities": len(s1_ids),
        "n_s1_with_candidates": int((counts_per_s1 > 0).sum()),
        "n_candidate_pairs": int(len(pairs)),
        "truth_pairs_total": total_truth,
        "truth_pairs_caught": caught,
        "recall_ceiling": (caught / total_truth) if total_truth else 1.0,
        "mean_candidates_per_s1": float(counts_per_s1.mean()) if len(s1_ids) else 0.0,
        "max_candidates_per_s1": int(counts_per_s1.max()) if len(s1_ids) else 0,
        "reduction_ratio": 1.0 - (len(pairs) / cross_product),
        "layer_recall": {
            name: (per_layer_caught[name] / total_truth if total_truth else 0.0)
            for name in LAYER_BITS
        },
        "layer_unique_contribution": {
            name: (unique_single_layer[name] / total_truth if total_truth else 0.0)
            for name in LAYER_BITS
        },
    }
    return report
