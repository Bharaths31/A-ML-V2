"""Entity-level stratified train/val splitting (never split pairs)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _strat_key(country: str, fanout: int) -> str:
    bucket = str(fanout) if fanout <= 3 else "4+"
    return f"{country}|{bucket}"


def _split_group(
    ids: List[str], frac: float, rng: np.random.RandomState
) -> Tuple[List[str], List[str]]:
    if frac <= 0 or len(ids) <= 1:
        return ids, []
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_val = int(round(frac * len(shuffled)))
    n_val = max(1, min(len(shuffled) - 1, n_val))
    return shuffled[n_val:], shuffled[:n_val]


def assign_entity_splits(
    s1_ids: Sequence[str],
    truth: Dict[str, Sequence[str]],
    country_by_id: Dict[str, str],
    test_frac: float = 0.2,
    calib_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign each S1 entity to 'fit', 'calib', or 'val'.

    Stratified by (country, fan-out bucket) so singleton rate is preserved.
    """
    groups: Dict[str, List[str]] = {}
    for entity in s1_ids:
        key = _strat_key(country_by_id.get(entity, "?"), len(truth.get(entity, [])))
        groups.setdefault(key, []).append(entity)

    rng = np.random.RandomState(seed)
    split: Dict[str, str] = {}
    for key in sorted(groups):
        ids = sorted(groups[key])
        train_ids, val_ids = _split_group(ids, test_frac, rng)
        fit_ids, calib_ids = _split_group(train_ids, calib_frac, rng)
        for entity in fit_ids:
            split[entity] = "fit"
        for entity in calib_ids:
            split[entity] = "calib"
        for entity in val_ids:
            split[entity] = "val"
    return split


def split_summary(split: Dict[str, str], truth: Dict[str, Sequence[str]]) -> pd.DataFrame:
    rows = []
    for name in ("fit", "calib", "val"):
        ids = [e for e, s in split.items() if s == name]
        fanouts = [len(truth.get(e, [])) for e in ids]
        singletons = sum(1 for f in fanouts if f == 0)
        rows.append(
            {
                "split": name,
                "n_entities": len(ids),
                "n_singletons": singletons,
                "singleton_rate": (singletons / len(ids)) if ids else 0.0,
                "mean_fanout": float(np.mean(fanouts)) if fanouts else 0.0,
            }
        )
    return pd.DataFrame(rows)
