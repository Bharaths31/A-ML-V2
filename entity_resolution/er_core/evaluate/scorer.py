"""Exact per-entity F0.5 scorer and macro-averaging harness.

This is the single source of truth for model selection: no other metric is used
anywhere in the pipeline.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence


def entity_f05(pred: Iterable[str], truth: Iterable[str]) -> float:
    """F0.5 for one Source 1 entity, with challenge conventions.

    - truth empty, pred empty -> 1.0
    - truth empty, pred non-empty -> 0.0
    - truth non-empty, pred empty -> 0.0
    """
    pred_set = set(pred)
    truth_set = set(truth)
    if not truth_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set)
    recall = tp / len(truth_set)
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(
    preds: Dict[str, Sequence[str]],
    truth: Dict[str, Sequence[str]],
    entities: Optional[Iterable[str]] = None,
) -> float:
    """Macro-average F0.5 across all Source 1 entities (singletons included)."""
    if entities is None:
        entities = sorted(truth.keys())
    entity_list = list(entities)
    if not entity_list:
        return 0.0
    total = 0.0
    for entity in entity_list:
        total += entity_f05(preds.get(entity, []), truth.get(entity, []))
    return total / len(entity_list)


def score_detailed(
    preds: Dict[str, Sequence[str]],
    truth: Dict[str, Sequence[str]],
    entities: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Macro-F0.5 plus slice breakdowns by fan-out bucket and singleton flag."""
    entity_list = list(entities) if entities is not None else sorted(truth.keys())
    per_entity: Dict[str, float] = {}
    by_fanout: Dict[str, List[float]] = defaultdict(list)
    singleton_scores: List[float] = []
    per_entity_hits: Dict[str, Dict[str, object]] = {}
    for entity in entity_list:
        score = entity_f05(preds.get(entity, []), truth.get(entity, []))
        per_entity[entity] = score
        truth_ids = truth.get(entity, [])
        fanout = len(truth_ids)
        bucket = str(fanout) if fanout <= 3 else "4+"
        by_fanout[bucket].append(score)
        if fanout == 0:
            singleton_scores.append(score)
        per_entity_hits[entity] = {
            "score": score,
            "pred_size": len(set(preds.get(entity, []))),
            "truth_size": fanout,
        }
    macro = sum(per_entity.values()) / len(entity_list) if entity_list else 0.0
    slices = {
        "fanout_" + bucket: (sum(values) / len(values))
        for bucket, values in sorted(by_fanout.items())
    }
    slices["singleton"] = (
        sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
    )
    slices["n_singletons"] = len(singleton_scores)
    slices["n_entities"] = len(entity_list)
    return {"macro_f05": macro, "slices": slices, "per_entity": per_entity_hits}
