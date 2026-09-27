"""Per-entity expected-F0.5 decision engine.

Given calibrated pair probabilities p1 >= ... >= pn for one Source 1 entity and
a singleton probability q = P(Y = empty | entity evidence), choose k in {0..n}
maximising the expected per-entity F0.5. Only prefixes of the sorted
probabilities need to be evaluated (prefix optimality).

The plug-in closed form is the production path; ``monte_carlo_expected`` is the
reference implementation used in tests to verify the closed form.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def entity_f05_from_counts(tp: int, pred_size: int, truth_size: int) -> float:
    if truth_size == 0:
        return 1.0 if pred_size == 0 else 0.0
    if pred_size == 0 or tp == 0:
        return 0.0
    precision = tp / pred_size
    recall = tp / truth_size
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def plugin_expected_scores(
    p_sorted: np.ndarray, q: float, m_hat_floor: float = 0.5
) -> Tuple[float, List[float]]:
    """Return (score of abstaining, list of expected scores for k = 1..n)."""
    p = np.clip(np.asarray(p_sorted, dtype=float), 0.0, 1.0)
    n = len(p)
    if n == 0:
        return float(q), []
    m_hat = max(float(p.sum()), float(m_hat_floor))
    cumulative = np.cumsum(p)
    scores: List[float] = []
    for k in range(1, n + 1):
        tp_expected = float(cumulative[k - 1])
        if tp_expected <= 0.0:
            scores.append(0.0)
            continue
        precision = tp_expected / k
        recall = tp_expected / m_hat
        denominator = 0.25 * precision + recall
        if denominator <= 0.0:
            scores.append(0.0)
            continue
        f_hat = (1.25 * precision * recall) / denominator
        scores.append((1.0 - float(q)) * f_hat)
    return float(q), scores


def decide_entity(
    ids_sorted: Sequence[str],
    p_sorted: Sequence[float],
    q: float,
    cfg: dict,
) -> Tuple[List[str], int, float]:
    """Choose the expected-F0.5-optimal prediction set for one entity."""
    decision_cfg = cfg.get("decision", {})
    m_hat_floor = float(decision_cfg.get("m_hat_floor", 0.5))
    lam = float(decision_cfg.get("lambda", 1.0))
    p = np.clip(np.asarray(list(p_sorted), dtype=float) * lam, 0.0, 1.0)
    ids = list(ids_sorted)

    if len(p) == 0:
        return [], 0, float(q)

    abstain_score, prefix_scores = plugin_expected_scores(p, q, m_hat_floor)
    best_k = 0
    best_score = abstain_score
    for index, score in enumerate(prefix_scores, start=1):
        if score > best_score:
            best_k = index
            best_score = score
    return ids[:best_k], best_k, float(best_score)


def monte_carlo_expected(
    p_sorted: Sequence[float],
    q: float,
    draws: int = 500,
    seed: int = 42,
) -> Tuple[float, List[float]]:
    """Reference expected scores via Bernoulli simulation.

    Returns (abstain score, scores for k = 1..n). Used to validate the plug-in.
    """
    p = np.clip(np.asarray(list(p_sorted), dtype=float), 0.0, 1.0)
    n = len(p)
    if n == 0:
        return float(q), []
    rng = np.random.RandomState(seed)
    # Simulate the pair-level draws; singleton branch is modelled by mixing in
    # the empty-truth case with probability q.
    totals = np.zeros(n + 1, dtype=float)
    n_empty = 0
    for _ in range(draws):
        truth = (rng.rand(n) < p)
        truth_size = int(truth.sum())
        if truth_size == 0:
            n_empty += 1
        for k in range(0, n + 1):
            pred_size = k
            tp = int(truth[:k].sum())
            totals[k] += entity_f05_from_counts(tp, pred_size, truth_size)
    empirical = totals / draws
    # Blend: with prob q the entity is a true singleton regardless of draws.
    blended = (1.0 - q) * empirical + q * np.array(
        [1.0] + [0.0] * n, dtype=float
    )
    return float(blended[0]), [float(x) for x in blended[1:]]


def decide_frame(
    entity_pairs: Dict[str, List[Tuple[str, float]]],
    q_by_entity: Dict[str, float],
    cfg: dict,
) -> Dict[str, List[str]]:
    """Apply the decision engine to all entities. Pairs must be (id, p) lists."""
    output: Dict[str, List[str]] = {}
    for entity, pairs in entity_pairs.items():
        ordered = sorted(pairs, key=lambda item: -item[1])
        ids = [item[0] for item in ordered]
        probs = [item[1] for item in ordered]
        chosen, _, _ = decide_entity(ids, probs, q_by_entity.get(entity, 0.0), cfg)
        output[entity] = chosen
    return output
