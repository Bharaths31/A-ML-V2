import itertools

import numpy as np

from src.decisions.expected_f05 import (
    decide_entity,
    monte_carlo_expected,
    plugin_expected_scores,
)


def _plugin_subset_score(p, subset, q):
    if not subset:
        return q
    s = sum(p[i] for i in subset)
    k = len(subset)
    m_hat = max(sum(p), 0.5)
    precision = s / k
    recall = s / m_hat
    f_hat = (1.25 * precision * recall) / (0.25 * precision + recall)
    return (1.0 - q) * f_hat


def test_prefix_optimality():
    p = np.array([0.9, 0.6, 0.4, 0.2])
    q = 0.1
    for k in range(1, len(p) + 1):
        best_subset = max(
            (_plugin_subset_score(p, combo, q) for combo in itertools.combinations(range(len(p)), k))
        )
        prefix = _plugin_subset_score(p, tuple(range(k)), q)
        assert abs(best_subset - prefix) < 1e-9


def test_plugin_matches_monte_carlo():
    p = np.array([0.85, 0.55, 0.3])
    q = 0.2
    _, plugin = plugin_expected_scores(p, q)
    _, mc = monte_carlo_expected(p, q, draws=4000, seed=7)
    for a, b in zip(plugin, mc):
        assert abs(a - b) < 0.05


def test_abstains_on_likely_singleton():
    cfg = {"decision": {"m_hat_floor": 0.5, "lambda": 1.0}}
    chosen, k, _ = decide_entity(["S2-1", "S2-2"], [0.15, 0.1], q=0.95, cfg=cfg)
    assert k == 0 and chosen == []


def test_predicts_when_confident():
    cfg = {"decision": {"m_hat_floor": 0.5, "lambda": 1.0}}
    chosen, k, _ = decide_entity(["S2-1", "S2-2"], [0.99, 0.2], q=0.01, cfg=cfg)
    assert k >= 1 and "S2-1" in chosen
