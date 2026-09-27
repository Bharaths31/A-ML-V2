"""Entity-level singleton classifier: q = P(Y = empty | entity-level evidence)."""
from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np
import pandas as pd

from ..blocking import EXACT_BITS

SINGLETON_FEATURES = [
    "n_candidates",
    "p_max",
    "p_gap",
    "p_sum",
    "cheap_top",
    "cheap_mean",
    "exact_hit",
    "name_idf_uniqueness",
    "country_agree",
    "component_completeness",
]


def build_entity_features(
    pair_scores: np.ndarray,
    features: pd.DataFrame,
    s1_ids: List[str],
) -> pd.DataFrame:
    """Aggregate pair-level evidence into one row per S1 entity.

    ``features`` must contain at least: s1_id, cheap_score, layer_mask,
    name_idf_uniqueness, same_country, component_presence_agreement.
    """
    frame = features.copy()
    frame["p"] = np.asarray(pair_scores, dtype=float)
    frame["exact_hit"] = ((frame["layer_mask"] & EXACT_BITS) != 0).astype(float)
    for column in ("name_idf_uniqueness", "same_country", "component_presence_agreement"):
        if column not in frame.columns:
            frame[column] = 0.0

    grouped = {s1: sub for s1, sub in frame.groupby("s1_id")}
    records = []
    for s1_id in s1_ids:
        sub = grouped.get(s1_id)
        if sub is None or sub.empty:
            records.append(
                {
                    "s1_id": s1_id,
                    "n_candidates": 0.0,
                    "p_max": 0.0,
                    "p_gap": 0.0,
                    "p_sum": 0.0,
                    "cheap_top": 0.0,
                    "cheap_mean": 0.0,
                    "exact_hit": 0.0,
                    "name_idf_uniqueness": 0.0,
                    "country_agree": 0.0,
                    "component_completeness": 0.0,
                }
            )
            continue
        probabilities = np.sort(sub["p"].to_numpy(dtype=float))[::-1]
        p_max = float(probabilities[0])
        p_gap = float(p_max - probabilities[1]) if len(probabilities) > 1 else p_max
        cheap = sub["cheap_score"].to_numpy(dtype=float)
        records.append(
            {
                "s1_id": s1_id,
                "n_candidates": float(len(sub)),
                "p_max": p_max,
                "p_gap": p_gap,
                "p_sum": float(probabilities.sum()),
                "cheap_top": float(cheap.max()),
                "cheap_mean": float(cheap.mean()),
                "exact_hit": float(sub["exact_hit"].max()),
                "name_idf_uniqueness": float(sub["name_idf_uniqueness"].iloc[0]),
                "country_agree": float(sub["same_country"].max()),
                "component_completeness": float(
                    sub["component_presence_agreement"].max()
                ),
            }
        )
    return pd.DataFrame(records)


class ConstantSingleton:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, X):
        import numpy as _np

        n = len(X)
        p = self.probability
        return _np.column_stack([_np.full(n, 1.0 - p), _np.full(n, p)])


def train_singleton_model(entity_features: pd.DataFrame, labels: np.ndarray, cfg: dict):
    from lightgbm import LGBMClassifier

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return ConstantSingleton(float(labels.mean()))

    params = cfg["model"]["lgbm"]
    model = LGBMClassifier(
        n_estimators=min(300, int(params["n_estimators"])),
        learning_rate=float(params["learning_rate"]),
        num_leaves=max(7, int(params["num_leaves"]) // 4),
        min_child_samples=int(params["min_child_samples"]),
        random_state=int(cfg.get("seed", 42)),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(entity_features[SINGLETON_FEATURES].to_numpy(dtype=float), labels)
    return model


def predict_singleton(model, entity_features: pd.DataFrame) -> np.ndarray:
    if entity_features.empty:
        return np.zeros(0, dtype=float)
    X = entity_features[SINGLETON_FEATURES].to_numpy(dtype=float)
    return model.predict_proba(X)[:, 1]


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)
