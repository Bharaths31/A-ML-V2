"""Calibrated pairwise match model (LightGBM, sklearn fallback)."""
from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..features.build import FEATURE_COLUMNS


class ConstantProba:
    """Fallback predictor when training labels have a single class."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, X):
        n = len(X)
        p = self.probability
        return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])


def _make_estimator(cfg: dict, positive_weight: float):
    backend = cfg["model"].get("backend", "lightgbm")
    params = cfg["model"]["lgbm"]
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            lgbm_kwargs = dict(
                n_estimators=int(params["n_estimators"]),
                learning_rate=float(params["learning_rate"]),
                num_leaves=int(params["num_leaves"]),
                min_child_samples=int(params["min_child_samples"]),
                colsample_bytree=float(params["feature_fraction"]),
                subsample=float(params["bagging_fraction"]),
                subsample_freq=1,
                scale_pos_weight=positive_weight,
                deterministic=bool(params.get("deterministic", True)),
                random_state=int(cfg.get("seed", 42)),
                n_jobs=-1,
                verbosity=int(params.get("verbosity", -1)),
            )
            device = str(params.get("device", "auto")).lower()
            if device == "gpu":
                lgbm_kwargs["device_type"] = "gpu"
                lgbm_kwargs["gpu_platform_id"] = int(params.get("gpu_platform_id", 0))
                lgbm_kwargs["gpu_device_id"] = int(params.get("gpu_device_id", 0))
            return LGBMClassifier(**lgbm_kwargs)
        except ImportError:  # pragma: no cover
            pass
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["num_leaves"]),
        random_state=int(cfg.get("seed", 42)),
    )


def train_pair_model(features: pd.DataFrame, cfg: dict):
    X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return ConstantProba(float(y.mean()))
    positives = max(1, int((y == 1).sum()))
    negatives = max(1, int((y == 0).sum()))
    positive_weight = negatives / positives
    model = _make_estimator(cfg, positive_weight)
    model.fit(X, y)
    return model


def predict_pair_proba(model, features: pd.DataFrame) -> np.ndarray:
    if features.empty:
        return np.zeros(0, dtype=float)
    X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def feature_importance(model, top_n: int = 25):
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(-values)[:top_n]
        return [
            {"feature": FEATURE_COLUMNS[i], "importance": float(values[i])} for i in order
        ]
    return []


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)
