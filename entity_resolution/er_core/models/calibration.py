"""Probability calibration (isotonic / sigmoid) with reliability diagnostics."""
from __future__ import annotations

import os
import pickle
from typing import Dict, List

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        self.model.fit(scores.reshape(-1, 1), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


class IdentityCalibrator:
    def fit(self, scores, labels):
        return self

    def predict(self, scores):
        return np.clip(scores, 0.0, 1.0)


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, method: str = "isotonic"):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return IdentityCalibrator()
    if method == "sigmoid":
        return SigmoidCalibrator().fit(scores, labels)
    if method == "none":
        return IdentityCalibrator()
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(scores, labels)
    return calibrator


def apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(scores), 0.0, 1.0)
    return np.clip(calibrator.predict(scores), 0.0, 1.0)


def reliability_report(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) == 0:
        return {"ece": 0.0, "bins": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []
    ece = 0.0
    n = len(scores)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        weight = float(mask.sum()) / n
        ece += weight * abs(confidence - accuracy)
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": int(mask.sum()),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": ece, "bins": bins}


def save_calibrator(calibrator, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(calibrator, handle)


def load_calibrator(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)
