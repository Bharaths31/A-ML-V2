"""Configuration: embedded defaults, optional YAML override, deep merge."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "paths": {
        "train_dir": "dataset/train",
        "test_dir": "dataset/test",
        "out_dir": "output",
        "cache_dir": "artifacts",
    },
    "canonicalize": {"strip_punct": True, "ascii_fold": True},
    "blocking": {
        "layers": {
            "A1": True, "A2": True, "A3": True, "A5": True, "A6": True,
            "B1": True, "B2": True, "B3": True, "C1": True, "C2": True,
        },
        "minhash": {"num_perm": 128, "bands": 32, "shingle": "char3"},
        "tfidf_topk": {"k": 8, "floor": 0.0, "chunk": 128},
        "mass": {"per_layer_topk": 8, "per_entity_cap": 12, "keep_exact_keys": True},
        "rare_token": {"rare_max_df": 5, "min_token_len": 4},
    },
    "features": {"use_embeddings": False},
    "model": {
        "backend": "lightgbm",
        "lgbm": {
            "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
            "min_child_samples": 20, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "deterministic": True, "verbosity": -1,
        },
        "neg_random_cap": 200000,
        "hard_negative_rounds": 0,
    },
    "calibration": {"method": "isotonic", "fold_frac": 0.2},
    "decision": {
        "m_hat_floor": 0.5, "lambda": 1.0, "q_blend": 0.5,
        "mc_draws": 500, "use_mc_in_tests": True,
    },
    "validation": {"test_frac": 0.2, "calib_frac": 0.2, "min_positives": 5},
    "submission": {"max_candidates": 15},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load defaults, merge an optional YAML file, then optional overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as handle:
                user = yaml.safe_load(handle) or {}
            _deep_merge(cfg, user)
        except ImportError:  # pragma: no cover - yaml is a pinned dependency
            pass
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def train_dir(cfg: Dict[str, Any]) -> str:
    return cfg["paths"]["train_dir"]


def test_dir(cfg: Dict[str, Any]) -> str:
    return cfg["paths"]["test_dir"]


def cache_dir(cfg: Dict[str, Any]) -> str:
    path = cfg["paths"]["cache_dir"]
    os.makedirs(path, exist_ok=True)
    return path


def out_dir(cfg: Dict[str, Any]) -> str:
    path = cfg["paths"]["out_dir"]
    os.makedirs(path, exist_ok=True)
    return path
