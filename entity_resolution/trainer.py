"""Trainer — flat module wrapper around er_core models / decisions / splits."""
from __future__ import annotations

from er_core.models.pair_model import (
    train_pair_model,
    predict_pair_proba,
    save_model as save_pair_model,
    load_model as load_pair_model,
    feature_importance,
)
from er_core.models.calibration import (
    fit_calibrator,
    apply_calibrator,
    save_calibrator,
    load_calibrator,
    reliability_report,
)
from er_core.models.singleton_model import (
    train_singleton_model,
    predict_singleton,
    save_model as save_singleton_model,
    load_model as load_singleton_model,
    build_entity_features,
)
from er_core.evaluate.splits import assign_entity_splits
from er_core.evaluate.scorer import macro_f05, score_detailed
from er_core.decisions.expected_f05 import apply_decisions
from er_core.postprocess.consistency import pool_near_duplicates
from er_core.postprocess.output_writer import write_outputs

__all__ = [
    "train_pair_model",
    "predict_pair_proba",
    "save_pair_model",
    "load_pair_model",
    "feature_importance",
    "fit_calibrator",
    "apply_calibrator",
    "save_calibrator",
    "load_calibrator",
    "reliability_report",
    "train_singleton_model",
    "predict_singleton",
    "save_singleton_model",
    "load_singleton_model",
    "build_entity_features",
    "assign_entity_splits",
    "macro_f05",
    "score_detailed",
    "apply_decisions",
    "pool_near_duplicates",
    "write_outputs",
]
