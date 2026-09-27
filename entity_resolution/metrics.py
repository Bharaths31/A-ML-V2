"""Metrics — flat module wrapper around er_core.evaluate.scorer."""
from __future__ import annotations

from er_core.evaluate.scorer import (
    entity_f05,
    macro_f05,
    score_detailed,
)

__all__ = ["entity_f05", "macro_f05", "score_detailed"]
