"""Tester — flat module wrapper around er_core evaluate + postprocess."""
from __future__ import annotations

from er_core.evaluate.scorer import macro_f05, score_detailed
from er_core.postprocess.output_writer import write_outputs
from er_core.utils.validate_submission import validate

__all__ = ["macro_f05", "score_detailed", "write_outputs", "validate"]
