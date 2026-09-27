"""Blocking — flat module wrapper around er_core.blocking."""
from __future__ import annotations

from er_core.blocking.cascade import build_candidate_pairs, compute_cheap_scores
from er_core.blocking.measure import blocking_report
from er_core.blocking import LAYER_BITS

__all__ = ["build_candidate_pairs", "compute_cheap_scores", "blocking_report", "LAYER_BITS"]
