"""
Profiler — flat module wrapper + the new V2 profiler/select entry points.

The concrete profiler implementation lives in `profiler_lib.py`; this module
re-exports the profiler-stage helpers so other flat modules can import them.
"""
from __future__ import annotations

from profiler_lib import (
    run_integrity,
    run_profile,
    run_select,
    run_report,
    build_profile,
    DEFAULT_PROFILE_SCHEMA,
)

__all__ = [
    "run_integrity",
    "run_profile",
    "run_select",
    "run_report",
    "build_profile",
    "DEFAULT_PROFILE_SCHEMA",
]
