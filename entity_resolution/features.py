"""Features — flat module wrapper around er_core.features.build."""
from __future__ import annotations

from er_core.features.build import (
    assemble_features,
    FEATURE_COLUMNS,
    ID_COLUMNS,
)

__all__ = ["assemble_features", "FEATURE_COLUMNS", "ID_COLUMNS"]
