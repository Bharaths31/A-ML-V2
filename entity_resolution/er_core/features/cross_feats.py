"""Cross-channel features (country and source)."""
from __future__ import annotations

from typing import Dict

from ..features.index import Record

CROSS_FEATURES = ["same_country", "source_flag"]


def compute(a: Record, b: Record) -> Dict[str, float]:
    return {
        "same_country": float(a.country == b.country),
        "source_flag": 0.0 if b.source == "S2" else 1.0,
    }
