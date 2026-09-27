"""Normalizer — flat module wrapper around er_core.normalize.canonicalize."""
from __future__ import annotations

from er_core.normalize.canonicalize import (
    canonicalize_frame,
    load_assets,
    AssetBundle,
    Record,
)

__all__ = ["canonicalize_frame", "load_assets", "AssetBundle", "Record"]
