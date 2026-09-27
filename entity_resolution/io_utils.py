"""IO utilities — flat module wrapper around er_core.utils.io."""
from __future__ import annotations

from er_core.utils.io import (
    SOURCE_COLS,
    GT_COLS,
    MATCH_COLS,
    CAND_COLS,
    ensure_dir,
    read_source_tsv,
    split_id_list,
    read_ground_truth,
    write_submission_tsv,
    format_id_list,
    save_table,
    load_table,
    save_json,
    load_json,
)

__all__ = [
    "SOURCE_COLS",
    "GT_COLS",
    "MATCH_COLS",
    "CAND_COLS",
    "ensure_dir",
    "read_source_tsv",
    "split_id_list",
    "read_ground_truth",
    "write_submission_tsv",
    "format_id_list",
    "save_table",
    "load_table",
    "save_json",
    "load_json",
]
