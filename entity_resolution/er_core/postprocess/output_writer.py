"""Output writer with hard invariants and validator integration."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import pandas as pd

from ..utils import io
from ..utils.validate_submission import validate


def _candidate_map(pairs: pd.DataFrame, cap: int) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if pairs is None or pairs.empty:
        return result
    frame = pairs.sort_values(["s1_id", "cheap_score"], ascending=[True, False])
    for s1_id, sub in frame.groupby("s1_id"):
        ids = [x for x in sub["s23_id"].tolist() if x]
        seen = []
        for value in ids:
            if value not in seen:
                seen.append(value)
        result[str(s1_id)] = seen[:cap]
    return result


def write_outputs(
    test_source1_ids: Sequence[str],
    decisions: Dict[str, List[str]],
    candidate_pairs: pd.DataFrame,
    valid_s23: set,
    out_dir: str,
    cfg: dict,
) -> Dict[str, object]:
    cap = int(cfg.get("submission", {}).get("max_candidates", 15))
    cand_map = _candidate_map(candidate_pairs, cap)

    match_rows = []
    cand_rows = []
    errors: List[str] = []
    s1_list = list(dict.fromkeys(str(x) for x in test_source1_ids))

    for s1_id in s1_list:
        candidates = [x for x in cand_map.get(s1_id, []) if x in valid_s23]
        matched = [x for x in decisions.get(s1_id, []) if x in valid_s23]
        matched = [x for x in matched if x in set(candidates)]
        match_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": io.format_id_list(matched)})
        cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": io.format_id_list(candidates)})

    match_df = pd.DataFrame(match_rows)
    cand_df = pd.DataFrame(cand_rows)

    # Invariants asserted in code before writing.
    if len(match_df) != len(s1_list):
        errors.append("matching_results: row count does not equal number of test S1 entities")
    if match_df["source1_entity_id"].duplicated().any():
        errors.append("matching_results: duplicate source1_entity_id rows")
    if cand_df["source1_entity_id"].duplicated().any():
        errors.append("candidate_pairs: duplicate source1_entity_id rows")

    os.makedirs(out_dir, exist_ok=True)
    matching_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")
    io.write_submission_tsv(match_df, matching_path, "matched_entity_ids")
    io.write_submission_tsv(cand_df, candidate_path, "candidate_entity_ids")

    validation_issues = validate(matching_path, candidate_path, cfg["paths"]["test_dir"])
    if validation_issues:
        errors.extend(validation_issues)

    return {
        "matching_path": matching_path,
        "candidate_path": candidate_path,
        "n_rows": len(match_df),
        "n_matches_total": int(sum(len(io.split_id_list(v)) for v in match_df["matched_entity_ids"])),
        "n_candidates_total": int(
            sum(len(io.split_id_list(v)) for v in cand_df["candidate_entity_ids"])
        ),
        "errors": errors,
        "validation_passed": not validation_issues,
    }
