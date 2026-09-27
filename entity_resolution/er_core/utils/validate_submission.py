#!/usr/bin/env python3
"""Standalone submission format validator (stdlib only).

Mirrors the rules enforced by the challenge-provided
``utils/validate_submission.py`` so submissions can be checked locally.

Usage:
    python3 validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Prints PASS (exit 0) or a numbered list of issues (exit 1).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Set, Tuple


def _read_tsv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    rows = [line.split("\t") for line in lines if line != ""]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _read_ids(path: str) -> Set[str]:
    header, rows = _read_tsv(path)
    if not header:
        raise ValueError(f"{path} is empty")
    id_col = 0
    ids = set()
    for row in rows:
        if row and len(row) > id_col:
            ids.add(row[id_col].strip())
    return ids


def _validate_file(
    path: str,
    expected_cols: List[str],
    test_s1: Set[str],
    valid_s23: Set[str],
    label: str,
    issues: List[str],
) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    if not os.path.exists(path):
        issues.append(f"{label}: file not found: {path}")
        return parsed
    try:
        header, rows = _read_tsv(path)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"{label}: cannot read file ({exc})")
        return parsed
    if header != expected_cols:
        issues.append(
            f"{label}: header must be {expected_cols} (tab-separated), found {header}"
        )
        return parsed

    for line_no, row in enumerate(rows, start=2):
        if len(row) != 2:
            issues.append(f"{label}: line {line_no} has {len(row)} columns, expected 2")
            continue
        s1 = row[0].strip()
        raw = row[1].strip()
        if not s1:
            issues.append(f"{label}: line {line_no} has empty source1_entity_id")
            continue
        if s1 in parsed:
            issues.append(f"{label}: duplicate source1_entity_id {s1} (line {line_no})")
            continue
        if s1 not in test_s1:
            issues.append(f"{label}: unknown Source 1 entity {s1} (line {line_no})")
        ids = [p.strip() for p in raw.split(",")] if raw else []
        ids = [p for p in ids if p]
        if len(ids) != len(set(ids)):
            issues.append(f"{label}: duplicate entity ids in list for {s1} (line {line_no})")
        for value in ids:
            if not (value.startswith("S2-") or value.startswith("S3-")):
                issues.append(f"{label}: {value} for {s1} is not an S2-/S3- id")
            elif value not in valid_s23:
                issues.append(f"{label}: {value} for {s1} does not exist in the test set")
        parsed[s1] = ids

    missing = test_s1 - set(parsed.keys())
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        issues.append(
            f"{label}: {len(missing)} Source 1 entities missing (e.g. {sample})"
        )
    return parsed


def validate(matching_path: str, candidate_path: str, test_dir: str) -> List[str]:
    issues: List[str] = []
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")
    for path in (test_s1_path, test_s2_path, test_s3_path):
        if not os.path.exists(path):
            issues.append(f"test file not found: {path}")
    if issues:
        return issues

    test_s1 = _read_ids(test_s1_path)
    valid_s23 = _read_ids(test_s2_path) | _read_ids(test_s3_path)

    matching = _validate_file(
        matching_path,
        ["source1_entity_id", "matched_entity_ids"],
        test_s1,
        valid_s23,
        "matching_results",
        issues,
    )
    candidate = _validate_file(
        candidate_path,
        ["source1_entity_id", "candidate_entity_ids"],
        test_s1,
        valid_s23,
        "candidate_pairs",
        issues,
    )

    for s1, ids in matching.items():
        cands = set(candidate.get(s1, []))
        extra = [value for value in ids if value not in cands]
        if extra:
            issues.append(
                f"matching_results: matched ids not in candidate_pairs for {s1}: {extra[:5]}"
            )
    return issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate ER submission files.")
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args(argv)

    issues = validate(args.matching, args.candidate, args.test_dir)
    if issues:
        print(f"FAIL: {len(issues)} issue(s) found")
        for index, issue in enumerate(issues, start=1):
            print(f"{index}. {issue}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
