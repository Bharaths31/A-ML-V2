"""IO utilities: strict TSV reading/writing and artifact persistence."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import pandas as pd

SOURCE_COLS: List[str] = ["entity_id", "business_name", "business_address", "country"]
GT_COLS: List[str] = ["source1_entity_id", "matched_entity_ids"]
MATCH_COLS: List[str] = ["source1_entity_id", "matched_entity_ids"]
CAND_COLS: List[str] = ["source1_entity_id", "candidate_entity_ids"]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_source_tsv(path: str) -> pd.DataFrame:
    """Read a source TSV with an explicit tab separator and no NA coercion."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    df = df[SOURCE_COLS].copy()
    df["source"] = df["entity_id"].str.slice(0, 2)
    for col in SOURCE_COLS:
        df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


def split_id_list(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def read_ground_truth(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    missing = [c for c in GT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    result: Dict[str, List[str]] = {}
    for s1, ids in zip(df[GT_COLS[0]], df[GT_COLS[1]]):
        result[s1.strip()] = split_id_list(ids)
    return result


def write_submission_tsv(df: pd.DataFrame, path: str, value_col: str) -> None:
    """Write a two-column submission TSV: single tabs, comma-joined lists, no quoting."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"source1_entity_id\t{value_col}\n")
        for s1, ids in zip(df["source1_entity_id"], df[value_col]):
            handle.write(f"{s1}\t{ids}\n")


def format_id_list(ids: Sequence[str]) -> str:
    seen: List[str] = []
    for value in ids:
        if value and value not in seen:
            seen.append(value)
    return ",".join(seen)


def save_table(df: pd.DataFrame, path_no_ext: str) -> str:
    """Persist a DataFrame as parquet, falling back to pickle."""
    ensure_dir(os.path.dirname(os.path.abspath(path_no_ext)))
    parquet = path_no_ext + ".parquet"
    try:
        df.to_parquet(parquet, index=False)
        return parquet
    except Exception:  # pragma: no cover - pyarrow is pinned but be safe
        pkl = path_no_ext + ".pkl"
        df.to_pickle(pkl)
        return pkl


def load_table(path_no_ext: str) -> pd.DataFrame:
    for ext in (".parquet", ".pkl"):
        candidate = path_no_ext + ext
        if os.path.exists(candidate):
            if ext == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_pickle(candidate)
    raise FileNotFoundError(f"no artifact found for {path_no_ext} (.parquet/.pkl)")


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
