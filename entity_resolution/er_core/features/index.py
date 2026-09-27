"""Precomputed per-record structures and global token IDF."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

import pandas as pd


def char3(text: str) -> FrozenSet[str]:
    text = str(text or "")
    if len(text) < 3:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))


def tokens(text: str) -> FrozenSet[str]:
    return frozenset(t for t in str(text or "").split() if t)


@dataclass
class Record:
    entity_id: str
    source: str
    country: str
    name_canon: str
    name_nosuffix: str
    name_sortedtok: str
    name_frag1: str
    name_frag2: str
    first_name_token: str
    had_legal_suffix: int
    addr_canon: str
    addr_street_tokens: str
    addr_house_no: str
    addr_pin: str
    addr_city: str
    addr_locality: str
    addr_landmark: str
    has_house: int
    has_street: int
    has_city: int
    has_pin: int
    has_landmark: int
    has_locality: int
    name_tokens: FrozenSet[str]
    nosuffix_tokens: FrozenSet[str]
    name_char3: FrozenSet[str]
    addr_tokens: FrozenSet[str]
    addr_char3: FrozenSet[str]
    landmark_tokens: FrozenSet[str]
    frag1_tokens: FrozenSet[str]
    frag2_tokens: FrozenSet[str]

    @property
    def presence_flags(self) -> Tuple[int, int, int, int, int, int]:
        return (
            self.has_house,
            self.has_street,
            self.has_city,
            self.has_pin,
            self.has_landmark,
            self.has_locality,
        )


def build_index(canon_df: pd.DataFrame) -> Tuple[Dict[str, Record], Dict[str, float]]:
    index: Dict[str, Record] = {}
    df_count: Dict[str, int] = {}
    rows = []
    for _, row in canon_df.iterrows():
        record = Record(
            entity_id=str(row["entity_id"]),
            source=str(row["source"]),
            country=str(row["country"]),
            name_canon=str(row["name_canon"]),
            name_nosuffix=str(row["name_nosuffix"]),
            name_sortedtok=str(row["name_sortedtok"]),
            name_frag1=str(row.get("name_frag1", "")),
            name_frag2=str(row.get("name_frag2", "")),
            first_name_token=str(row.get("first_name_token", "")),
            had_legal_suffix=int(row.get("had_legal_suffix", 0)),
            addr_canon=str(row["addr_canon"]),
            addr_street_tokens=str(row["addr_street_tokens"]),
            addr_house_no=str(row.get("addr_house_no", "")),
            addr_pin=str(row.get("addr_pin", "")),
            addr_city=str(row.get("addr_city", "")),
            addr_locality=str(row.get("addr_locality", "")),
            addr_landmark=str(row.get("addr_landmark", "")),
            has_house=int(row.get("has_house", 0)),
            has_street=int(row.get("has_street", 0)),
            has_city=int(row.get("has_city", 0)),
            has_pin=int(row.get("has_pin", 0)),
            has_landmark=int(row.get("has_landmark", 0)),
            has_locality=int(row.get("has_locality", 0)),
            name_tokens=tokens(row["name_canon"]),
            nosuffix_tokens=tokens(row["name_nosuffix"]),
            name_char3=char3(row["name_canon"]),
            addr_tokens=tokens(row["addr_street_tokens"]),
            addr_char3=char3(row["addr_canon"]),
            landmark_tokens=tokens(row.get("addr_landmark", "")),
            frag1_tokens=tokens(row.get("name_frag1", "")),
            frag2_tokens=tokens(row.get("name_frag2", "")),
        )
        index[record.entity_id] = record
        rows.append(record)
        for token in record.name_tokens:
            df_count[token] = df_count.get(token, 0) + 1

    n_docs = max(1, len(rows))
    idf = {
        token: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
        for token, count in df_count.items()
    }
    return index, idf
