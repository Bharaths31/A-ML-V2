"""Stage 1 - Canonicalization.

Country-agnostic by construction: assets are unions across conventions and the
country field is never used as a filter. Produces the canonical record columns
consumed by blocking (stage 2) and features (stage 3).
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Set, Tuple

import pandas as pd

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_DBA_PATTERNS = [
    r"\bd/?b/?a\b",
    r"\btrading as\b",
    r"\bt/a\b",
    r"\ba unit of\b",
]
_DIGITS_RE = re.compile(r"\d")
_HOUSE_RE = re.compile(r"^(\d+[a-z]?)$")
_PIN_RE = re.compile(r"^(\d{4,6})$")
_LANDMARK_CAPTURE = re.compile(
    r"(?i)\b(?:near|opposite|opp|behind|beside|adjacent to)\b[\s:,-]*([^,;]+)"
)
_WS_RE = re.compile(r"\s+")


def load_assets(assets_dir: str = ASSETS_DIR) -> Dict[str, object]:
    suffix_map: Dict[str, str] = {}
    with open(os.path.join(assets_dir, "legal_suffixes.txt"), encoding="utf-8") as handle:
        for line in handle:
            tokens = [t.strip() for t in line.strip().split(",") if t.strip()]
            if not tokens:
                continue
            canonical = tokens[0]
            for token in tokens:
                suffix_map[token] = canonical

    street_map: Dict[str, str] = {}
    with open(os.path.join(assets_dir, "street_types.txt"), encoding="utf-8") as handle:
        for line in handle:
            tokens = [t.strip() for t in line.strip().split(",") if t.strip()]
            if not tokens:
                continue
            canonical = tokens[0]
            for token in tokens:
                street_map[token] = canonical

    with open(os.path.join(assets_dir, "stopwords.txt"), encoding="utf-8") as handle:
        stopwords = {line.strip() for line in handle if line.strip()}

    return {"suffix_map": suffix_map, "street_map": street_map, "stopwords": stopwords}


def normalize_text(text: str, ascii_fold: bool = True, strip_punct: bool = True) -> str:
    text = str(text or "")
    text = text.replace("&", " and ")
    if ascii_fold:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    if strip_punct:
        text = re.sub(r"[^a-z0-9]+", " ", text)
    else:
        text = re.sub(r"[^a-z0-9&]+", " ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    text = text.strip()
    return text.split(" ") if text else []


def _strip_trailing_suffix(
    tokens: List[str], suffix_map: Dict[str, str]
) -> Tuple[List[str], bool, List[str]]:
    """Strip up to 3 trailing legal-suffix tokens.

    Returns (remaining_tokens, had_suffix, canonical_suffix_tokens).
    """
    remaining = list(tokens)
    had = False
    removed: List[str] = []
    while remaining and len(removed) < 3 and remaining[-1] in suffix_map:
        removed.append(suffix_map[remaining[-1]])
        remaining.pop()
        had = True
    return remaining, had, removed


def _split_dba(raw_lower: str) -> List[str]:
    text = raw_lower
    for pattern in _DBA_PATTERNS:
        match = re.search(pattern, text)
        if match:
            left = text[: match.start()]
            right = text[match.end() :]
            if left.strip() and right.strip():
                return [left.strip(), right.strip()]
    return []


def canonicalize_name(name: str, assets: Dict[str, object]) -> Dict[str, object]:
    suffix_map: Dict[str, str] = assets["suffix_map"]  # type: ignore[assignment]
    raw = str(name or "")
    canon = normalize_text(raw)
    tokens = tokenize(canon)
    kept, had_suffix, _ = _strip_trailing_suffix(tokens, suffix_map)
    nosuffix = " ".join(kept)
    sorted_tok = " ".join(sorted(kept))
    frags = [normalize_text(f) for f in _split_dba(raw.lower())]
    first_token = kept[0] if kept else (tokens[0] if tokens else "")
    return {
        "name_raw": raw,
        "name_canon": canon,
        "name_nosuffix": nosuffix,
        "name_sortedtok": sorted_tok,
        "name_tokens": canon,
        "name_frag1": frags[0] if frags else "",
        "name_frag2": frags[1] if len(frags) > 1 else "",
        "first_name_token": first_token,
        "had_legal_suffix": int(had_suffix),
        "n_name_tokens": len(tokens),
    }


def canonicalize_address(addr: str, assets: Dict[str, object]) -> Dict[str, object]:
    street_map: Dict[str, str] = assets["street_map"]  # type: ignore[assignment]
    stopwords: Set[str] = assets["stopwords"]  # type: ignore[assignment]
    raw = str(addr or "")
    landmark_match = _LANDMARK_CAPTURE.search(raw)
    landmark = ""
    working = raw
    if landmark_match:
        # Landmark phrase is the segment after the keyword, up to the next comma.
        landmark = normalize_text(landmark_match.group(1))
        working = (raw[: landmark_match.start()] + " " + raw[landmark_match.end() :]).strip()
    canon = normalize_text(working)
    tokens = tokenize(canon)

    house_no = ""
    pin = ""
    for token in tokens:
        if not house_no and _HOUSE_RE.match(token):
            house_no = token
        if _PIN_RE.match(token):
            pin = token

    street_tokens: List[str] = []
    for token in tokens:
        if token in stopwords:
            continue
        if _DIGITS_RE.search(token):
            continue
        if token in street_map:
            token = street_map[token]
        street_tokens.append(token)

    # city = last street token (postal convention: city closest to end), locality = rest
    city = street_tokens[-1] if street_tokens else ""
    locality = " ".join(street_tokens[:-1]) if len(street_tokens) > 1 else ""

    return {
        "addr_raw": raw,
        "addr_canon": canon,
        "addr_street_tokens": " ".join(street_tokens),
        "addr_house_no": house_no,
        "addr_pin": pin,
        "addr_city": city,
        "addr_locality": locality,
        "addr_landmark": landmark,
        "has_house": int(bool(house_no)),
        "has_street": int(bool(street_tokens)),
        "has_city": int(bool(city)),
        "has_pin": int(bool(pin)),
        "has_landmark": int(bool(landmark)),
        "has_locality": int(bool(locality)),
        "n_addr_tokens": len(tokens),
    }


def canonicalize_frame(df: pd.DataFrame, assets: Dict[str, object]) -> pd.DataFrame:
    """Canonicalize a source DataFrame; returns one row per input record."""
    name_rows = [canonicalize_name(v, assets) for v in df["business_name"]]
    addr_rows = [canonicalize_address(v, assets) for v in df["business_address"]]
    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].values,
            "source": df["source"].values,
            "country": df["country"].astype(str).values,
        }
    )
    out = pd.concat([out, pd.DataFrame(name_rows), pd.DataFrame(addr_rows)], axis=1)
    return out.reset_index(drop=True)


def canonicalize_sources(records: pd.DataFrame, assets: Dict[str, object]) -> pd.DataFrame:
    return canonicalize_frame(records, assets)
