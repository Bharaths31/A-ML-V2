"""Address-channel pairwise features."""
from __future__ import annotations

from typing import Dict

from rapidfuzz.distance import Levenshtein

from ..features.index import Record

ADDRESS_FEATURES = [
    "street_tok_jaccard",
    "street_tok_containment",
    "addr_lev_norm",
    "addr_tfidf_cos_char3",
    "housenum_match",
    "pin_exact",
    "pin_prefix_match",
    "city_match",
    "locality_containment",
    "landmark_tok_overlap",
    "street_type_agreement",
    "component_presence_agreement",
    "addr_len_diff",
    "addr_sorted_tok_exact",
    "addr_missing_any",
    "housenum_present_both",
    "pin_present_both",
]


def _jaccard(a, b) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _containment(a, b) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _ternary_equal(x: str, y: str) -> float:
    if x and y:
        return 1.0 if x == y else 0.0
    return 0.5


def compute(a: Record, b: Record, tfidf_cos: float) -> Dict[str, float]:
    pin_prefix = 0.5
    if a.addr_pin and b.addr_pin:
        pin_prefix = 1.0 if a.addr_pin[:3] == b.addr_pin[:3] else 0.0
    city = 0.5
    if a.addr_city and b.addr_city:
        city = 1.0 if a.addr_city == b.addr_city else 0.0
    presence = sum(1 for x, y in zip(a.presence_flags, b.presence_flags) if x == y) / 6.0
    street_intersection = a.addr_tokens & b.addr_tokens
    height = max(len(a.addr_canon), len(b.addr_canon), 1)
    return {
        "street_tok_jaccard": _jaccard(a.addr_tokens, b.addr_tokens),
        "street_tok_containment": _containment(a.addr_tokens, b.addr_tokens),
        "addr_lev_norm": Levenshtein.normalized_similarity(a.addr_canon, b.addr_canon),
        "addr_tfidf_cos_char3": float(tfidf_cos),
        "housenum_match": _ternary_equal(a.addr_house_no, b.addr_house_no),
        "pin_exact": _ternary_equal(a.addr_pin, b.addr_pin),
        "pin_prefix_match": pin_prefix,
        "city_match": city,
        "locality_containment": _containment(
            set(a.addr_locality.split()), set(b.addr_locality.split())
        ),
        "landmark_tok_overlap": _jaccard(a.landmark_tokens, b.landmark_tokens),
        "street_type_agreement": 1.0 if street_intersection else 0.0,
        "component_presence_agreement": presence,
        "addr_len_diff": float(abs(len(a.addr_canon) - len(b.addr_canon))) / height,
        "addr_sorted_tok_exact": float(
            a.addr_street_tokens == b.addr_street_tokens and bool(a.addr_street_tokens)
        ),
        "addr_missing_any": float(not a.addr_tokens or not b.addr_tokens),
        "housenum_present_both": float(bool(a.addr_house_no) and bool(b.addr_house_no)),
        "pin_present_both": float(bool(a.addr_pin) and bool(b.addr_pin)),
    }
