"""Name-channel pairwise features."""
from __future__ import annotations

from typing import Dict

from rapidfuzz.distance import JaroWinkler, LCSseq, Levenshtein, OSA

from ..features.index import Record

NAME_FEATURES = [
    "jaccard_tok",
    "jaccard_char3",
    "lev_norm",
    "osa_norm",
    "tfidf_cos_char3",
    "tokensort_containment",
    "lcs_ratio",
    "jaro_winkler",
    "dba_max_frag_sim",
    "token_align_frac",
    "first_token_match",
    "suffix_agreement",
    "token_count_diff",
    "idf_weighted_overlap_sum",
    "name_len_diff",
    "exact_canon",
    "exact_nosuffix",
    "sorted_tok_exact",
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


def _align_frac(a_tokens, b_tokens) -> float:
    if not a_tokens:
        return 0.0
    b_list = list(b_tokens)
    matched = 0
    for token in a_tokens:
        for other in b_list:
            if abs(len(token) - len(other)) > 2:
                continue
            if Levenshtein.distance(token, other) <= 2:
                matched += 1
                break
    return matched / len(a_tokens)


def _dba_sim(a: Record, b: Record) -> float:
    base = _jaccard(a.name_tokens, b.name_tokens)
    options = [base]
    a_frags = [f for f in (a.frag1_tokens, a.frag2_tokens) if f]
    b_frags = [f for f in (b.frag1_tokens, b.frag2_tokens) if f]
    for frag in a_frags:
        options.append(_jaccard(frag, b.name_tokens))
    for frag in b_frags:
        options.append(_jaccard(a.name_tokens, frag))
    return max(options) if options else 0.0


def compute(a: Record, b: Record, tfidf_cos: float, idf: Dict[str, float]) -> Dict[str, float]:
    len_a = len(a.name_canon)
    len_b = len(b.name_canon)
    denom_len = max(1, max(len_a, len_b))
    shared = a.name_tokens & b.name_tokens
    idf_overlap = sum(idf.get(token, 0.0) for token in shared)
    suffix_agreement = 1.0 if a.had_legal_suffix == b.had_legal_suffix else 0.5
    return {
        "jaccard_tok": _jaccard(a.name_tokens, b.name_tokens),
        "jaccard_char3": _jaccard(a.name_char3, b.name_char3),
        "lev_norm": Levenshtein.normalized_similarity(a.name_canon, b.name_canon),
        "osa_norm": OSA.normalized_similarity(a.name_canon, b.name_canon),
        "tfidf_cos_char3": float(tfidf_cos),
        "tokensort_containment": _containment(a.nosuffix_tokens, b.nosuffix_tokens),
        "lcs_ratio": LCSseq.normalized_similarity(a.name_canon, b.name_canon),
        "jaro_winkler": JaroWinkler.similarity(a.name_canon, b.name_canon),
        "dba_max_frag_sim": _dba_sim(a, b),
        "token_align_frac": _align_frac(a.name_tokens, b.name_tokens),
        "first_token_match": float(
            bool(a.first_name_token) and a.first_name_token == b.first_name_token
        ),
        "suffix_agreement": suffix_agreement,
        "token_count_diff": float(abs(len(a.name_tokens) - len(b.name_tokens))),
        "idf_weighted_overlap_sum": float(idf_overlap),
        "name_len_diff": float(abs(len_a - len_b)) / denom_len,
        "exact_canon": float(a.name_canon == b.name_canon and bool(a.name_canon)),
        "exact_nosuffix": float(
            a.name_nosuffix == b.name_nosuffix and bool(a.name_nosuffix)
        ),
        "sorted_tok_exact": float(a.name_sortedtok == b.name_sortedtok and bool(a.name_sortedtok)),
    }
