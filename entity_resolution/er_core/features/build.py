"""Assemble the pairwise feature matrix from a candidate-pair frame."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..features.address_feats import ADDRESS_FEATURES, compute as addr_compute
from ..features.cross_feats import CROSS_FEATURES, compute as cross_compute
from ..features.global_feats import GLOBAL_FEATURES, compute as global_compute
from ..features.index import build_index
from ..features.name_feats import NAME_FEATURES, compute as name_compute

FEATURE_COLUMNS: List[str] = (
    NAME_FEATURES + ADDRESS_FEATURES + CROSS_FEATURES + GLOBAL_FEATURES + ["cheap_score"]
)

ID_COLUMNS = ["s1_id", "s23_id"]


def _pair_cosine(pairs: pd.DataFrame, all_canon: pd.DataFrame, field: str) -> np.ndarray:
    if pairs.empty:
        return np.zeros(0, dtype=float)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True)
    matrix = vectorizer.fit_transform(all_canon[field].astype(str).tolist())
    id_to_row = {eid: i for i, eid in enumerate(all_canon["entity_id"].tolist())}
    rows_a = np.array([id_to_row.get(x, -1) for x in pairs["s1_id"]], dtype=int)
    rows_b = np.array([id_to_row.get(x, -1) for x in pairs["s23_id"]], dtype=int)
    result = np.zeros(len(pairs), dtype=float)
    valid = (rows_a >= 0) & (rows_b >= 0)
    if valid.any():
        xs = matrix[rows_a[valid]]
        ys = matrix[rows_b[valid]]
        result[valid] = np.asarray(xs.multiply(ys).sum(axis=1)).ravel()
    return result


def assemble_features(
    pairs: pd.DataFrame, canon_df: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """Return a feature matrix (s1_id, s23_id + FEATURE_COLUMNS)."""
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=ID_COLUMNS + FEATURE_COLUMNS)

    index, idf = build_index(canon_df)
    frame = pairs.copy().reset_index(drop=True)
    if "layer_mask" not in frame.columns:
        frame["layer_mask"] = 0
    if "cheap_score" not in frame.columns:
        frame["cheap_score"] = 0.0

    name_cos = _pair_cosine(frame, canon_df, "name_canon")
    addr_cos = _pair_cosine(frame, canon_df, "addr_canon")

    name_rows = []
    addr_rows = []
    cross_rows = []
    valid_mask = []
    for position, (s1_id, s23_id) in enumerate(zip(frame["s1_id"], frame["s23_id"])):
        a = index.get(s1_id)
        b = index.get(s23_id)
        if a is None or b is None:
            valid_mask.append(False)
            name_rows.append({name: 0.0 for name in NAME_FEATURES})
            addr_rows.append({name: 0.0 for name in ADDRESS_FEATURES})
            cross_rows.append({"same_country": 0.0, "source_flag": 0.0})
            continue
        valid_mask.append(True)
        name_rows.append(name_compute(a, b, name_cos[position], idf))
        addr_rows.append(addr_compute(a, b, addr_cos[position]))
        cross_rows.append(cross_compute(a, b))

    features = pd.concat(
        [
            frame[ID_COLUMNS + ["layer_mask", "cheap_score"]].reset_index(drop=True),
            pd.DataFrame(name_rows),
            pd.DataFrame(addr_rows),
            pd.DataFrame(cross_rows),
        ],
        axis=1,
    )
    features = global_compute(features, idf, index)
    for column in FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return features
