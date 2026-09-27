"""Business Entity Resolution pipeline (Amazon ML Challenge 2026).

Stage map:
    1 normalize.canonicalize  -> canonical records
    2 blocking.cascade        -> candidate pairs (candidate_pairs.tsv)
    3 features.build          -> pairwise feature matrix
    4 models.*                -> calibrated pair model + singleton model
    5 decisions.expected_f05  -> per-entity expected-F0.5 decision
    6 postprocess + output    -> matching_results.tsv / candidate_pairs.tsv
"""

__version__ = "1.0.0"
