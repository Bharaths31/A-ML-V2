import pandas as pd

from src.blocking.cascade import build_candidate_pairs
from src.blocking.measure import blocking_report
from src.blocking.mass_control import apply_mass_control
from src.config.config import load_config
from src.normalize.canonicalize import canonicalize_frame, load_assets


def _frame(rows):
    df = pd.DataFrame(rows)
    df["source"] = df["entity_id"].str.slice(0, 2)
    return canonicalize_frame(df, load_assets())


def test_exact_layers_catch_variants():
    cfg = load_config()
    records = _frame(
        [
            {"entity_id": "S1-1", "business_name": "Alpha Traders Pvt Ltd", "business_address": "12 Main St, Pune 411001", "country": "IN"},
            {"entity_id": "S2-1", "business_name": "Alpha Traders", "business_address": "12 Main St, Pune 411001", "country": "IN"},
            {"entity_id": "S3-1", "business_name": "Traders Alpha Limited", "business_address": "12 Main St Pune 411001", "country": "IN"},
            {"entity_id": "S2-2", "business_name": "Beta Corp", "business_address": "99 Other Rd, Pune 411002", "country": "IN"},
        ]
    )
    s1 = records[records["source"] == "S1"]
    s23 = records[records["source"].isin(["S2", "S3"])]
    pairs = build_candidate_pairs(s1, s23, records, cfg)
    found = set(zip(pairs["s1_id"], pairs["s23_id"]))
    assert ("S1-1", "S2-1") in found
    assert ("S1-1", "S3-1") in found


def test_mass_control_caps():
    cfg = load_config()
    cfg["blocking"]["mass"]["per_entity_cap"] = 2
    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1"] * 5,
            "s23_id": [f"S2-{i}" for i in range(5)],
            "layer_mask": [32] * 5,
            "cheap_score": [0.1, 0.9, 0.5, 0.7, 0.3],
        }
    )
    out = apply_mass_control(pairs, cfg)
    assert len(out) == 2
    assert set(out["s23_id"]) == {"S2-1", "S2-3"}


def test_minhash_layers_catch_near_variants():
    from src.blocking.layers_lsh import minhash_layer_pairs

    cfg = load_config()
    records = _frame(
        [
            {"entity_id": "S1-1", "business_name": "Alpha Traders", "business_address": "12 Main Street, Pune 411001", "country": "IN"},
            {"entity_id": "S2-1", "business_name": "Alpha Trader", "business_address": "12 Main St, Pune 411001", "country": "IN"},
        ]
    )
    s1 = records[records["source"] == "S1"]
    s23 = records[records["source"].isin(["S2", "S3"])]
    frames = minhash_layer_pairs(s1, s23, {"B1": True, "B2": True}, cfg, 42)
    found = set()
    for frame in frames:
        if not frame.empty:
            found |= set(zip(frame["s1_id"], frame["s23_id"]))
    assert ("S1-1", "S2-1") in found


def test_blocking_report_recall():
    cfg = load_config()
    records = _frame(
        [
            {"entity_id": "S1-1", "business_name": "Alpha Traders", "business_address": "12 Main St, Pune 411001", "country": "IN"},
            {"entity_id": "S2-1", "business_name": "Alpha Traders", "business_address": "12 Main St, Pune 411001", "country": "IN"},
            {"entity_id": "S3-9", "business_name": "Gamma Foods", "business_address": "1 Lake Rd, Surat 395001", "country": "IN"},
        ]
    )
    s1 = records[records["source"] == "S1"]
    s23 = records[records["source"].isin(["S2", "S3"])]
    pairs = build_candidate_pairs(s1, s23, records, cfg)
    truth = {"S1-1": ["S2-1"]}
    report = blocking_report(pairs, truth, ["S1-1"], len(s23))
    assert report["recall_ceiling"] == 1.0
    assert report["mean_candidates_per_s1"] >= 1
