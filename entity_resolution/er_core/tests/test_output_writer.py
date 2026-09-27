import os

import pandas as pd

from src.config.config import load_config
from src.postprocess.output_writer import write_outputs


def _make_test_dir(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAlpha\tRoad 1\tIN\n"
        "S1-2\tBeta\tRoad 2\tIN\n",
        encoding="utf-8",
    )
    (test_dir / "test_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tAlpha\tRoad 1\tIN\n"
        "S2-2\tBeta\tRoad 2\tIN\n",
        encoding="utf-8",
    )
    (test_dir / "test_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-1\tAlpha\tRoad 1\tIN\n",
        encoding="utf-8",
    )
    return str(test_dir)


def test_write_outputs_valid(tmp_path):
    test_dir = _make_test_dir(tmp_path)
    cfg = load_config()
    cfg["paths"]["test_dir"] = test_dir
    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "s23_id": ["S2-1", "S3-1", "S2-2"],
            "layer_mask": [1, 1, 1],
            "cheap_score": [0.9, 0.8, 0.7],
        }
    )
    decisions = {"S1-1": ["S2-1"], "S1-2": []}
    valid = {"S2-1", "S2-2", "S3-1"}
    result = write_outputs(
        ["S1-1", "S1-2"], decisions, pairs, valid, str(tmp_path / "out"), cfg
    )
    assert result["validation_passed"] is True
    assert result["errors"] == []
    assert result["n_rows"] == 2


def test_matches_must_be_subset_of_candidates(tmp_path):
    test_dir = _make_test_dir(tmp_path)
    cfg = load_config()
    cfg["paths"]["test_dir"] = test_dir
    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1"],
            "s23_id": ["S2-1"],
            "layer_mask": [1],
            "cheap_score": [0.9],
        }
    )
    # decision references S3-1 which is not a candidate for S1-1
    decisions = {"S1-1": ["S3-1"]}
    result = write_outputs(
        ["S1-1", "S1-2"], decisions, pairs, {"S2-1", "S2-2", "S3-1"}, str(tmp_path / "out"), cfg
    )
    assert result["validation_passed"] is True
    # S3-1 silently dropped because it is not a candidate; outputs stay valid
    assert result["n_matches_total"] == 0
