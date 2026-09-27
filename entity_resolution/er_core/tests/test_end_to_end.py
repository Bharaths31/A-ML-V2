import os

import pytest

from src.config.config import load_config
from src.pipeline import run_pipeline
from src.utils.synthetic import generate_dataset


@pytest.mark.slow
def test_end_to_end(tmp_path):
    data_root = str(tmp_path / "dataset")
    generate_dataset(
        data_root,
        n_train_businesses=160,
        n_test_businesses=90,
        seed=7,
        with_test_gt=True,
    )
    cfg = load_config()
    cfg["paths"]["train_dir"] = os.path.join(data_root, "train")
    cfg["paths"]["test_dir"] = os.path.join(data_root, "test")
    cfg["paths"]["out_dir"] = str(tmp_path / "output")
    cfg["paths"]["cache_dir"] = str(tmp_path / "artifacts")
    cfg["model"]["lgbm"]["n_estimators"] = 120

    result = run_pipeline(cfg)

    assert result["submission"]["validation_passed"] is True
    assert result["submission"]["errors"] == []
    assert result["blocking_val"]["recall_ceiling"] >= 0.9
    assert result["validation"]["macro_f05"] > 0.3
    assert result["test_macro_f05_hidden"] is not None
    assert os.path.exists(os.path.join(cfg["paths"]["out_dir"], "matching_results.tsv"))
    assert os.path.exists(os.path.join(cfg["paths"]["out_dir"], "candidate_pairs.tsv"))
