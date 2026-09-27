"""End-to-end pipeline orchestration (stages 1-6)."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .blocking.cascade import build_candidate_pairs
from .blocking.measure import blocking_report
from .config import config as cfgmod
from .evaluate import reports
from .evaluate.hard_pairs import build_zoo, zoo_score
from .evaluate.scorer import macro_f05, score_detailed
from .evaluate.splits import assign_entity_splits, split_summary
from .features.build import assemble_features
from .models.calibration import (
    apply_calibrator,
    fit_calibrator,
    reliability_report,
    save_calibrator,
)
from .models.pair_model import (
    feature_importance,
    predict_pair_proba,
    save_model,
    train_pair_model,
)
from .models.singleton_model import (
    build_entity_features,
    predict_singleton,
    train_singleton_model,
)
from .models.singleton_model import save_model as save_singleton
from .normalize.canonicalize import canonicalize_frame, load_assets
from .postprocess.consistency import apply_decisions, pool_near_duplicates
from .postprocess.output_writer import write_outputs
from .utils import io


def load_split(data_dir: str, prefix: str) -> Tuple[pd.DataFrame, Optional[Dict[str, List[str]]]]:
    parts = [
        io.read_source_tsv(os.path.join(data_dir, f"{prefix}_source{i}.tsv"))
        for i in (1, 2, 3)
    ]
    records = pd.concat(parts, ignore_index=True)
    gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
    truth = io.read_ground_truth(gt_path) if os.path.exists(gt_path) else None
    return records, truth


def _label_pairs(features: pd.DataFrame, truth: Dict[str, List[str]]) -> pd.DataFrame:
    truth_sets = {key: set(value) for key, value in truth.items()}
    frame = features.copy()
    frame["label"] = [
        1 if s23 in truth_sets.get(s1, ()) else 0
        for s1, s23 in zip(frame["s1_id"], frame["s23_id"])
    ]
    return frame


def _block_and_featurize(s1_df, s23_pool, all_canon, cfg):
    pairs = build_candidate_pairs(s1_df, s23_pool, all_canon, cfg)
    features = assemble_features(pairs, all_canon, cfg)
    return pairs, features


def _select_s1(canon_df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    return canon_df[canon_df["entity_id"].isin(set(ids))].reset_index(drop=True)


def predict_split(
    features: pd.DataFrame,
    s1_ids: List[str],
    model,
    calibrator,
    singleton_model,
    canon_df: pd.DataFrame,
    cfg: dict,
):
    """Predict, calibrate, pool near-duplicates, fit q, and run the decision engine."""
    if features is None or features.empty:
        pair_frame = pd.DataFrame(columns=["s1_id", "s23_id", "p"])
        pooled = pair_frame.assign(p_pooled=[])
        entity_features = build_entity_features(np.zeros(0), pd.DataFrame(columns=["s1_id", "cheap_score", "layer_mask", "name_idf_uniqueness", "same_country", "component_presence_agreement"]), list(s1_ids))
        q = predict_singleton(singleton_model, entity_features)
        q_by = dict(zip(entity_features["s1_id"], q))
        return {}, q_by, entity_features, pooled, np.zeros(0)

    raw = predict_pair_proba(model, features)
    calibrated = apply_calibrator(calibrator, raw)
    pair_frame = features[["s1_id", "s23_id"]].copy()
    pair_frame["p"] = calibrated
    pooled = pool_near_duplicates(pair_frame, canon_df, cfg)
    entity_features = build_entity_features(
        pooled["p_pooled"].to_numpy(dtype=float), features, list(s1_ids)
    )
    q = predict_singleton(singleton_model, entity_features)
    q_by = dict(zip(entity_features["s1_id"], q))
    decisions = apply_decisions(pooled, q_by, cfg, prob_column="p_pooled")
    return decisions, q_by, entity_features, pooled, calibrated


def run_pipeline(cfg: dict) -> Dict[str, object]:
    started = time.time()
    seed = int(cfg.get("seed", 42))
    train_dir = cfg["paths"]["train_dir"]
    test_dir = cfg["paths"]["test_dir"]
    cache = cfgmod.cache_dir(cfg)
    out = cfgmod.out_dir(cfg)
    reports_dir = os.path.join(cache, "reports")

    # ---- Stage 0: load + EDA ------------------------------------------------
    train_records, truth = load_split(train_dir, "train")
    if truth is None:
        raise ValueError("training ground truth is required (train_ground_truth.tsv)")
    test_records, test_truth = load_split(test_dir, "test")

    assets = load_assets()
    canon_train = canonicalize_frame(train_records, assets)
    canon_test = canonicalize_frame(test_records, assets)
    s23_train = canon_train[canon_train["source"].isin(["S2", "S3"])].reset_index(drop=True)
    s23_test = canon_test[canon_test["source"].isin(["S2", "S3"])].reset_index(drop=True)

    eda_train = reports.eda_report(train_records, truth, "train")
    reports.save_report(eda_train, os.path.join(reports_dir, "eda_train.json"))
    reports.save_report(reports.eda_report(test_records, None, "test"),
                        os.path.join(reports_dir, "eda_test.json"))

    # ---- Stage 0b: entity-level splits -------------------------------------
    train_s1_ids = canon_train.loc[canon_train["source"] == "S1", "entity_id"].tolist()
    country_by_id = dict(zip(canon_train["entity_id"], canon_train["country"]))
    split = assign_entity_splits(
        train_s1_ids,
        truth,
        country_by_id,
        test_frac=float(cfg["validation"]["test_frac"]),
        calib_frac=float(cfg["validation"]["calib_frac"]),
        seed=seed,
    )
    fit_ids = [e for e in train_s1_ids if split.get(e) == "fit"]
    calib_ids = [e for e in train_s1_ids if split.get(e) == "calib"]
    val_ids = [e for e in train_s1_ids if split.get(e) == "val"]
    reports.save_report(
        split_summary(split, truth).to_dict(orient="records"),
        os.path.join(reports_dir, "split_summary.json"),
    )

    # ---- Stages 1-3: block + featurize each split --------------------------
    pairs_fit, feats_fit = _block_and_featurize(_select_s1(canon_train, fit_ids), s23_train, canon_train, cfg)
    pairs_calib, feats_calib = _block_and_featurize(_select_s1(canon_train, calib_ids), s23_train, canon_train, cfg)
    pairs_val, feats_val = _block_and_featurize(_select_s1(canon_train, val_ids), s23_train, canon_train, cfg)
    feats_fit = _label_pairs(feats_fit, truth)
    feats_calib = _label_pairs(feats_calib, truth)
    feats_val = _label_pairs(feats_val, truth)

    blocking_val = blocking_report(pairs_val, truth, val_ids, len(s23_train))
    reports.save_report(blocking_val, os.path.join(reports_dir, "blocking_val.json"))

    # Union (pre-mass-control) recall ceiling: the raw blocker's coverage.
    pairs_val_union = build_candidate_pairs(
        _select_s1(canon_train, val_ids), s23_train, canon_train, cfg, apply_mass=False
    )
    blocking_val_union = blocking_report(pairs_val_union, truth, val_ids, len(s23_train))
    reports.save_report(
        blocking_val_union, os.path.join(reports_dir, "blocking_val_union.json")
    )

    # ---- Stage 4: train + calibrate ----------------------------------------
    model = train_pair_model(feats_fit, cfg)
    raw_calib = predict_pair_proba(model, feats_calib)
    calibrator = fit_calibrator(
        raw_calib, feats_calib["label"].to_numpy(dtype=int), cfg["calibration"]["method"]
    )
    p_calib = apply_calibrator(calibrator, raw_calib)
    reliability = reliability_report(p_calib, feats_calib["label"].to_numpy(dtype=int))
    reports.save_report(reliability, os.path.join(reports_dir, "calibration.json"))

    entity_calib = build_entity_features(p_calib, feats_calib, calib_ids)
    singleton_labels = np.array([1 if len(truth.get(e, [])) == 0 else 0 for e in calib_ids])
    singleton_model = train_singleton_model(entity_calib, singleton_labels, cfg)

    save_model(model, os.path.join(cache, "pair_model.pkl"))
    save_calibrator(calibrator, os.path.join(cache, "calibrator.pkl"))
    save_singleton(singleton_model, os.path.join(cache, "singleton_model.pkl"))

    # ---- Stage 5: decision on validation -----------------------------------
    dec_val, q_val, _, pooled_val, _ = predict_split(
        feats_val, val_ids, model, calibrator, singleton_model, canon_train, cfg
    )
    detail_val = score_detailed(dec_val, truth, val_ids)
    reports.save_report(detail_val, os.path.join(reports_dir, "validation_scores.json"))
    diag_val = reports.decision_diagnostics(dec_val, truth, val_ids)
    reports.save_report(diag_val, os.path.join(reports_dir, "decision_diagnostics.json"))

    zoo = build_zoo(feats_val, truth)
    zoo_f = zoo_score(dec_val, truth, zoo["entities"]) if zoo["entities"] else 0.0
    reports.save_report({**zoo, "zoo_f05": zoo_f}, os.path.join(reports_dir, "hard_pairs.json"))

    # ---- Stage 6: inference + outputs on test ------------------------------
    test_s1 = canon_test[canon_test["source"] == "S1"].reset_index(drop=True)
    test_s1_ids = test_s1["entity_id"].tolist()
    pairs_test, feats_test = _block_and_featurize(test_s1, s23_test, canon_test, cfg)
    dec_test, q_test, _, pooled_test, _ = predict_split(
        feats_test, test_s1_ids, model, calibrator, singleton_model, canon_test, cfg
    )
    valid_s23 = set(s23_test["entity_id"])
    write_result = write_outputs(test_s1_ids, dec_test, pairs_test, valid_s23, out, cfg)
    reports.save_report(
        write_result, os.path.join(reports_dir, "submission_check.json")
    )

    blocking_test = blocking_report(pairs_test, {}, test_s1_ids, len(s23_test))
    reports.save_report(blocking_test, os.path.join(reports_dir, "blocking_test.json"))

    # ---- diagnostics + artifact persistence --------------------------------
    test_macro = (
        macro_f05(dec_test, test_truth, test_s1_ids) if test_truth is not None else None
    )
    runtime = time.time() - started
    experiment_row = {
        "seed": seed,
        "val_macro_f05": round(detail_val["macro_f05"], 4),
        "val_singleton_slice": round(detail_val["slices"].get("singleton", 0.0), 4),
        "zoo_f05": round(zoo_f, 4),
        "blocking_recall_val": round(blocking_val["recall_ceiling"], 4),
        "blocking_union_recall_val": round(blocking_val_union["recall_ceiling"], 4),
        "mean_cands_val": round(blocking_val["mean_candidates_per_s1"], 3),
        "reduction_ratio_val": round(blocking_val["reduction_ratio"], 6),
        "calibration_ece": round(reliability["ece"], 4),
        "test_macro_f05_hidden": round(test_macro, 4) if test_macro is not None else "",
        "runtime_sec": round(runtime, 1),
        "validation_passed": write_result["validation_passed"],
        "top_features": ";".join(
            f"{item['feature']}={item['importance']:.1f}"
            for item in feature_importance(model, top_n=8)
        ),
    }
    reports.append_experiment(os.path.join(cache, "experiments.csv"), experiment_row)

    io.save_table(canon_train, os.path.join(cache, "canon_train"))
    io.save_table(canon_test, os.path.join(cache, "canon_test"))
    io.save_table(pairs_test, os.path.join(cache, "pairs_test"))
    io.save_table(feats_test, os.path.join(cache, "features_test"))
    io.save_table(dec_val if isinstance(dec_val, pd.DataFrame) else pd.DataFrame(
        [{"source1_entity_id": k, "matched": ",".join(v)} for k, v in dec_val.items()]
    ), os.path.join(cache, "predictions_val"))

    return {
        "eda_train": eda_train,
        "blocking_val": blocking_val,
        "blocking_val_union": blocking_val_union,
        "blocking_test": blocking_test,
        "validation": detail_val,
        "calibration": {"ece": reliability["ece"]},
        "decision_diagnostics": diag_val,
        "hard_pairs": {**zoo, "zoo_f05": zoo_f},
        "submission": write_result,
        "experiment": experiment_row,
        "test_macro_f05_hidden": test_macro,
        "runtime_sec": runtime,
    }
