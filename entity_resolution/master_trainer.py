#!/usr/bin/env python3
"""
Master trainer CLI.

Orchestrates the full training / inference pipeline with profiler-driven
minimal-fit selection. Delegates the heavy lifting to er_core (ported V1 engine).

Stages:
  prepare   : canonicalize train + test, run blocking for selected/calib/val S1
  train     : featurize, train pair model, isotonic calibration, singleton model
  evaluate  : run decision engine on validation, report macro-F0.5
  predict   : inference on test, write matching_results.tsv + candidate_pairs.tsv
  submit    : run official validator + package zip
  all       : run prepare → train → evaluate → predict → submit

Usage:
    python master_trainer.py --stage all --config config.yaml
    python master_trainer.py --stage train --ladder 300k
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from er_core.pipeline import load_split, _label_pairs, _block_and_featurize, _select_s1, predict_split
from er_core.normalize.canonicalize import canonicalize_frame, load_assets
from er_core.evaluate.scorer import macro_f05, score_detailed
from er_core.evaluate.splits import assign_entity_splits
from er_core.models.pair_model import train_pair_model, predict_pair_proba, save_model as save_pair_model, load_model as load_pair_model
from er_core.models.calibration import fit_calibrator, apply_calibrator, save_calibrator, load_calibrator, reliability_report
from er_core.models.singleton_model import train_singleton_model, predict_singleton, save_model as save_singleton_model, load_singleton_model
from er_core.postprocess.output_writer import write_outputs
from er_core.utils.validate_submission import validate


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_selection(profile_dir: str) -> dict[str, Any]:
    path = Path(profile_dir) / "selection.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_splits(data_root: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Canonicalize train and test; return canon_train, canon_test, truth, (optionally test_truth)."""
    train_dir = cfg["paths"]["train_dir"] if os.path.isabs(cfg["paths"]["train_dir"]) else os.path.join(data_root, cfg["paths"]["train_dir"])
    test_dir = cfg["paths"]["test_dir"] if os.path.isabs(cfg["paths"]["test_dir"]) else os.path.join(data_root, cfg["paths"]["test_dir"])

    train_records, truth = load_split(train_dir, "train")
    test_records, test_truth = load_split(test_dir, "test")

    assets = load_assets()
    canon_train = canonicalize_frame(train_records, assets)
    canon_test = canonicalize_frame(test_records, assets)
    return canon_train, canon_test, truth, test_truth


def build_splits_with_selection(
    canon_train: pd.DataFrame,
    truth: dict[str, list[str]],
    cfg: dict[str, Any],
    selection: dict[str, Any] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Return fit_ids, calib_ids, val_ids. If selection is enabled, fit_ids come from selection.json."""
    train_s1_ids = canon_train.loc[canon_train["source"] == "S1", "entity_id"].tolist()
    country_by_id = dict(zip(canon_train["entity_id"], canon_train["country"]))

    if selection and selection.get("enabled"):
        fit_ids = selection["fit_s1_ids"]
        calib_ids = selection["calib_s1_ids"]
        val_ids = selection["val_s1_ids"]
        # defensive: intersect with actual train S1 ids
        s1_set = set(train_s1_ids)
        fit_ids = [e for e in fit_ids if e in s1_set]
        calib_ids = [e for e in calib_ids if e in s1_set]
        val_ids = [e for e in val_ids if e in s1_set]
        # fill any shortfall from remaining S1 ids using V1 splitter
        used = set(fit_ids) | set(calib_ids) | set(val_ids)
        remaining = [e for e in train_s1_ids if e not in used]
        # simple fill: not stratified, but selection should already be representative
        while len(fit_ids) < selection.get("round1_entities", len(fit_ids)) and remaining:
            fit_ids.append(remaining.pop(0))
        return fit_ids, calib_ids, val_ids

    # fall back to V1 splitter
    split = assign_entity_splits(
        train_s1_ids,
        truth,
        country_by_id,
        test_frac=float(cfg["validation"]["test_frac"]),
        calib_frac=float(cfg["validation"]["calib_frac"]),
        seed=int(cfg.get("seed", 42)),
    )
    fit_ids = [e for e in train_s1_ids if split.get(e) == "fit"]
    calib_ids = [e for e in train_s1_ids if split.get(e) == "calib"]
    val_ids = [e for e in train_s1_ids if split.get(e) == "val"]
    return fit_ids, calib_ids, val_ids


def train_stage(
    canon_train: pd.DataFrame,
    truth: dict[str, list[str]],
    fit_ids: list[str],
    calib_ids: list[str],
    val_ids: list[str],
    cfg: dict[str, Any],
    cache: str,
) -> tuple[Any, Any, Any, pd.DataFrame, pd.DataFrame]:
    """Train pair model + calibrator + singleton model; return model, calibrator, singleton_model, feats_val, pairs_val."""
    s23_train = canon_train[canon_train["source"].isin(["S2", "S3"])].reset_index(drop=True)

    print("[trainer] blocking + featurizing fit split...")
    pairs_fit, feats_fit = _block_and_featurize(_select_s1(canon_train, fit_ids), s23_train, canon_train, cfg)
    print(f"[trainer] fit pairs: {len(pairs_fit)}")

    print("[trainer] blocking + featurizing calib split...")
    pairs_calib, feats_calib = _block_and_featurize(_select_s1(canon_train, calib_ids), s23_train, canon_train, cfg)

    print("[trainer] blocking + featurizing val split...")
    pairs_val, feats_val = _block_and_featurize(_select_s1(canon_train, val_ids), s23_train, canon_train, cfg)

    feats_fit = _label_pairs(feats_fit, truth)
    feats_calib = _label_pairs(feats_calib, truth)
    feats_val = _label_pairs(feats_val, truth)

    print("[trainer] training pair model...")
    model = train_pair_model(feats_fit, cfg)

    print("[trainer] calibrating...")
    raw_calib = predict_pair_proba(model, feats_calib)
    calibrator = fit_calibrator(raw_calib, feats_calib["label"].to_numpy(dtype=int), cfg["calibration"]["method"])
    p_calib = apply_calibrator(calibrator, raw_calib)
    reliability = reliability_report(p_calib, feats_calib["label"].to_numpy(dtype=int))
    print(f"[trainer] calibration ECE: {reliability.get('ece', 999):.4f}")

    print("[trainer] training singleton model...")
    # entity features require predictions on calib
    from er_core.models.singleton_model import build_entity_features
    entity_calib = build_entity_features(p_calib, feats_calib, calib_ids)
    singleton_labels = np.array([1 if len(truth.get(e, [])) == 0 else 0 for e in calib_ids])
    singleton_model = train_singleton_model(entity_calib, singleton_labels, cfg)

    # save bundle
    os.makedirs(cache, exist_ok=True)
    save_pair_model(model, os.path.join(cache, "pair_model.pkl"))
    save_calibrator(calibrator, os.path.join(cache, "calibrator.pkl"))
    save_singleton_model(singleton_model, os.path.join(cache, "singleton_model.pkl"))
    bundle = {
        "feature_columns": list(feats_fit.columns),
        "normalizer_version": "v1_port",
        "decision": cfg.get("decision", {}),
        "val_report": {},
    }
    save_json(bundle, os.path.join(cache, "bundle.json"))

    return model, calibrator, singleton_model, feats_val, pairs_val


def evaluate_stage(
    model: Any,
    calibrator: Any,
    singleton_model: Any,
    canon_train: pd.DataFrame,
    feats_val: pd.DataFrame,
    pairs_val: pd.DataFrame,
    val_ids: list[str],
    truth: dict[str, list[str]],
    cfg: dict[str, Any],
    cache: str,
) -> dict[str, Any]:
    """Run decision engine on validation and report macro-F0.5."""
    dec_val, q_val, _, _, _ = predict_split(feats_val, val_ids, model, calibrator, singleton_model, canon_train, cfg)
    detail = score_detailed(dec_val, truth, val_ids)
    print(f"[trainer] validation macro-F0.5: {detail['macro_f05']:.4f}")
    print(f"[trainer] singleton slice: {detail.get('slices', {}).get('singleton', 0):.4f}")
    save_json(detail, os.path.join(cache, "validation_scores.json"))

    # update bundle
    bundle_path = os.path.join(cache, "bundle.json")
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    bundle["val_report"] = {
        "macro_f05": round(detail["macro_f05"], 4),
        "slices": {k: round(v, 4) for k, v in detail.get("slices", {}).items()},
    }
    save_json(bundle, bundle_path)
    return detail


def predict_stage(
    model: Any,
    calibrator: Any,
    singleton_model: Any,
    canon_test: pd.DataFrame,
    cfg: dict[str, Any],
    out_dir: str,
) -> dict[str, Any]:
    """Run inference on test and write submission files."""
    s23_test = canon_test[canon_test["source"].isin(["S2", "S3"])].reset_index(drop=True)
    test_s1 = canon_test[canon_test["source"] == "S1"].reset_index(drop=True)
    test_s1_ids = test_s1["entity_id"].tolist()

    print("[trainer] blocking + featurizing test split...")
    pairs_test, feats_test = _block_and_featurize(test_s1, s23_test, canon_test, cfg)

    print("[trainer] predicting test...")
    dec_test, q_test, _, _, _ = predict_split(feats_test, test_s1_ids, model, calibrator, singleton_model, canon_test, cfg)

    valid_s23 = set(s23_test["entity_id"])
    write_result = write_outputs(test_s1_ids, dec_test, pairs_test, valid_s23, out_dir, cfg)
    print(f"[trainer] validator passed: {write_result.get('validation_passed')}")
    save_json(write_result, os.path.join(out_dir, "submission_check.json"))
    return write_result


def submit_stage(out_dir: str, data_root: str) -> int:
    """Run the official validator from student_resource."""
    matching = os.path.join(out_dir, "matching_results.tsv")
    candidate = os.path.join(out_dir, "candidate_pairs.tsv")
    test_dir = os.path.join(data_root, "test")
    errors = validate(matching, candidate, test_dir)
    if errors:
        print("[submit] VALIDATION FAIL:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return 1
    print("[submit] VALIDATION PASS")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V2 Business Entity Resolution — Master Trainer")
    parser.add_argument("--config", default="config.yaml", help="Config YAML")
    parser.add_argument("--data-root", default="data", help="Root folder containing train/ and test/")
    parser.add_argument("--stage", default="all",
                        choices=["prepare", "train", "evaluate", "predict", "submit", "all"],
                        help="Trainer stage")
    parser.add_argument("--cache", default=None, help="Cache directory (default from config)")
    parser.add_argument("--output", default=None, help="Output directory (default from config)")
    parser.add_argument("--ladder", default=None,
                        choices=["100k", "300k", "500k", "800k"],
                        help="Override minimal_fit round1_entities for scaling ladder")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_root = Path(args.data_root).resolve()
    cfg["paths"]["train_dir"] = str(data_root / "train")
    cfg["paths"]["test_dir"] = str(data_root / "test")

    cache = args.cache or cfg["paths"]["cache_dir"]
    out_dir = args.output or cfg["paths"]["out_dir"]
    os.makedirs(cache, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # load selection if available
    selection = None
    selection_path = Path(cfg["paths"].get("profile_dir", "profile")) / "selection.json"
    if selection_path.exists():
        selection = load_selection(str(selection_path.parent))
        print(f"[trainer] loaded selection: fit={selection.get('round1_entities')}, calib={selection.get('calib_entities')}, val={selection.get('val_entities')}")

    # ladder override
    if args.ladder and selection:
        cfg["training"]["minimal_fit"]["round1_entities"] = int(args.ladder.replace("k", "000"))
        print(f"[trainer] ladder override: round1_entities={cfg['training']['minimal_fit']['round1_entities']}")

    started = time.time()

    if args.stage in ("prepare", "train", "evaluate", "predict", "submit", "all"):
        print("[trainer] canonicalizing...")
        canon_train, canon_test, truth, _ = canonicalize_splits(str(data_root), cfg)
        fit_ids, calib_ids, val_ids = build_splits_with_selection(canon_train, truth, cfg, selection)
        print(f"[trainer] splits: fit={len(fit_ids)}, calib={len(calib_ids)}, val={len(val_ids)}")
        save_json({"fit": fit_ids, "calib": calib_ids, "val": val_ids}, os.path.join(cache, "split_ids.json"))

    model = calibrator = singleton_model = None
    feats_val = pairs_val = None

    if args.stage in ("train", "evaluate", "predict", "submit", "all"):
        model, calibrator, singleton_model, feats_val, pairs_val = train_stage(
            canon_train, truth, fit_ids, calib_ids, val_ids, cfg, cache
        )

    if args.stage in ("evaluate", "predict", "submit", "all"):
        detail = evaluate_stage(
            model, calibrator, singleton_model, canon_train, feats_val, pairs_val,
            val_ids, truth, cfg, cache
        )

    if args.stage in ("predict", "submit", "all"):
        predict_stage(model, calibrator, singleton_model, canon_test, cfg, out_dir)

    if args.stage in ("submit", "all"):
        rc = submit_stage(out_dir, str(data_root))
        if rc != 0:
            return rc

    print(f"[trainer] stage={args.stage} completed in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
