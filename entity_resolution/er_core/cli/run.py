"""CLI: generate synthetic data, run the pipeline, validate, or score."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from ..config.config import load_config
from ..evaluate.scorer import macro_f05
from ..utils import io
from ..utils.synthetic import generate_dataset
from ..utils.validate_submission import validate


def _parse_overrides(pairs: List[str]):
    overrides = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        target = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides


def _apply_data_root(cfg: dict, data_root: str) -> None:
    cfg["paths"]["train_dir"] = os.path.join(data_root, "train")
    cfg["paths"]["test_dir"] = os.path.join(data_root, "test")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--stage", default="all", choices=["all", "synthdata", "validate", "score"])
    parser.add_argument("--data-root", default=None, help="override dataset root (train/ test/)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="config override key=value")
    # synthetic-data options
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--with-test-gt", action="store_true")
    parser.add_argument("--gt", default=None, help="ground truth for --stage score")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, _parse_overrides(args.set))
    if args.data_root:
        _apply_data_root(cfg, args.data_root)
    if args.out_dir:
        cfg["paths"]["out_dir"] = args.out_dir
    if args.cache_dir:
        cfg["paths"]["cache_dir"] = args.cache_dir
    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.stage == "synthdata":
        root = args.data_root or "dataset"
        paths = generate_dataset(
            root,
            n_train_businesses=args.n_train,
            n_test_businesses=args.n_test,
            seed=cfg.get("seed", 42),
            with_test_gt=args.with_test_gt,
        )
        print(json.dumps(paths, indent=2))
        return 0

    if args.stage == "validate":
        out = cfg["paths"]["out_dir"]
        issues = validate(
            os.path.join(out, "matching_results.tsv"),
            os.path.join(out, "candidate_pairs.tsv"),
            cfg["paths"]["test_dir"],
        )
        if issues:
            print(f"FAIL: {len(issues)} issue(s)")
            for index, issue in enumerate(issues, start=1):
                print(f"{index}. {issue}")
            return 1
        print("PASS")
        return 0

    if args.stage == "score":
        if not args.gt:
            print("--gt is required for --stage score")
            return 2
        truth = io.read_ground_truth(args.gt)
        out = cfg["paths"]["out_dir"]
        match_df = None
        from ..utils import io as _io  # noqa: F401

        predictions = {}
        with open(os.path.join(out, "matching_results.tsv"), encoding="utf-8") as handle:
            header = handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    predictions[parts[0]] = io.split_id_list(parts[1])
        entities = sorted(truth.keys())
        score = macro_f05(predictions, truth, entities)
        print(f"macro_f05 = {score:.4f} over {len(entities)} entities")
        return 0

    # default: full pipeline
    from ..pipeline import run_pipeline

    result = run_pipeline(cfg)
    print(json.dumps(result["experiment"], indent=2))
    if result["submission"]["errors"]:
        print("SUBMISSION ERRORS:")
        for issue in result["submission"]["errors"]:
            print(" -", issue)
        return 1
    print(
        f"\nValidation macro-F0.5: {result['validation']['macro_f05']:.4f} | "
        f"blocking recall (union): {result['blocking_val_union']['recall_ceiling']:.4f} | "
        f"blocking recall (final): {result['blocking_val']['recall_ceiling']:.4f} | "
        f"mean candidates: {result['blocking_val']['mean_candidates_per_s1']:.2f} | "
        f"validator: {'PASS' if result['submission']['validation_passed'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
