#!/usr/bin/env python3
"""
Master profiler CLI.

Stages:
  integrity   : row counts, malformed, duplicate ids, GT coverage, matched-id resolvability
  profile     : field quality, noise, address forensics, GT stats, drift
  select      : build selection.json for the minimal training set
  report      : render report.md from profile.json
  all         : run integrity + profile + report + select (if config present)

Usage:
    python master_profiler.py --stage all --data-root data --out profile
    python master_profiler.py --stage select --data-root data --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from profiler_lib import build_profile, run_integrity, run_profile, run_select, run_report


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V2 Business Entity Resolution — Master Profiler")
    parser.add_argument("--data-root", default="data", help="Root folder containing train/ and test/")
    parser.add_argument("--out", default="profile", help="Output folder for profile artifacts")
    parser.add_argument("--stage", default="all",
                        choices=["integrity", "profile", "select", "report", "all"],
                        help="Profiler stage to run")
    parser.add_argument("--config", default="config.yaml", help="Config YAML (needed for select)")
    parser.add_argument("--sample-s1", type=int, default=100000,
                        help="S1 sample size for blocking simulation (future)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_path = out_dir / "profile.json"
    selection_path = out_dir / "selection.json"
    report_path = out_dir / "report.md"

    if args.stage == "integrity":
        integrity = run_integrity(str(data_root))
        save_json(integrity, out_dir / "integrity.json")
        print(json.dumps(integrity, indent=2))
        return 0

    if args.stage == "profile":
        profile = build_profile(str(data_root), sample_s1=args.sample_s1, seed=args.seed)
        save_json(profile, profile_path)
        print(f"Profile written to {profile_path}")
        return 0

    if args.stage == "select":
        if not profile_path.exists():
            print(f"ERROR: {profile_path} not found. Run --stage profile first.")
            return 1
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        config = load_config(args.config)
        selection = run_select(str(data_root), profile, config)
        save_json(selection, selection_path)
        print(f"Selection written to {selection_path}")
        print(json.dumps({k: v for k, v in selection.items() if k not in ("fit_s1_ids", "calib_s1_ids", "val_s1_ids")}, indent=2))
        return 0

    if args.stage == "report":
        if not profile_path.exists():
            print(f"ERROR: {profile_path} not found. Run --stage profile first.")
            return 1
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        text = run_report(profile, str(report_path))
        print(f"Report written to {report_path}")
        print(text[:500] + "...")
        return 0

    # --stage all
    print("[profiler] running integrity...")
    integrity = run_integrity(str(data_root))
    save_json(integrity, out_dir / "integrity.json")
    print("[profiler] running profile...")
    profile = build_profile(str(data_root), sample_s1=args.sample_s1, seed=args.seed)
    profile["integrity"] = integrity
    save_json(profile, profile_path)
    print("[profiler] running report...")
    run_report(profile, str(report_path))
    print("[profiler] running select...")
    config = load_config(args.config)
    selection = run_select(str(data_root), profile, config)
    save_json(selection, selection_path)
    print(f"[profiler] done. Artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
