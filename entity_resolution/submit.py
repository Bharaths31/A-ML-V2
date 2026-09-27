#!/usr/bin/env python3
"""
Submission helper — validates the output files and packages the final zip.

Two actions in one place:
  1. validate()  runs the official stdlib-only utils/validate_submission.py
                 against output/matching_results.tsv + output/candidate_pairs.tsv
                 and data/test/.
  2. package()   builds <team_name>_submission.zip in the exact required layout:

        <team_name>_submission.zip
        ├── output/
        │   ├── matching_results.tsv
        │   └── candidate_pairs.tsv
        ├── code/
        │   └── business_entity_resolution/
        │       ├── src/                 # all source code
        │       ├── README.md
        │       └── requirements.txt
        └── Documentation_template.md

Usage:
    python submit.py --team-name myteam                 # validate, then package
    python submit.py --team-name myteam --validate-only
    python submit.py --team-name myteam --package-only
    python submit.py --team-name myteam --skip-validate  # package without the gate (not recommended)

Exit codes: 0 = success, 1 = validation failed / packaging error.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Source files/folders copied into code/business_entity_resolution/src/
SOURCE_FILES = [
    "master_profiler.py",
    "master_trainer.py",
    "run_train.py",
    "run_test.py",
    "profiler.py",
    "profiler_lib.py",
    "blocker.py",
    "trainer.py",
    "tester.py",
    "features.py",
    "normalizer.py",
    "metrics.py",
    "io_utils.py",
    "cuda_utils.py",
    "setup.py",
    "submit.py",
    "config.yaml",
]
SOURCE_DIRS = ["er_core"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def log(msg: str) -> None:
    print(f"[submit] {msg}", flush=True)


# ---------------------------------------------------------------------------
# action 1: validate
# ---------------------------------------------------------------------------

def find_validator(data_root: Path) -> Path:
    candidates = [
        data_root / "utils" / "validate_submission.py",
        PROJECT_ROOT / "data" / "utils" / "validate_submission.py",
        PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py",
        PROJECT_ROOT.parent / "student_resource" / "utils" / "validate_submission.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find utils/validate_submission.py. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def validate(
    matching: Path,
    candidate: Path,
    test_dir: Path,
    validator: Path,
) -> tuple[bool, str]:
    """Run the official validator. Returns (passed, output_text)."""
    if not matching.exists():
        return False, f"matching_results.tsv not found: {matching}"
    if not candidate.exists():
        return False, f"candidate_pairs.tsv not found: {candidate}"
    if not test_dir.exists():
        return False, f"test dir not found: {test_dir}"

    cmd = [
        sys.executable, str(validator),
        "--matching", str(matching),
        "--candidate", str(candidate),
        "--test-dir", str(test_dir),
    ]
    log("running validator: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


# ---------------------------------------------------------------------------
# action 2: package
# ---------------------------------------------------------------------------

def _copy_tree(src: Path, dst: Path) -> None:
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_PARTS]
        rel = os.path.relpath(root, src)
        target_dir = dst if rel == "." else dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(".pyc"):
                continue
            shutil.copy2(Path(root) / name, target_dir / name)


def package(
    team_name: str,
    out_dir: Path,
    data_root: Path,
    zip_path: Path | None = None,
) -> Path:
    """Build the submission zip. Returns the zip path."""
    matching = out_dir / "matching_results.tsv"
    candidate = out_dir / "candidate_pairs.tsv"
    if not matching.exists() or not candidate.exists():
        raise FileNotFoundError(
            f"output files missing: need both {matching} and {candidate}. "
            "Run master_trainer.py --stage predict first."
        )

    requirements = PROJECT_ROOT / "requirements.txt"
    readme = PROJECT_ROOT / "README.md"

    # Documentation template: prefer data/, fall back to project root / student_resource.
    doc_candidates = [
        data_root / "Documentation_template.md",
        PROJECT_ROOT / "Documentation_template.md",
        PROJECT_ROOT.parent / "student_resource" / "Documentation_template.md",
    ]
    doc = next((p for p in doc_candidates if p.exists()), None)

    zip_path = zip_path or (PROJECT_ROOT / f"{team_name}_submission.zip")

    with tempfile.TemporaryDirectory(prefix="er_submission_") as tmp:
        stage = Path(tmp)
        # output/
        (stage / "output").mkdir(parents=True, exist_ok=True)
        shutil.copy2(matching, stage / "output" / "matching_results.tsv")
        shutil.copy2(candidate, stage / "output" / "candidate_pairs.tsv")

        # code/business_entity_resolution/{src, README.md, requirements.txt}
        code_dir = stage / "code" / "business_entity_resolution"
        src_dir = code_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        for name in SOURCE_FILES:
            p = PROJECT_ROOT / name
            if p.exists():
                shutil.copy2(p, src_dir / name)
            else:
                log(f"WARNING: source file missing, skipping: {name}")
        for name in SOURCE_DIRS:
            p = PROJECT_ROOT / name
            if p.exists():
                _copy_tree(p, src_dir / name)
            else:
                log(f"WARNING: source dir missing, skipping: {name}")
        if readme.exists():
            shutil.copy2(readme, code_dir / "README.md")
        if requirements.exists():
            shutil.copy2(requirements, code_dir / "requirements.txt")

        # Documentation_template.md at zip root
        if doc:
            shutil.copy2(doc, stage / "Documentation_template.md")
        else:
            log("WARNING: Documentation_template.md not found; zip will omit it")

        # Zip the staged tree
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(stage):
                dirs[:] = [d for d in dirs if d not in SKIP_PARTS]
                for name in files:
                    full = Path(root) / name
                    arcname = str(full.relative_to(stage))
                    zf.write(full, arcname)

    log(f"wrote {zip_path}")
    return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate and package the submission")
    parser.add_argument("--team-name", default="team", help="Team name for <team_name>_submission.zip")
    parser.add_argument("--output-dir", default=None, help="Folder with matching_results.tsv (default: output/)")
    parser.add_argument("--data-root", default=None, help="Folder with train/ and test/ (default: data/)")
    parser.add_argument("--zip-path", default=None, help="Explicit output zip path")
    parser.add_argument("--validate-only", action="store_true", help="Only run the validator")
    parser.add_argument("--package-only", action="store_true", help="Only build the zip")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Package without running the validator (not recommended)")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else PROJECT_ROOT / "output"
    data_root = Path(args.data_root).resolve() if args.data_root else PROJECT_ROOT / "data"
    matching = out_dir / "matching_results.tsv"
    candidate = out_dir / "candidate_pairs.tsv"

    passed = True
    if not args.package_only:
        try:
            validator = find_validator(data_root)
            passed, output = validate(matching, candidate, data_root / "test", validator)
        except FileNotFoundError as e:
            log(str(e))
            passed, output = False, ""
        print(output)
        log("VALIDATION PASS" if passed else "VALIDATION FAIL")
        if args.validate_only:
            return 0 if passed else 1
        if not passed:
            log("Not packaging because validation failed. Fix the issues above, or use --skip-validate.")
            return 1

    if args.validate_only:
        return 0 if passed else 1

    try:
        zip_path = package(
            args.team_name,
            out_dir,
            data_root,
            Path(args.zip_path).resolve() if args.zip_path else None,
        )
    except FileNotFoundError as e:
        log(str(e))
        return 1

    log(f"SUCCESS -> {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
