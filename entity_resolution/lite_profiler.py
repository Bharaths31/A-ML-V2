#!/usr/bin/env python3
"""
Lite profiler — the essential half of profiling, sampled, in ~1-3 minutes.

Skips the expensive parts of the full profiler:
  * no matched-id resolvability scan (the 10M-row pool pass)
  * no cross-country analysis
  * no full blocking-structure build (that's the `select` stage's job)
  * sampled field scans via --stride

What it still gives you (enough to drive blocking/training decisions):
  * file inventory (rows, bytes)
  * field quality: empty rate, length p50/p95/p99, script mix, noise rates
  * ground truth: singleton rate, fanout distribution, S2/S3 propensity
  * country mix per source

Usage:
    python lite_profiler.py --data-root data --out profile --stride 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
_RE_PIN = re.compile(r"\b\d{4,6}\b")
_RE_GARBAGE = re.compile(r"^[^A-Za-z0-9\u0900-\u097f]+")
_RE_DOMAIN = re.compile(r"\.[a-z]{2,6}$")

SOURCES = ["train_source1", "train_source2", "train_source3",
           "test_source1", "test_source2", "test_source3"]


def log(msg: str) -> None:
    print(f"[lite] {msg}", flush=True)


def _script_class(text: str) -> str:
    if not text:
        return "empty"
    for c in text:
        o = ord(c)
        if 0x0900 <= o <= 0x097F:
            return "devanagari"
        if o > 127:
            return "extended_latin"
    return "ascii_latin"


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def count_rows(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if line.strip():
                n += 1
    return n


def scan_source_lite(path: str, stride: int = 20) -> dict:
    """Sampled single-pass field stats."""
    n = 0
    countries = Counter()
    name_len, addr_len = [], []
    name_empty = addr_empty = 0
    scripts = Counter()
    garbage = domain = 0
    sampled = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            if i % stride:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            n += 1
            name, addr, country = p[1], p[2], p[3]
            countries[country] += 1
            if not name.strip():
                name_empty += 1
                scripts["empty"] += 1
            else:
                name_len.append(len(name))
                scripts[_script_class(name)] += 1
            if not addr.strip():
                addr_empty += 1
            else:
                addr_len.append(len(addr))
            if name:
                if _RE_GARBAGE.match(name):
                    garbage += 1
                if _RE_DOMAIN.search(name.lower()):
                    domain += 1
            sampled += 1
    name_len.sort(); addr_len.sort()
    return {
        "sampled_rows": n,
        "stride": stride,
        "countries": dict(countries),
        "name_empty_rate": round(name_empty / max(sampled, 1), 5),
        "addr_empty_rate": round(addr_empty / max(sampled, 1), 5),
        "name_len_p50": round(_pct(name_len, 0.5), 1),
        "name_len_p95": round(_pct(name_len, 0.95), 1),
        "addr_len_p50": round(_pct(addr_len, 0.5), 1),
        "addr_len_p95": round(_pct(addr_len, 0.95), 1),
        "script_distribution": dict(scripts),
        "garbage_prefix_rate": round(garbage / max(sampled, 1), 4),
        "domain_as_name_rate": round(domain / max(sampled, 1), 4),
    }


def gt_analysis(gt_path: str) -> dict:
    total = empty = 0
    fanout = Counter()
    has_s2 = has_s3 = both = 0
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            total += 1
            ids = [x.strip() for x in p[1].split(",") if x.strip()]
            if not ids:
                empty += 1
                continue
            fanout[len(ids)] += 1
            s2 = any(x.startswith("S2-") for x in ids)
            s3 = any(x.startswith("S3-") for x in ids)
            if s2:
                has_s2 += 1
            if s3:
                has_s3 += 1
            if s2 and s3:
                both += 1
    nonempty = total - empty
    return {
        "rows": total,
        "singleton_rate": round(empty / max(total, 1), 5),
        "singleton_count": empty,
        "nonempty": nonempty,
        "avg_fanout": round(sum(k * v for k, v in fanout.items()) / max(nonempty, 1), 3),
        "max_fanout": max(fanout) if fanout else 0,
        "fanout_distribution": dict(sorted(fanout.items())),
        "has_s2_rate": round(has_s2 / max(nonempty, 1), 4),
        "has_s3_rate": round(has_s3 / max(nonempty, 1), 4),
        "both_rate": round(both / max(nonempty, 1), 4),
    }


def render_report(prof: dict, path: str) -> None:
    lines = ["# Lite Profile Report", ""]
    lines.append(f"- generated: {prof['generated_at']}")
    lines.append(f"- stride: {prof['stride']} (sampled field stats)")
    lines.append("")
    lines.append("## Ground truth")
    gt = prof["ground_truth"]
    lines.append(f"- singleton rate: {gt['singleton_rate']:.4f} ({gt['singleton_count']})")
    lines.append(f"- avg fanout: {gt['avg_fanout']}  max: {gt['max_fanout']}")
    lines.append(f"- S2 propensity: {gt['has_s2_rate']}  S3: {gt['has_s3_rate']}  both: {gt['both_rate']}")
    lines.append("")
    lines.append("## Sources")
    for key, s in prof["sources"].items():
        lines.append(f"### {key}")
        lines.append(f"- rows: {s['rows']}  bytes: {s['bytes']}")
        lines.append(f"- countries: {s['fields']['countries']}")
        lines.append(f"- name empty: {s['fields']['name_empty_rate']}  addr empty: {s['fields']['addr_empty_rate']}")
        lines.append(f"- name len p50/p95: {s['fields']['name_len_p50']}/{s['fields']['name_len_p95']}")
        lines.append(f"- scripts: {s['fields']['script_distribution']}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lite profiler (fast, sampled)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="profile")
    ap.add_argument("--stride", type=int, default=20)
    args = ap.parse_args(argv)

    t0 = time.time()
    root = Path(args.data_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    prof = {"schema_version": "lite-1.0", "stride": args.stride,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sources": {}}

    for key in SOURCES:
        split, name = key.split("_", 1)
        sub = "train" if split == "train" else "test"
        path = root / sub / f"{key}.tsv"
        if not path.exists():
            log(f"missing {path}")
            continue
        log(f"scanning {key} (stride={args.stride})...")
        rows = count_rows(str(path))
        fields = scan_source_lite(str(path), args.stride)
        prof["sources"][key] = {"rows": rows, "bytes": path.stat().st_size, "fields": fields}

    gt_path = root / "train" / "train_ground_truth.tsv"
    if gt_path.exists():
        log("ground-truth analysis...")
        prof["ground_truth"] = gt_analysis(str(gt_path))

    (out / "lite_profile.json").write_text(json.dumps(prof, indent=2), encoding="utf-8")
    render_report(prof, str(out / "lite_report.md"))
    log(f"wrote {out/'lite_profile.json'} and {out/'lite_report.md'} in {time.time()-t0:.1f}s")

    # short console summary
    gt = prof.get("ground_truth", {})
    print(json.dumps({
        "singleton_rate": gt.get("singleton_rate"),
        "avg_fanout": gt.get("avg_fanout"),
        "max_fanout": gt.get("max_fanout"),
        "propensity": {"s2": gt.get("has_s2_rate"), "s3": gt.get("has_s3_rate")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
