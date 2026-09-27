#!/usr/bin/env python3
"""
V2 profiler implementation: streaming, cheap, deterministic.

Stages:
  - integrity   : row counts, malformed, duplicate ids, GT coverage
  - profile     : field quality, noise, address forensics, GT stats, drift
  - select      : profiler-driven minimal training set (uses GT + exact-layer structure)
  - report      : markdown rendering of profile.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# schema constants
# ---------------------------------------------------------------------------

DEFAULT_PROFILE_SCHEMA = "2.0"

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _path(root: str, *parts: str) -> str:
    return os.path.join(root, *parts)


def _file_size(path: str) -> int:
    return os.path.getsize(path)


def _head_sha1(path: str, nbytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def _read_rows(path: str):
    """Stream rows from a TSV; yields dicts."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def _count_rows(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for _ in f:
            if _.strip():
                n += 1
    return n


def _script_class(text: str) -> str:
    if not text:
        return "empty"
    # detect Devanagari block
    if any("\u0900" <= c <= "\u097f" for c in text):
        return "devanagari"
    # detect extended latin / accents
    if any(ord(c) > 127 for c in text):
        return "extended_latin"
    return "ascii_latin"


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _is_empty(v: str | None) -> bool:
    return v is None or v.strip() == ""


def _split_ids(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _pin_zip(text: str) -> str | None:
    """Extract a 4-6 digit postal/ZIP candidate from the tail of an address."""
    if not text:
        return None
    # prefer digits near the end
    m = list(re.finditer(r"\b\d{4,6}\b", text))
    if not m:
        return None
    return m[-1].group()


def _house_number(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"^\s*(\d+[-/\d]*)\b", text)
    return m.group(1) if m else None


def _first_name_token(name: str) -> str | None:
    if not name:
        return None
    toks = name.lower().split()
    return toks[0] if toks else None


def _city_token(text: str) -> str | None:
    """Naive city proxy: last comma-separated token before any trailing ZIP."""
    if not text:
        return None
    # remove trailing digits
    cleaned = re.sub(r"\b\d{4,6}\b", "", text)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts[-1].lower() if parts else None


def _street_type_token(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    for pat in ["st", "street", "rd", "road", "ave", "av", "avenue", "blvd", "bd", "boulevard",
                "ln", "lane", "dr", "drive", "nagar", "colony", "sector", "rue", "chemin"]:
        if re.search(rf"\b{pat}\b", lowered):
            return pat
    return None

# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------

def run_integrity(data_root: str) -> dict[str, Any]:
    train_dir = _path(data_root, "train")
    test_dir = _path(data_root, "test")

    files = {
        "train_source1": _path(train_dir, "train_source1.tsv"),
        "train_source2": _path(train_dir, "train_source2.tsv"),
        "train_source3": _path(train_dir, "train_source3.tsv"),
        "train_ground_truth": _path(train_dir, "train_ground_truth.tsv"),
        "test_source1": _path(test_dir, "test_source1.tsv"),
        "test_source2": _path(test_dir, "test_source2.tsv"),
        "test_source3": _path(test_dir, "test_source3.tsv"),
    }

    integrity = {"files": {}, "malformed_rows": {}, "duplicate_entity_ids": {},
                 "gt_covers_all_s1": None, "matched_ids_resolvable": None}

    for key, path in files.items():
        if not os.path.exists(path):
            integrity["files"][key] = {"exists": False}
            continue
        rows = _count_rows(path)
        integrity["files"][key] = {
            "exists": True,
            "rows": rows,
            "bytes": _file_size(path),
            "sha1_head": _head_sha1(path),
        }

    # malformed + dup ids per source file
    for key in ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]:
        path = files[key]
        if not os.path.exists(path):
            continue
        malformed = 0
        seen = set()
        dups = set()
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    malformed += 1
                    continue
                eid = parts[0].strip()
                if eid in seen:
                    dups.add(eid)
                seen.add(eid)
        integrity["malformed_rows"][key] = malformed
        integrity["duplicate_entity_ids"][key] = len(dups)

    # GT coverage
    gt_path = files["train_ground_truth"]
    s1_path = files["train_source1"]
    if os.path.exists(gt_path) and os.path.exists(s1_path):
        gt_s1 = set()
        with open(gt_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p:
                    gt_s1.add(p[0].strip())
        s1_ids = set()
        with open(s1_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 1:
                    s1_ids.add(p[0].strip())
        integrity["gt_covers_all_s1"] = (gt_s1 == s1_ids) and len(s1_ids) > 0

    # matched ids resolvable
    gt_path = files["train_ground_truth"]
    if os.path.exists(gt_path):
        matched = set()
        with open(gt_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    matched.update(_split_ids(p[1]))
        pool_ids = set()
        for key in ["train_source2", "train_source3"]:
            path = files[key]
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 1:
                        pool_ids.add(p[0].strip())
        integrity["matched_ids_resolvable"] = matched.issubset(pool_ids) and len(matched) > 0
        integrity["n_matched_ids"] = len(matched)

    return integrity

# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

# pre-compiled regexes for speed
_RE_PIN = re.compile(r"\b\d{4,6}\b")
_RE_HOUSE = re.compile(r"^\s*(\d+[-/\d]*)\b")
_RE_NEAR = re.compile(r"\bnear\b", re.IGNORECASE)
_RE_GARBAGE_PREFIX = re.compile(r"^[^A-Za-z0-9\u0900-\u097f]+")
_RE_DOMAIN = re.compile(r"\.[a-z]{2,6}$")
_RE_DBA = re.compile(r"\b(dba|trading as|t/a|a unit of)\b", re.IGNORECASE)
_STREET_TYPES = ["st", "street", "rd", "road", "ave", "av", "avenue", "blvd", "bd", "boulevard",
                 "ln", "lane", "dr", "drive", "nagar", "colony", "sector", "rue", "chemin"]
_RE_STREET = re.compile(r"\b(?:" + "|".join(map(re.escape, _STREET_TYPES)) + r")\b")


def _script_class_fast(text: str) -> str:
    if not text:
        return "empty"
    for c in text:
        o = ord(c)
        if 0x0900 <= o <= 0x097F:
            return "devanagari"
        if o > 127:
            return "extended_latin"
    return "ascii_latin"


def _city_token_fast(text: str) -> str | None:
    if not text:
        return None
    i = len(text)
    while i > 0 and (text[i - 1].isdigit() or text[i - 1].isspace() or text[i - 1] == ","):
        i -= 1
    cleaned = text[:i]
    idx = cleaned.rfind(",")
    tok = cleaned[idx + 1:].strip() if idx >= 0 else cleaned.strip()
    return tok.lower() if tok else None


def _scan_source(path: str) -> dict[str, Any]:
    """Single-pass scan of a source file: field quality + address forensics + noise."""
    n = 0
    countries = Counter()

    name_lengths = []
    name_empty = 0
    name_scripts = Counter()
    name_punct_sum = 0.0
    name_digit_sum = 0.0
    name_upper = 0
    name_lower = 0

    addr_lengths = []
    addr_empty = 0
    addr_scripts = Counter()

    by_country = defaultdict(lambda: {"n": 0, "pin": 0, "house": 0, "city": 0,
                                      "street_type": 0, "missing": 0, "landmark": 0})

    noise_sample_every = 1000
    garbage_prefix = 0
    domain_as_name = 0
    dba = 0
    sampled = 0

    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            n += 1
            name = p[1]
            addr = p[2]
            country = p[3].strip()
            countries[country] += 1

            if _is_empty(name):
                name_empty += 1
                name_scripts["empty"] += 1
                name_lengths.append(0)
            else:
                name_lengths.append(len(name))
                name_scripts[_script_class_fast(name)] += 1
                L = len(name)
                name_punct_sum += sum(1 for c in name if not c.isalnum() and not c.isspace()) / max(L, 1)
                name_digit_sum += sum(1 for c in name if c.isdigit()) / max(L, 1)
                if name.isupper():
                    name_upper += 1
                elif name.islower():
                    name_lower += 1

            if _is_empty(addr):
                addr_empty += 1
                addr_scripts["empty"] += 1
                addr_lengths.append(0)
            else:
                addr_lengths.append(len(addr))
                addr_scripts[_script_class_fast(addr)] += 1

            by_country[country]["n"] += 1
            addr_l = addr.lower().strip() if addr else ""
            if not addr_l:
                by_country[country]["missing"] += 1
            else:
                if _RE_PIN.search(addr_l):
                    by_country[country]["pin"] += 1
                if _RE_HOUSE.match(addr_l):
                    by_country[country]["house"] += 1
                if _city_token_fast(addr_l):
                    by_country[country]["city"] += 1
                if _RE_STREET.search(addr_l):
                    by_country[country]["street_type"] += 1
                if _RE_NEAR.search(addr_l):
                    by_country[country]["landmark"] += 1

            if i % noise_sample_every == 0 and name:
                sampled += 1
                if _RE_GARBAGE_PREFIX.match(name):
                    garbage_prefix += 1
                if _RE_DOMAIN.search(name.lower()):
                    domain_as_name += 1
                if _RE_DBA.search(name):
                    dba += 1

    name_lengths_sorted = sorted(name_lengths)
    addr_lengths_sorted = sorted(addr_lengths)

    address_forensics = {}
    for c, stats in by_country.items():
        nc = stats["n"]
        address_forensics[c] = {
            "rows": nc,
            "missing_rate": round(stats["missing"] / nc, 4),
            "pin_rate": round(stats["pin"] / nc, 4),
            "house_no_rate": round(stats["house"] / nc, 4),
            "city_rate": round(stats["city"] / nc, 4),
            "street_type_rate": round(stats["street_type"] / nc, 4),
            "landmark_rate": round(stats["landmark"] / nc, 4),
        }

    return {
        "rows": n,
        "countries": dict(countries),
        "business_name": {
            "rows": n,
            "empty_rate": round(name_empty / max(n, 1), 6),
            "mean_length": round(sum(name_lengths) / max(len(name_lengths), 1), 2),
            "p50_length": round(_percentile(name_lengths_sorted, 0.5), 1),
            "p95_length": round(_percentile(name_lengths_sorted, 0.95), 1),
            "p99_length": round(_percentile(name_lengths_sorted, 0.99), 1),
            "script_distribution": dict(name_scripts),
            "punctuation_ratio": round(name_punct_sum / max(n, 1), 4),
            "digit_ratio": round(name_digit_sum / max(n, 1), 4),
            "uppercase_rate": round(name_upper / max(n, 1), 4),
            "lowercase_rate": round(name_lower / max(n, 1), 4),
        },
        "business_address": {
            "rows": n,
            "empty_rate": round(addr_empty / max(n, 1), 6),
            "mean_length": round(sum(addr_lengths) / max(len(addr_lengths), 1), 2),
            "p50_length": round(_percentile(addr_lengths_sorted, 0.5), 1),
            "p95_length": round(_percentile(addr_lengths_sorted, 0.95), 1),
            "p99_length": round(_percentile(addr_lengths_sorted, 0.99), 1),
            "script_distribution": dict(addr_scripts),
        },
        "address_forensics": address_forensics,
        "noise": {
            "sampled_rows": sampled,
            "garbage_prefix_rate": round(garbage_prefix / max(sampled, 1), 4),
            "domain_as_name_rate": round(domain_as_name / max(sampled, 1), 4),
            "dba_rate": round(dba / max(sampled, 1), 4),
        },
    }


def _gt_analysis(data_root: str, s1_country: dict[str, str]) -> dict[str, Any]:
    gt_path = _path(data_root, "train", "train_ground_truth.tsv")
    if not os.path.exists(gt_path):
        return {}

    total = 0
    empty = 0
    fanout_counts = Counter()
    has_s2 = 0
    has_s3 = 0
    both = 0
    fan_by_country = defaultdict(list)
    singleton_by_country = Counter()
    nonempty_by_country = Counter()
    cross_country = 0
    total_pairs = 0

    # resolve country of matched ids (expensive but one-time)
    matched = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            ids = _split_ids(p[1])
            matched.update(ids)
    match_country: dict[str, str] = {}
    for key in ["train_source2.tsv", "train_source3.tsv"]:
        path = _path(data_root, "train", key)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 4 and p[0].strip() in matched:
                    match_country[p[0].strip()] = p[3].strip()

    with open(gt_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            total += 1
            s1 = p[0].strip()
            ids = _split_ids(p[1])
            c1 = s1_country.get(s1, "?")
            if not ids:
                empty += 1
                singleton_by_country[c1] += 1
                continue
            k = len(ids)
            fanout_counts[k] += 1
            fan_by_country[c1].append(k)
            nonempty_by_country[c1] += 1
            total_pairs += k
            if any(x.startswith("S2-") for x in ids):
                has_s2 += 1
            if any(x.startswith("S3-") for x in ids):
                has_s3 += 1
            if any(x.startswith("S2-") for x in ids) and any(x.startswith("S3-") for x in ids):
                both += 1
            for mid in ids:
                if match_country.get(mid, c1) != c1:
                    cross_country += 1

    nonempty = total - empty
    avg_fan = sum((k * v) for k, v in fanout_counts.items()) / max(nonempty, 1)
    return {
        "rows": total,
        "singletons": {"count": empty, "rate": round(empty / max(total, 1), 6),
                       "by_country": dict(singleton_by_country)},
        "nonempty": nonempty,
        "avg_fanout": round(avg_fan, 3),
        "max_fanout": max(fanout_counts.keys()) if fanout_counts else 0,
        "fanout_distribution": dict(sorted(fanout_counts.items())),
        "propensity": {
            "has_s2": round(has_s2 / max(nonempty, 1), 4),
            "has_s3": round(has_s3 / max(nonempty, 1), 4),
            "both": round(both / max(nonempty, 1), 4),
        },
        "total_true_pairs": total_pairs,
        "cross_country_pairs": cross_country,
        "avg_fanout_by_country": {c: round(sum(vs) / max(len(vs), 1), 3) for c, vs in fan_by_country.items()},
    }


def run_profile(data_root: str, sample_s1: int = 100000, seed: int = 42) -> dict[str, Any]:
    random.seed(seed)

    s1_country: dict[str, str] = {}
    s1_path = _path(data_root, "train", "train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 4:
                s1_country[p[0].strip()] = p[3].strip()

    sources = {}
    for key in ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]:
        if key.startswith("test"):
            path = _path(data_root, "test", f"{key}.tsv")
        else:
            path = _path(data_root, "train", f"{key}.tsv")
        if not os.path.exists(path):
            continue
        sources[key] = _scan_source(path)

    gt = _gt_analysis(data_root, s1_country)

    # train vs test drift (S1 only)
    train_s1_path = _path(data_root, "train", "train_source1.tsv")
    test_s1_path = _path(data_root, "test", "test_source1.tsv")
    drift = {}
    for label, path in [("train", train_s1_path), ("test", test_s1_path)]:
        if not os.path.exists(path):
            continue
        lengths = []
        with open(path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 4:
                    lengths.append(len(p[1]))
        drift[label] = {
            "name_len_p50": round(_percentile(sorted(lengths), 0.5), 1),
            "name_len_p95": round(_percentile(sorted(lengths), 0.95), 1),
        }

    return {
        "sources": sources,
        "ground_truth": gt,
        "drift": drift,
        "sample": {"n_s1": sample_s1, "seed": seed},
    }


# ---------------------------------------------------------------------------
# select (profiler-driven minimal training set)
# ---------------------------------------------------------------------------

def _load_id_country_map(path: str) -> dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 4:
                out[p[0].strip()] = p[3].strip()
    return out


def _exact_keys(entity_id: str, name: str, addr: str, country: str) -> dict[str, str | None]:
    """Return the exact-layer keys for one record."""
    name_l = name.lower().strip() if name else ""
    addr_l = addr.lower().strip() if addr else ""
    first_tok = _first_name_token(name_l)
    pin = _pin_zip(addr_l)
    city = _city_token(addr_l)
    house = _house_number(addr_l)
    return {
        "name_canon": name_l,
        "first_token": first_tok,
        "pin": pin,
        "city": city,
        "house": house,
        "country": country,
    }


def _compute_blocking_structure(data_root: str) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, int]]:
    """
    Build an exact-layer-only blocking structure for selection.
    Returns (s1_keys, s23_buckets, s1_candidate_counts).
    """
    s1_keys: dict[str, dict] = {}
    s23_buckets: dict[str, list[str]] = defaultdict(list)

    # load train S1
    with open(_path(data_root, "train", "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 4:
                eid, name, addr, country = [x.strip() for x in p]
                s1_keys[eid] = _exact_keys(eid, name, addr, country)

    # load train S2/S3 into buckets
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        with open(_path(data_root, "train", fname), "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) != 4:
                    continue
                eid, name, addr, country = [x.strip() for x in p]
                keys = _exact_keys(eid, name, addr, country)
                # bucket by exact keys
                for k in ["name_canon", "first_token", "pin", "city", "house"]:
                    v = keys.get(k)
                    if v:
                        s23_buckets[f"{k}:{country}:{v}"].append(eid)

    # count candidates per S1 under exact-layer union
    counts = {}
    for eid, keys in s1_keys.items():
        cands = set()
        for k in ["name_canon", "first_token", "pin", "city", "house"]:
            v = keys.get(k)
            if v:
                cands.update(s23_buckets.get(f"{k}:{keys['country']}:{v}", []))
        counts[eid] = len(cands)
    return s1_keys, s23_buckets, counts


def run_select(data_root: str, profile: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Build selection.json for the minimal training set."""
    cfg = config.get("training", {}).get("minimal_fit", {})
    if not cfg.get("enabled", True):
        return {"enabled": False, "note": "minimal_fit disabled in config"}

    round1_n = cfg.get("round1_entities", 300000)
    probe_n = cfg.get("probe_entities", 150000)
    round2_max = cfg.get("round2_max_add", 150000)
    calib_n = cfg.get("calib_entities", 100000)
    val_n = cfg.get("val_entities", 200000)

    # load GT and S1 countries
    gt_path = _path(data_root, "train", "train_ground_truth.tsv")
    s1_country = _load_id_country_map(_path(data_root, "train", "train_source1.tsv"))
    gt: dict[str, list[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                gt[p[0].strip()] = _split_ids(p[1])

    # compute strata
    strata = defaultdict(list)
    _, _, cand_counts = _compute_blocking_structure(data_root)

    for eid, country in s1_country.items():
        matches = gt.get(eid, [])
        fanout = len(matches)
        singleton = fanout == 0
        fanout_bucket = "5+" if fanout >= 5 else str(fanout)
        has_s2 = any(x.startswith("S2-") for x in matches)
        has_s3 = any(x.startswith("S3-") for x in matches)
        if has_s2 and has_s3:
            propensity = "both"
        elif has_s2:
            propensity = "s2_only"
        elif has_s3:
            propensity = "s3_only"
        else:
            propensity = "none"
        # script: cheap proxy using S1 name class
        script = _script_class(eid)  # placeholder; better: compare with matched names
        # weak-layer: count exact keys with candidates (proxy for how "easy" the entity is)
        n_cands = cand_counts.get(eid, 0)
        competitor_dense = n_cands >= 20
        # address completeness placeholder
        addr_complete = "unknown"
        key = (country, str(singleton), fanout_bucket, propensity, script, addr_complete, competitor_dense)
        strata[key].append(eid)

    # allocate round1: proportional within country, with hard-strata oversampling
    random.seed(config.get("seed", 42))
    round1_ids: set[str] = set()
    hard_keys = {("true", "5+"), ("true", "1"), ("s2_only",), ("s3_only",), (True,)}  # rough

    # simple allocation: shuffle each stratum, take proportional targets
    total_nonempty = sum(1 for v in gt.values() if v)
    target_per_entity = round1_n / max(len(s1_country), 1)
    for key, ids in strata.items():
        is_hard = any(h in key for h in hard_keys)
        weight = 2.0 if is_hard else 1.0
        n_take = max(2, int(len(ids) * target_per_entity * weight / sum(1 for _ in strata.values())))
        random.shuffle(ids)
        round1_ids.update(ids[:n_take])

    # clamp to round1_n
    if len(round1_ids) > round1_n:
        round1_ids = set(random.sample(sorted(round1_ids), round1_n))

    # calib/val: natural stratified from remaining
    remaining = [eid for eid in s1_country if eid not in round1_ids]
    random.shuffle(remaining)
    calib_ids = set(remaining[:calib_n])
    val_ids = set(remaining[calib_n:calib_n + val_n])

    per_stratum_counts = Counter()
    for eid in round1_ids:
        # recompute key for reporting
        matches = gt.get(eid, [])
        fanout = len(matches)
        per_stratum_counts[f"fanout_{fanout if fanout < 5 else '5+'}"] += 1

    return {
        "enabled": True,
        "round1_entities": len(round1_ids),
        "calib_entities": len(calib_ids),
        "val_entities": len(val_ids),
        "fit_s1_ids": sorted(round1_ids),
        "calib_s1_ids": sorted(calib_ids),
        "val_s1_ids": sorted(val_ids),
        "per_stratum_counts": dict(per_stratum_counts),
        "note": "Round 2 (probe/uncertainty) is run by master_trainer after pilot training",
    }

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def run_report(profile: dict[str, Any], out_path: str) -> str:
    lines = ["# V2 Profiler Report", f"Generated: {datetime.utcnow().isoformat()}Z", ""]

    integrity = profile.get("integrity", {})
    lines.append("## Integrity")
    lines.append(f"- GT covers all S1: {integrity.get('gt_covers_all_s1')}")
    lines.append(f"- Matched IDs resolvable: {integrity.get('matched_ids_resolvable')} ({integrity.get('n_matched_ids')} ids)")
    lines.append("")

    gt = profile.get("ground_truth", {})
    lines.append("## Ground Truth")
    lines.append(f"- Singleton rate: {gt.get('singletons', {}).get('rate', 0):.4f}")
    lines.append(f"- Avg fanout: {gt.get('avg_fanout', 0):.2f}, max: {gt.get('max_fanout', 0)}")
    lines.append(f"- Cross-country pairs: {gt.get('cross_country_pairs', 'N/A')}")
    lines.append("")

    sources = profile.get("sources", {})
    lines.append("## Sources")
    for key, info in sources.items():
        lines.append(f"### {key}")
        lines.append(f"- rows: {info.get('rows')}")
        lines.append(f"- countries: {info.get('countries')}")
        lines.append(f"- address missing rate: {info.get('business_address', {}).get('empty_rate', 0):.4f}")
        lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

# ---------------------------------------------------------------------------
# full profile builder
# ---------------------------------------------------------------------------

def build_profile(data_root: str, sample_s1: int = 100000, seed: int = 42) -> dict[str, Any]:
    integrity = run_integrity(data_root)
    profile_body = run_profile(data_root, sample_s1, seed)
    return {
        "schema_version": DEFAULT_PROFILE_SCHEMA,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "normalizer_version": "v1_port",
        "integrity": integrity,
        **profile_body,
    }
