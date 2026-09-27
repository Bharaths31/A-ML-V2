#!/usr/bin/env python3
"""
FAST submission path — for when the window is tight (~30 min).

A deliberately small, memory-safe, country-partitioned pipeline that trades
recall ceiling for speed and gets a *validated, scored* submission on the board:

    blocking = exact normalized-name (+ sorted-token + capped PIN) hash keys
    features = 6 cheap lexical similarities
    model    = LogisticRegression (trained on a small labeled sample)
    decision = per-entity threshold tuned for macro-F0.5

Run:
    python fast_submit.py --data-root data --output output --train-entities 150000

It does NOT touch master_profiler/master_trainer; it's a standalone fallback.
"""
from __future__ import annotations

import argparse
import os
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# normalization (C-fast str.translate, Devanagari/accents preserved)
# ---------------------------------------------------------------------------
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})

LEGAL_SUFFIXES = {
    "corp", "corporation", "co", "company", "inc", "incorporated", "ltd", "limited",
    "llc", "llp", "lp", "pvt", "private", "pte", "plc", "gmbh", "mbh", "sarl",
    "sas", "sasu", "sa", "sci", "snc", "eurl",
}

_RE_PIN = re.compile(r"\b\d{4,6}\b")


def toks(s: str) -> tuple:
    s = s.lower().replace("&", " and ").translate(_PUNCT_TABLE)
    return tuple(t for t in s.split() if t)


def nosuffix(tokens: tuple) -> tuple:
    return tuple(t for t in tokens if t not in LEGAL_SUFFIXES)


def sorted_key(tokens: tuple) -> tuple:
    return tuple(sorted(nosuffix(tokens)))


def pin(addr: str) -> str:
    m = _RE_PIN.findall(addr)
    return m[-1] if m else ""


def city(addr: str) -> str:
    cleaned = _RE_PIN.sub("", addr)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts[-1].lower() if parts else ""


def jaccard(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa) + len(sb) - inter
    return inter / union if union else 0.0


def entity_f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    p = tp / len(pred)
    r = tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r) if p and r else 0.0


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def read_gt(path: str) -> dict:
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                gt[p[0].strip()] = set(x.strip() for x in p[1].split(",") if x.strip())
    return gt


def read_source(path: str, max_rows: int = 0) -> dict:
    """Return {entity_id: (name_toks, addr_toks, pin, city, country)}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            eid, name, addr, country = p[0], p[1], p[2], p[3]
            out[eid] = (toks(name), toks(addr), pin(addr), city(addr), country)
    return out


def build_pool(paths, country: str, max_rows: int = 0):
    """Build exact-key buckets + fields for one country. Returns (b_nos, b_srt, b_pin, fields)."""
    b_nos = defaultdict(list)
    b_srt = defaultdict(list)
    b_pin = defaultdict(list)
    fields = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            next(f, None)
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                p = line.rstrip("\n").split("\t")
                if len(p) != 4:
                    continue
                eid, name, addr, c = p[0], p[1], p[2], p[3]
                if c != country:
                    continue
                nt = toks(name)
                ns = nosuffix(nt)
                sk = sorted_key(nt)
                pi = pin(addr)
                if ns:
                    b_nos[ns].append(eid)
                if sk:
                    b_srt[sk].append(eid)
                if pi:
                    b_pin[pi].append(eid)
                fields[eid] = (nt, toks(addr), pi, city(addr))
    return b_nos, b_srt, b_pin, fields


PIN_BUCKET_CAP = 20


def candidates(nt, ns, sk, pi, b_nos, b_srt, b_pin) -> list:
    cands = set()
    if ns:
        cands.update(b_nos.get(ns, ()))
    if sk:
        cands.update(b_srt.get(sk, ()))
    if pi:
        pb = b_pin.get(pi, ())
        if len(pb) <= PIN_BUCKET_CAP:
            cands.update(pb)
    return list(cands)


def featurize(nt, at, pi, ci, pool_fields, cand_ids) -> tuple:
    X = []
    for cid in cand_ids:
        pnt, pat, ppi, pci = pool_fields[cid]
        f = [
            float(nt == pnt),
            float(nosuffix(nt) == nosuffix(pnt)),
            jaccard(nt, pnt),
            jaccard(at, pat),
            float(pi != "" and pi == ppi),
            float(ci != "" and ci == pci),
        ]
        X.append(f)
    return X


def log(msg: str) -> None:
    print(f"[fast] {msg}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Fast validated submission path")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output", default="output")
    ap.add_argument("--train-entities", type=int, default=150000,
                    help="Cap on labeled training S1 entities (total)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override threshold (default: auto-tune on val)")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Debug: cap rows per file (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    root = Path(args.data_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    train_dir = root / "train"
    test_dir = root / "test"

    # ---- load ground truth + S1 train ----
    log("loading ground truth + S1 train...")
    gt = read_gt(str(train_dir / "train_ground_truth.tsv"))
    s1_train = read_source(str(train_dir / "train_source1.tsv"), args.max_rows)
    log(f"s1 train entities: {len(s1_train)} (max_rows={args.max_rows})")

    # ---- training pairs, country-partitioned ----
    rng = np.random.default_rng(args.seed)
    train_rows = []   # (feature vec, label)
    train_country_pool = {}

    # build pool index once per country and reuse for training
    for country in ["US", "India"]:
        log(f"building TRAIN pool index for {country}...")
        b_nos, b_srt, b_pin, fields = build_pool(
            [str(train_dir / "train_source2.tsv"), str(train_dir / "train_source3.tsv")],
            country, args.max_rows)
        train_country_pool[country] = (b_nos, b_srt, b_pin, fields)
        log(f"  pool size {country}: {len(fields)}")

    # sample training entities (include all singletons for hard negatives)
    s1_train_ids = list(s1_train.keys())
    rng.shuffle(s1_train_ids)
    cap = args.train_entities
    if cap and cap < len(s1_train_ids):
        non_single = [e for e in s1_train_ids if gt.get(e)]
        single = [e for e in s1_train_ids if not gt.get(e)]
        non_single = non_single[: int(cap * 0.9)]
        single = single[: int(cap * 0.1)]
        train_sample = non_single + single
        rng.shuffle(train_sample)
    else:
        train_sample = s1_train_ids

    val_sample = train_sample[-max(1, len(train_sample) // 5):]
    fit_sample = train_sample[: -len(val_sample)]

    log(f"featurizing training pairs ({len(fit_sample)} entities)...")
    for eid in fit_sample:
        nt, at, pi, ci, country = s1_train[eid]
        ns = nosuffix(nt)
        sk = sorted_key(nt)
        b_nos, b_srt, b_pin, fields = train_country_pool.get(country, (None, None, None, None))
        if b_nos is None:
            continue
        cands = candidates(nt, ns, sk, pi, b_nos, b_srt, b_pin)
        truth = gt.get(eid, set())
        X = featurize(nt, at, pi, ci, fields, cands)
        for cid, f in zip(cands, X):
            train_rows.append((f, 1 if cid in truth else 0))

    log(f"training LogisticRegression on {len(train_rows)} pairs...")
    X = np.asarray([r[0] for r in train_rows], dtype=np.float32)
    y = np.asarray([r[1] for r in train_rows], dtype=np.int8)
    if len(np.unique(y)) < 2:
        log("single-class training labels; falling back to threshold rule")
        model = None
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y)

    # ---- threshold selection on val sample ----
    best_thresh = args.threshold or 0.6
    if args.threshold is None and model is not None:
        log("tuning threshold on val sample...")
        # featurize val once, then sweep thresholds cheaply
        val_ents = []   # (cands, Xv, truth)
        for eid in val_sample:
            nt, at, pi, ci, country = s1_train[eid]
            ns = nosuffix(nt); sk = sorted_key(nt)
            b_nos, b_srt, b_pin, fields = train_country_pool.get(country, (None, None, None, None))
            if b_nos is None:
                continue
            cands = candidates(nt, ns, sk, pi, b_nos, b_srt, b_pin)
            truth = gt.get(eid, set())
            if not cands:
                val_ents.append(([], None, truth))
                continue
            Xv = np.asarray(featurize(nt, at, pi, ci, fields, cands), dtype=np.float32)
            pv = model.predict_proba(Xv)[:, 1]
            val_ents.append((cands, pv, truth))

        best_score, best_t = -1.0, 0.6
        for t in np.arange(0.30, 0.91, 0.05):
            scores = []
            for cands, pv, truth in val_ents:
                if pv is None:
                    pred = set()
                else:
                    pred = {cid for cid, pr in zip(cands, pv) if pr >= t}
                scores.append(entity_f05(pred, truth))
            sc = float(np.mean(scores)) if scores else 0.0
            if sc > best_score:
                best_score, best_t = sc, float(t)
        best_thresh = best_t
        log(f"  best threshold {best_thresh:.2f} (val macro-F0.5 {best_score:.4f})")
    log(f"final threshold: {best_thresh:.3f}")

    # free training memory
    del s1_train, train_rows, X, y, train_country_pool, gt

    # ---- inference on test ----
    log("loading S1 test...")
    s1_test = read_source(str(test_dir / "test_source1.tsv"), args.max_rows)
    log(f"s1 test entities: {len(s1_test)}")

    matching_path = out / "matching_results.tsv"
    candidate_path = out / "candidate_pairs.tsv"
    countries = ["US", "India", "France"]

    with open(matching_path, "w", encoding="utf-8", newline="\n") as fm, \
         open(candidate_path, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in countries:
            log(f"building TEST pool index for {country}...")
            b_nos, b_srt, b_pin, fields = build_pool(
                [str(test_dir / "test_source2.tsv"), str(test_dir / "test_source3.tsv")],
                country, args.max_rows)
            log(f"  pool size {country}: {len(fields)}")
            n_ent = 0
            for eid, (nt, at, pi, ci, c) in s1_test.items():
                if c != country:
                    continue
                n_ent += 1
                ns = nosuffix(nt); sk = sorted_key(nt)
                cands = candidates(nt, ns, sk, pi, b_nos, b_srt, b_pin)
                fc.write(f"{eid}\t{','.join(cands)}\n")
                if not cands:
                    fm.write(f"{eid}\t\n")
                    continue
                if model is None:
                    # rule fallback: exact name or (jaccard>0.8 and pin match)
                    pred = []
                    for cid in cands:
                        pnt, pat, ppi, pci = fields[cid]
                        if nosuffix(nt) == nosuffix(pnt) or (jaccard(nt, pnt) > 0.8 and pi != "" and pi == ppi):
                            pred.append(cid)
                    fm.write(f"{eid}\t{','.join(pred)}\n")
                else:
                    Xt = featurize(nt, at, pi, ci, fields, cands)
                    pv = model.predict_proba(np.asarray(Xt, dtype=np.float32))[:, 1]
                    pred = [cid for cid, pr in zip(cands, pv) if pr >= best_thresh]
                    fm.write(f"{eid}\t{','.join(pred)}\n")
            log(f"  processed {country}: {n_ent} entities")
            del b_nos, b_srt, b_pin, fields

    log(f"wrote {matching_path} and {candidate_path}")
    log(f"total elapsed: {time.time() - t0:.1f}s")

    # ---- validate ----
    validator = root / "utils" / "validate_submission.py"
    if not validator.exists():
        validator = root / ".." / "utils" / "validate_submission.py"
    if validator.exists():
        import subprocess
        r = subprocess.run(
            [sys.executable, str(validator),
             "--matching", str(matching_path),
             "--candidate", str(candidate_path),
             "--test-dir", str(test_dir)],
            capture_output=True, text=True)
        print((r.stdout or "") + (r.stderr or ""))
        return 0 if r.returncode == 0 else 1
    log("WARNING: validator not found; skipping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
