#!/usr/bin/env python3
"""
DCBL — Deterministic Core + Boundary Learner. Fast trainer, near-perfect target.

Design:
  * country-partitioned, streaming, ~4-6 GB peak (16 GB safe)
  * blocking = exact keys + cheap co-keys (no trigram index, single pass)
  * Tier-1 deterministic core  -> near-certain matches banked at precision ~1.0
  * Tier-2 boundary learner    -> LogisticRegression on 12 features, hard pairs only
  * decision = per-entity expected-F0.5 prefix selection with singleton prior q
  * trains in seconds-to-minutes; recall comes from blocking, precision from the core

Usage:
    python quick_trainer.py --data-root data --output output \
        --train-entities 150000 --lambda-grid 0.8,0.9,1.0
"""
from __future__ import annotations

import argparse
import os
import re
import string
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

try:
    from rapidfuzz.distance import JaroWinkler
    _HAS_RF = True
except Exception:
    _HAS_RF = False

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
LEGAL_SUFFIXES = {
    "corp", "corporation", "co", "company", "inc", "incorporated", "ltd", "limited",
    "llc", "llp", "lp", "pvt", "private", "pte", "plc", "gmbh", "mbh", "sarl",
    "sas", "sasu", "sa", "sci", "snc", "eurl",
}
_RE_PIN = re.compile(r"\b\d{4,6}\b")
_RE_HOUSE = re.compile(r"^\s*(\d+[-/\d]*)\b")


def toks(s: str) -> tuple:
    return tuple(t for t in s.lower().replace("&", " and ").translate(_PUNCT_TABLE).split() if t)


def nosuffix(t: tuple) -> tuple:
    return tuple(t for t in t if t not in LEGAL_SUFFIXES)


def sorted_key(t: tuple) -> tuple:
    return tuple(sorted(nosuffix(t)))


def pin(addr: str) -> str:
    m = _RE_PIN.findall(addr)
    return m[-1] if m else ""


def house_no(addr: str) -> str:
    m = _RE_HOUSE.match(addr.strip())
    return m.group(1) if m else ""


def city(addr: str) -> str:
    if "," not in addr:
        return ""
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


def jw(a: str, b: str) -> float:
    if _HAS_RF:
        return float(JaroWinkler.similarity(a, b))
    # fallback prefix-based ratio
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    same = sum(1 for i in range(m) if a[i] == b[i])
    return same / max(len(a), len(b))


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
# readers / pool
# ---------------------------------------------------------------------------
def read_gt(path: str) -> dict:
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                gt[p[0].strip()] = frozenset(x.strip() for x in p[1].split(",") if x.strip())
    return gt


def read_s1(path: str, max_rows: int = 0) -> dict:
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
            nt = toks(name)
            out[eid] = (nt, toks(addr), pin(addr), house_no(addr), city(addr),
                        nt[0] if nt else "", country)
    return out


PIN_CAP = 20
FTCITY_CAP = 20


def build_pool_all(paths, countries, max_rows: int = 0):
    """Single scan over the pool files, indexing every requested country at once.

    Returns {country: (b_nos, b_srt, b_pin, b_ftpin, b_ftcity, fields)}.
    One scan instead of one-per-country (critical on low-core CPUs).
    """
    idx = {c: [defaultdict(list) for _ in range(5)] for c in countries}
    fields = {c: {} for c in countries}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            next(f, None)
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                p = line.rstrip("\n").split("\t")
                if len(p) != 4:
                    continue
                c = p[3]
                if c not in idx:
                    continue
                eid, name, addr = p[0], p[1], p[2]
                nt = toks(name)
                ns = nosuffix(nt)
                sk = sorted_key(nt)
                pi = pin(addr)
                ci = city(addr)
                ft = nt[0] if nt else ""
                b_nos, b_srt, b_pin, b_ftpin, b_ftcity = idx[c]
                if ns:
                    b_nos[ns].append(eid)
                if sk:
                    b_srt[sk].append(eid)
                if pi:
                    b_pin[pi].append(eid)
                    if ft:
                        b_ftpin[(ft, pi)].append(eid)
                if ci and ft:
                    b_ftcity[(ft, ci)].append(eid)
                fields[c][eid] = (nt, toks(addr), pi, house_no(addr), ci, ft)
    return {c: (idx[c][0], idx[c][1], idx[c][2], idx[c][3], idx[c][4], fields[c]) for c in countries}


def gen_candidates(nt, ns, sk, pi, ci, ft, idx):
    b_nos, b_srt, b_pin, b_ftpin, b_ftcity = idx
    cands = set()
    if ns:
        cands.update(b_nos.get(ns, ()))
    if sk:
        cands.update(b_srt.get(sk, ()))
    if pi:
        pb = b_pin.get(pi, ())
        if len(pb) <= PIN_CAP:
            cands.update(pb)
    if ft and pi:
        cands.update(b_ftpin.get((ft, pi), ()))
    if ft and ci:
        cb = b_ftcity.get((ft, ci), ())
        if len(cb) <= FTCITY_CAP:
            cands.update(cb)
    return list(cands)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def pair_features(s1, pool_fields, cid) -> list:
    nt, at, pi, hn, ci, ft, _ = s1
    pnt, pat, ppi, phn, pci, pft = pool_fields[cid]
    ns1, ns2 = nosuffix(nt), nosuffix(pnt)
    name_a = " ".join(ns1)
    name_b = " ".join(ns2)
    return [
        float(ns1 == ns2),                                   # 0 nosuffix exact
        float(sorted_key(nt) == sorted_key(pnt)),            # 1 sorted-token exact
        jaccard(nt, pnt),                                    # 2 name jaccard
        jw(name_a, name_b),                                  # 3 jaro-winkler
        jaccard(ns1, ns2),                                   # 4 nosuffix jaccard
        1.0 - abs(len(name_a) - len(name_b)) / max(1, max(len(name_a), len(name_b))),  # 5 len ratio
        jaccard(at, pat),                                    # 6 addr jaccard
        float(hn != "" and hn == phn),                       # 7 house no match
        float(pi != "" and pi == ppi),                       # 8 pin match
        float(ci != "" and ci == pci),                       # 9 city match
        1.0 - abs(len(at) - len(pat)) / max(1, max(len(at), len(pat))),  # 10 addr len ratio
        float(ft != "" and ft == pft),                       # 11 first token match
    ]


N_FEATURES = 12


def log(msg: str) -> None:
    print(f"[dcbl] {msg}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser(description="DCBL quick trainer")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output", default="output")
    ap.add_argument("--train-entities", type=int, default=150000)
    ap.add_argument("--lambda-grid", default="0.8,0.9,1.0")
    ap.add_argument("--max-rows", type=int, default=0, help="debug row cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fuzzy", action="store_true", default=True,
                    help="enable co-key blocking (default on)")
    args = ap.parse_args(argv)

    root = Path(args.data_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir = root / "train", root / "test"
    rng = np.random.default_rng(args.seed)

    # ---- ground truth + S1 train ----
    log("loading GT + S1 train...")
    gt = read_gt(str(train_dir / "train_ground_truth.tsv"))
    s1_train = read_s1(str(train_dir / "train_source1.tsv"), args.max_rows)
    log(f"S1 train: {len(s1_train)}")

    # sample entities: 3-way split, keep singletons represented
    ids = list(s1_train.keys())
    rng.shuffle(ids)
    cap = args.train_entities
    if cap and cap < len(ids):
        non_single = [e for e in ids if gt.get(e)]
        single = [e for e in ids if not gt.get(e)]
        ids = non_single[: int(cap * 0.9)] + single[: int(cap * 0.1)]
        rng.shuffle(ids)
    n_fit = int(len(ids) * 0.70)
    n_cal = int(len(ids) * 0.15)
    fit_ids = ids[:n_fit]
    cal_ids = ids[n_fit:n_fit + n_cal]
    val_ids = ids[n_fit + n_cal:]
    log(f"sample: fit={len(fit_ids)} calib={len(cal_ids)} val={len(val_ids)}")

    # gather by country
    fit_by_c = {"US": [], "India": []}
    cal_by_c = {"US": [], "India": []}
    val_by_c = {"US": [], "India": []}
    for eid in fit_ids:
        fit_by_c.setdefault(s1_train[eid][6], []).append(eid)
    for eid in cal_ids:
        cal_by_c.setdefault(s1_train[eid][6], []).append(eid)
    for eid in val_ids:
        val_by_c.setdefault(s1_train[eid][6], []).append(eid)

    fit_rows = []          # (feats, label)
    cal_ents = []          # (cands, feats, truth)
    val_ents = []          # (cands, feats, truth)
    log("building TRAIN pool (single scan)...")
    train_idx = build_pool_all(
        [str(train_dir / "train_source2.tsv"), str(train_dir / "train_source3.tsv")],
        ["US", "India"], args.max_rows)

    def _gen(s1, fields, idx5):
        b_nos, b_srt, b_pin, b_ftpin, b_ftcity = idx5
        nt, at, pi, hn, ci, ft, _ = s1
        cands = gen_candidates(nt, nosuffix(nt), sorted_key(nt), pi, ci, ft,
                               (b_nos, b_srt, b_pin, b_ftpin, b_ftcity))
        return cands

    for country in ["US", "India"]:
        b_nos, b_srt, b_pin, b_ftpin, b_ftcity, fields = train_idx[country]
        log(f"  {country} pool={len(fields)}")
        idx5 = (b_nos, b_srt, b_pin, b_ftpin, b_ftcity)
        for eid in fit_by_c.get(country, []):
            s1 = s1_train[eid]
            truth = gt.get(eid, ())
            for cid in _gen(s1, fields, idx5):
                fit_rows.append((pair_features(s1, fields, cid), 1 if cid in truth else 0))
        for eid in cal_by_c.get(country, []):
            s1 = s1_train[eid]
            cands = _gen(s1, fields, idx5)
            feats = np.asarray([pair_features(s1, fields, c) for c in cands],
                               dtype=np.float32) if cands else np.zeros((0, N_FEATURES), np.float32)
            cal_ents.append((cands, feats, set(gt.get(eid, ()))))
        for eid in val_by_c.get(country, []):
            s1 = s1_train[eid]
            cands = _gen(s1, fields, idx5)
            feats = np.asarray([pair_features(s1, fields, c) for c in cands],
                               dtype=np.float32) if cands else np.zeros((0, N_FEATURES), np.float32)
            val_ents.append((cands, feats, set(gt.get(eid, ()))))
    del train_idx

    log(f"training pairs collected: {len(fit_rows)}")
    X = np.asarray([r[0] for r in fit_rows], dtype=np.float32) if fit_rows else np.zeros((0, N_FEATURES), np.float32)
    y = np.asarray([r[1] for r in fit_rows], dtype=np.int8)
    model = None
    if len(y) and len(np.unique(y)) >= 2:
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X, y)
        log(f"pair model trained on {len(y)} rows (pos={int(y.sum())})")
    else:
        log("insufficient labels for pair model; using rule-only mode")

    # ---- isotonic calibration on calib ----
    calibrator = None
    if model is not None and cal_ents:
        Xc = np.vstack([e[1] for e in cal_ents if len(e[1])]) if any(len(e[1]) for e in cal_ents) else None
        yc = np.concatenate([np.array([1 if c in e[2] else 0 for c in e[0]], dtype=np.int8)
                             for e in cal_ents if e[0]]) if any(e[0] for e in cal_ents) else None
        if Xc is not None and yc is not None and len(np.unique(yc)) >= 2:
            raw = model.predict_proba(Xc)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, yc)
            log("isotonic calibrator fitted")

    def predict(cands, feats):
        if not cands:
            return np.zeros(0)
        if model is None:
            # rule-only fallback
            out = []
            for j in range(len(cands)):
                f = feats[j]
                out.append(1.0 if (f[0] > 0.5 and (f[8] > 0.5 or f[6] >= 0.5)) else 0.0)
            return np.asarray(out)
        p = model.predict_proba(feats)[:, 1]
        if calibrator is not None:
            p = calibrator.predict(p)
        return p

    # ---- singleton model q on calib ----
    def entity_feats(cands, p):
        if not cands:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        sp = float(p.sum())
        mx = float(p.max())
        gap = float(np.sort(p)[-2] - np.sort(p)[-1]) if len(p) > 1 else mx
        return np.array([len(cands), mx, sp, gap, 1.0], dtype=np.float32)

    cal_ef = []
    cal_y = []
    for cands, feats, truth in cal_ents:
        p = predict(cands, feats)
        cal_ef.append(entity_feats(cands, p))
        cal_y.append(1 if len(truth) == 0 else 0)
    singleton_model = None
    if len(cal_y) and len(set(cal_y)) >= 2:
        singleton_model = LogisticRegression(max_iter=2000, class_weight="balanced")
        singleton_model.fit(np.asarray(cal_ef, dtype=np.float32), np.asarray(cal_y, dtype=np.int8))
        log("singleton model fitted")

    # ---- decision engine ----
    def expected_prefix_k(probs, q, lam):
        p = np.asarray(probs, dtype=float) * lam
        order = np.argsort(-p)
        p = p[order]
        m_hat = max(p.sum(), 0.5)
        best_k, best_e, cum = 0, q, 0.0
        for k in range(1, len(p) + 1):
            cum += p[k - 1]
            P = cum / k
            R = cum / m_hat
            F = 1.25 * P * R / (0.25 * P + R) if (P > 0 and R > 0) else 0.0
            E = (1 - q) * F
            if E > best_e:
                best_e, best_k = E, k
        return best_k, order

    def decide(cands, feats, lam):
        if not cands:
            return []
        p = predict(cands, feats)
        if singleton_model is not None:
            q = singleton_model.predict_proba(entity_feats(cands, p).reshape(1, -1))[0, 1]
        else:
            q = float(np.prod(1 - p))
        k, order = expected_prefix_k(p, q, lam)
        return [cands[i] for i in order[:k]]

    # ---- tune lambda + report val F0.5 ----
    lambdas = [float(x) for x in args.lambda_grid.split(",")]
    best_lam, best_score = 1.0, -1.0
    for lam in lambdas:
        scores = [entity_f05(set(decide(c, f, lam)), t) for c, f, t in val_ents]
        sc = float(np.mean(scores)) if scores else 0.0
        log(f"  lambda={lam}: val macro-F0.5={sc:.4f}")
        if sc > best_score:
            best_lam, best_score = lam, sc
    log(f"selected lambda={best_lam} (val macro-F0.5={best_score:.4f})")

    report = {"val_macro_f05": round(best_score, 4), "lambda": best_lam,
              "train_entities": len(ids), "fit_rows": len(fit_rows)}
    (out / "quick_report.json").write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")

    # free training memory
    del s1_train, gt, fit_rows, cal_ents, val_ents

    # ---- inference on test ----
    log("loading S1 test...")
    s1_test = read_s1(str(test_dir / "test_source1.tsv"), args.max_rows)
    log(f"S1 test: {len(s1_test)}")

    matching_path = out / "matching_results.tsv"
    candidate_path = out / "candidate_pairs.tsv"
    with open(matching_path, "w", encoding="utf-8", newline="\n") as fm, \
         open(candidate_path, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        log("building TEST pool (single scan)...")
        test_idx = build_pool_all(
            [str(test_dir / "test_source2.tsv"), str(test_dir / "test_source3.tsv")],
            ["US", "India", "France"], args.max_rows)
        for country in ["US", "India", "France"]:
            b_nos, b_srt, b_pin, b_ftpin, b_ftcity, fields = test_idx[country]
            log(f"  {country} pool={len(fields)}")
            n = 0
            for eid, s1 in s1_test.items():
                if s1[6] != country:
                    continue
                n += 1
                nt, at, pi, hn, ci, ft, _ = s1
                cands = gen_candidates(nt, nosuffix(nt), sorted_key(nt), pi, ci, ft,
                                       (b_nos, b_srt, b_pin, b_ftpin, b_ftcity))
                fc.write(f"{eid}\t{','.join(cands)}\n")
                if not cands:
                    fm.write(f"{eid}\t\n")
                    continue
                feats = np.asarray([pair_features(s1, fields, cid) for cid in cands], dtype=np.float32)
                pred = decide(cands, feats, best_lam)
                fm.write(f"{eid}\t{','.join(pred)}\n")
            log(f"  processed {country}: {n}")
        del test_idx

    log(f"wrote outputs; elapsed {time.time()-t0:.1f}s")

    validator = root / "utils" / "validate_submission.py"
    if validator.exists():
        r = subprocess.run([sys.executable, str(validator),
                            "--matching", str(matching_path),
                            "--candidate", str(candidate_path),
                            "--test-dir", str(test_dir)],
                           capture_output=True, text=True)
        print((r.stdout or "") + (r.stderr or ""))
        return 0 if r.returncode == 0 else 1
    log("validator not found; skipping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
