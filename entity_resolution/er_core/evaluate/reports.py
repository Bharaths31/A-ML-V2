"""Reporting: EDA, blocking attribution, calibration, decision diagnostics."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def eda_report(records: pd.DataFrame, truth: Dict[str, Sequence[str]] | None = None,
               label: str = "train") -> Dict[str, object]:
    report: Dict[str, object] = {"label": label}
    report["n_records"] = int(len(records))
    report["by_source"] = {
        str(k): int(v) for k, v in records["source"].value_counts().items()
    }
    report["by_country"] = {
        str(k): int(v) for k, v in records["country"].value_counts().items()
    }
    report["null_rate"] = {
        column: float((records[column].astype(str).str.len() == 0).mean())
        for column in ("business_name", "business_address", "country")
        if column in records.columns
    }
    if truth is not None:
        fanouts = [len(truth.get(e, [])) for e in records.loc[records["source"] == "S1", "entity_id"]]
        if fanouts:
            report["s1_entities"] = len(fanouts)
            report["singleton_rate"] = float(np.mean([f == 0 for f in fanouts]))
            report["mean_fanout"] = float(np.mean(fanouts))
            report["fanout_histogram"] = {
                str(k): int(v) for k, v in pd.Series(fanouts).value_counts().sort_index().items()
            }
        total = sum(len(v) for v in truth.values())
        from_s2 = sum(1 for ids in truth.values() for x in ids if x.startswith("S2-"))
        from_s3 = sum(1 for ids in truth.values() for x in ids if x.startswith("S3-"))
        report["truth_pairs_total"] = total
        report["truth_from_s2"] = from_s2
        report["truth_from_s3"] = from_s3
    return report


def save_report(report: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)


def decision_diagnostics(
    decisions: Dict[str, List[str]],
    truth: Dict[str, Sequence[str]],
    entity_ids: Sequence[str],
) -> Dict[str, object]:
    rows = []
    for entity in entity_ids:
        pred = set(decisions.get(entity, []))
        true = set(truth.get(entity, []))
        rows.append(
            {
                "entity": entity,
                "pred_size": len(pred),
                "truth_size": len(true),
                "correct": len(pred & true),
                "abstained": len(pred) == 0,
                "is_singleton": len(true) == 0,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    singleton = frame[frame["is_singleton"]]
    non_singleton = frame[~frame["is_singleton"]]
    return {
        "n_entities": int(len(frame)),
        "chosen_k_distribution": {
            str(k): int(v) for k, v in frame["pred_size"].value_counts().sort_index().items()
        },
        "singleton_abstain_rate": float(singleton["abstained"].mean()) if len(singleton) else 0.0,
        "non_singleton_predicted_rate": float((~non_singleton["abstained"]).mean())
        if len(non_singleton)
        else 0.0,
        "singleton_false_merge_rate": float((~singleton["abstained"]).mean())
        if len(singleton)
        else 0.0,
    }


def append_experiment(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame = pd.DataFrame([row])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
