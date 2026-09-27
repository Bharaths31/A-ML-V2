"""Synthetic dataset generator.

Produces a challenge-shaped dataset (train/test sources + ground truth) so the
pipeline can be exercised end-to-end without the official data. The generator
mimics the documented noise catalog and, by default, introduces an unseen
country (France) only in the test split to simulate the zero-shot shift.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import pandas as pd

NAME_HEADS = [
    "Alpha", "Bravo", "Sunrise", "Global", "Blue", "Prime", "Metro", "Royal",
    "Silver", "Golden", "Green", "Star", "Unity", "Progress", "Modern", "National",
    "City", "United", "Grand", "Elite", "Nova", "Orbit", "Pioneer", "Summit",
    "Bright", "Everest", "Horizon", "Crystal", "Falcon", "Titan", "Apex", "Vista",
    "Noble", "Pearl", "Quantum", "Radiant", "Sapphire", "Trident", "Vertex", "Zenith",
    "Coastal", "Highland", "Imperial", "Liberty", "Meridian", "Northern", "Pacific", "Southern",
]
NAME_MIDDLE = [
    "Star", "Bridge", "Point", "Gate", "Wood", "Field", "Stone", "River", "Bay", "Hill",
    "Park", "Lake", "Valley", "Grove", "Court", "Plaza", "Square", "Cross", "Way", "View",
]
NAME_TAILS = [
    "Traders", "Enterprises", "Industries", "Solutions", "Services", "Agencies",
    "Stores", "Distributors", "Holdings", "Exports", "Foods", "Motors", "Textiles",
    "Pharma", "Electronics", "Builders", "Logistics", "Caterers", "Printers", "Traders",
    "Furnishings", "Hardware", "Systems", "Networks", "Labs", "Works", "Mills", "Plastics",
    "Metals", "Chemicals", "Packaging", "Fashions", "Jewellers", "Opticals", "Medicals",
    "Automobiles", "Realty", "Infotech", "Consultants", "Marketing",
]
SUFFIXES = ["", " Inc", " LLC", " Pvt Ltd", " Ltd", " Corp", " Co", " SARL", " SAS"]
STREETS = ["Main", "Market", "Park", "Lake", "Hill", "Garden", "Station", "Church", "Mill", "Bridge"]
STREET_TYPES = {
    "US": ["St", "Ave", "Rd", "Blvd", "Ln"],
    "IN": ["Road", "Nagar", "Colony", "Sector", "Cross"],
    "FR": ["Rue", "Avenue", "Boulevard", "Chemin", "Quai"],
}
CITIES = {
    "US": ["Springfield", "Riverside", "Franklin", "Greenville", "Clinton"],
    "IN": ["Mumbai", "Pune", "Nagpur", "Surat", "Indore"],
    "FR": ["Lyon", "Nantes", "Lille", "Toulouse", "Bordeaux"],
}
STATES = {
    "US": ["IL", "CA", "TX", "NY", "OH"],
    "IN": ["MH", "KA", "GJ", "TN", "UP"],
    "FR": ["69", "44", "59", "31", "33"],
}
TIPOS = [("suffix_swap", 0.35), ("typo", 0.25), ("word_order", 0.15), ("punct", 0.25)]
ADDR_TIPOS = [("abbrev", 0.3), ("missing_pin", 0.2), ("landmark", 0.15), ("reorder", 0.2), ("none", 0.15)]


def _pin(rng: random.Random, country: str) -> str:
    if country == "IN":
        return f"{rng.randint(400000, 499999)}"
    if country == "FR":
        return f"{rng.randint(10000, 95999):05d}"
    return f"{rng.randint(10000, 99999):05d}"


def _make_business(rng: random.Random, country: str) -> Dict[str, object]:
    head = rng.choice(NAME_HEADS)
    tail = rng.choice(NAME_TAILS)
    middle = rng.choice(NAME_MIDDLE) if rng.random() < 0.5 else ""
    suffix = rng.choice(SUFFIXES)
    core = f"{head} {middle} {tail}".strip() if middle else f"{head} {tail}"
    name = f"{core}{suffix}"
    city = rng.choice(CITIES[country])
    state = rng.choice(STATES[country])
    street = rng.choice(STREETS)
    stype = rng.choice(STREET_TYPES[country])
    house = str(rng.randint(1, 400))
    pin = _pin(rng, country)
    address = f"{house} {street} {stype}, {city}, {state} {pin}"
    return {"name": name, "country": country, "address": address, "city": city, "pin": pin}


def _noisy_name(rng: random.Random, name: str) -> str:
    roll = rng.random()
    if roll < 0.35:
        for suf in (" Inc", " LLC", " Pvt Ltd", " Ltd", " Corp", " Co", " SARL", " SAS"):
            if name.endswith(suf):
                return name[: -len(suf)]
        return name + rng.choice(SUFFIXES)
    if roll < 0.6 and len(name) > 5:
        position = rng.randrange(1, len(name) - 1)
        return name[:position] + rng.choice("aeiourstn") + name[position + 1 :]
    if roll < 0.75:
        tokens = name.split()
        if len(tokens) >= 2:
            tokens[0], tokens[1] = tokens[1], tokens[0]
            return " ".join(tokens)
    if roll < 0.9:
        return name.replace("&", "and").replace(".", "")
    return name


def _noisy_address(rng: random.Random, address: str, country: str) -> str:
    roll = rng.random()
    if roll < 0.3:
        replacements = {"St": "Street", "Ave": "Avenue", "Rd": "Road", "Blvd": "Boulevard",
                        "Nagar": "Ngr", "Colony": "Col", "Road": "Rd", "Rue": "R."}
        for key, value in replacements.items():
            address = address.replace(key, value)
        return address
    if roll < 0.5:
        parts = address.split(", ")
        return ", ".join(parts[:-1]) if len(parts) > 1 else address
    if roll < 0.65:
        return f"Near {rng.choice(['SBI ATM', 'Post Office', 'City Mall', 'Metro Station'])}, {address}"
    if roll < 0.85:
        parts = address.split(", ")
        if len(parts) >= 2:
            return ", ".join([parts[1], parts[0]] + parts[2:])
        return address
    return address


def generate_dataset(
    out_dir: str,
    n_train_businesses: int = 400,
    n_test_businesses: int = 200,
    countries_train: Tuple[str, ...] = ("US", "IN"),
    countries_test: Tuple[str, ...] = ("US", "IN", "FR"),
    seed: int = 42,
    with_test_gt: bool = True,
    singleton_rate: float = 0.3,
) -> Dict[str, str]:
    rng = random.Random(seed)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    paths: Dict[str, str] = {}

    def build_split(prefix: str, n: int, countries: Tuple[str, ...], id_base: int, write_gt: bool):
        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
        counter = id_base

        def new_id(source: str) -> str:
            nonlocal counter
            counter += 1
            return f"{source}-{counter:06d}"

        for _ in range(n):
            country = rng.choice(countries)
            business = _make_business(rng, country)
            s1_id = new_id("S1")
            s1_rows.append(
                {
                    "entity_id": s1_id,
                    "business_name": _noisy_name(rng, business["name"]),
                    "business_address": _noisy_address(rng, business["address"], country),
                    "country": country,
                }
            )
            matches: List[str] = []
            if rng.random() >= singleton_rate:
                for source, rows in (("S2", s2_rows), ("S3", s3_rows)):
                    for _ in range(rng.choice([1, 1, 1, 2])):
                        rec_id = new_id(source)
                        rows.append(
                            {
                                "entity_id": rec_id,
                                "business_name": _noisy_name(rng, business["name"]),
                                "business_address": _noisy_address(rng, business["address"], country),
                                "country": country,
                            }
                        )
                        matches.append(rec_id)
            # Distractor: a different business in the same city with a similar head name.
            if rng.random() < 0.5:
                source = rng.choice(["S2", "S3"])
                rows = s2_rows if source == "S2" else s3_rows
                distractor = dict(business)
                distractor["name"] = f"{rng.choice(NAME_HEADS)} {rng.choice(NAME_TAILS)}"
                rows.append(
                    {
                        "entity_id": new_id(source),
                        "business_name": _noisy_name(rng, distractor["name"]),
                        "business_address": distractor["address"],
                        "country": country,
                    }
                )
            gt_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "matched_entity_ids": ",".join(matches),
                }
            )

        s1_path = os.path.join(out_dir, prefix, f"{prefix}_source1.tsv")
        s2_path = os.path.join(out_dir, prefix, f"{prefix}_source2.tsv")
        s3_path = os.path.join(out_dir, prefix, f"{prefix}_source3.tsv")
        pd.DataFrame(s1_rows).to_csv(s1_path, sep="\t", index=False)
        pd.DataFrame(s2_rows).to_csv(s2_path, sep="\t", index=False)
        pd.DataFrame(s3_rows).to_csv(s3_path, sep="\t", index=False)
        paths[f"{prefix}_source1"] = s1_path
        paths[f"{prefix}_source2"] = s2_path
        paths[f"{prefix}_source3"] = s3_path
        if write_gt:
            gt_path = os.path.join(out_dir, prefix, f"{prefix}_ground_truth.tsv")
            pd.DataFrame(gt_rows).to_csv(gt_path, sep="\t", index=False)
            paths[f"{prefix}_ground_truth"] = gt_path
        return len(s1_rows)

    build_split("train", n_train_businesses, countries_train, 0, write_gt=True)
    build_split("test", n_test_businesses, countries_test, 500000, write_gt=with_test_gt)
    return paths
