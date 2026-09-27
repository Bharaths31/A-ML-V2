import pandas as pd

from src.normalize.canonicalize import (
    canonicalize_address,
    canonicalize_frame,
    canonicalize_name,
    load_assets,
)

ASSETS = load_assets()


def test_suffix_stripping():
    result = canonicalize_name("Alpha Traders Pvt Ltd", ASSETS)
    assert result["had_legal_suffix"] == 1
    assert result["name_nosuffix"] == "alpha traders"


def test_punctuation_and_ampersand():
    a = canonicalize_name("Smith & Sons, Inc.", ASSETS)
    b = canonicalize_name("Smith and Sons Inc", ASSETS)
    assert a["name_nosuffix"] == b["name_nosuffix"]


def test_word_order_sorted_key():
    a = canonicalize_name("Global Foods", ASSETS)
    b = canonicalize_name("Foods Global", ASSETS)
    assert a["name_sortedtok"] == b["name_sortedtok"]


def test_address_components():
    result = canonicalize_address("12 Main St, Springfield, IL 62704", ASSETS)
    assert result["addr_house_no"] == "12"
    assert result["addr_pin"] == "62704"
    assert "springfield" in result["addr_street_tokens"]
    assert result["has_pin"] == 1


def test_landmark_isolated():
    result = canonicalize_address("Near SBI ATM, 5 Market Road, Pune 411001", ASSETS)
    assert result["has_landmark"] == 1
    assert "market" in result["addr_street_tokens"]
    assert "sbi" not in result["addr_street_tokens"]


def test_country_agnostic_frame():
    df = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "source": ["S1"],
            "business_name": ["Cafe Lyon SARL"],
            "business_address": ["3 Rue Victor Hugo, Lyon 69003"],
            "country": ["France"],
        }
    )
    out = canonicalize_frame(df, ASSETS)
    assert out.loc[0, "name_nosuffix"] == "cafe lyon"
    assert out.loc[0, "addr_pin"] == "69003"
