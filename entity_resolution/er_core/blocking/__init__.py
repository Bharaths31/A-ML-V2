"""Blocking cascade: exact keys, MinHash LSH, TF-IDF top-k, safety nets."""

LAYER_BITS = {
    "A1": 1 << 0,   # canonical-name exact
    "A2": 1 << 1,   # name-minus-suffix exact
    "A3": 1 << 2,   # sorted-token-set exact
    "A5": 1 << 3,   # (pin, first-name-token)
    "A6": 1 << 4,   # (city, first-name-token)
    "B1": 1 << 5,   # char-3-gram MinHash LSH, name
    "B2": 1 << 6,   # token MinHash LSH, street tokens
    "B3": 1 << 7,   # TF-IDF char-3-gram top-k, name
    "C1": 1 << 8,   # (country, rare-name-token)
    "C2": 1 << 9,   # (house number, first-name-token)
}

EXACT_LAYERS = ("A1", "A2", "A3", "A5", "A6", "C2")
EXACT_BITS = 0
for _name in EXACT_LAYERS:
    EXACT_BITS |= LAYER_BITS[_name]


def layers_to_mask(layer_names):
    mask = 0
    for name in layer_names:
        mask |= LAYER_BITS[name]
    return mask


def mask_to_layers(mask: int):
    return [name for name, bit in LAYER_BITS.items() if mask & bit]
