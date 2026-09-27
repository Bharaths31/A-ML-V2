#!/usr/bin/env python3
"""Run V2 inference on the test set (convenience wrapper around master_trainer.py --stage predict+submit)."""
from __future__ import annotations

import sys

from master_trainer import main

if __name__ == "__main__":
    # default: predict + submit
    args = ["--stage", "submit"] if len(sys.argv) == 1 else sys.argv[1:]
    sys.exit(main(args))
