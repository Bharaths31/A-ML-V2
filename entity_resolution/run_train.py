#!/usr/bin/env python3
"""Train the V2 matcher (convenience wrapper around master_trainer.py --stage all up to evaluate)."""
from __future__ import annotations

import sys

from master_trainer import main

if __name__ == "__main__":
    # default: run everything except test prediction
    args = ["--stage", "evaluate"] if len(sys.argv) == 1 else sys.argv[1:]
    sys.exit(main(args))
