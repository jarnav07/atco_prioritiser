"""CLI: python -m harness.validate_sub submissions/<run_id>.csv"""
import sys

import pandas as pd

from .data import load_test
from .submission import InvalidSubmission, validate

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sub = pd.read_csv(sys.argv[1], dtype={"prediction": str})
    try:
        validate(sub, load_test())
    except InvalidSubmission as e:
        sys.exit(f"INVALID: {e}")
    print(f"OK: {sys.argv[1]} ({len(sub)} samples)")
