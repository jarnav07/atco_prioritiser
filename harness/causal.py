"""Causality check for feature builders: a snapshot at time t may only use rows of its session with t_s <= t.

Mechanical test: build the features on whole sessions and on the same sessions cut at random times; every row at or
before the cut must get bit-identical values. A feature that reads any later snapshot fails.

    python -m harness.causal raw+f6          # check a FEATURE_SETS entry on 3 labelled sessions x 4 cuts
"""
import sys

import numpy as np
import pandas as pd

from .config import SEED


def check(build, df: pd.DataFrame, n_sessions: int = 3, n_cuts: int = 4, seed: int = SEED) -> list[str]:
    """Returns the columns that differ between the full and the truncated build (empty = causal)."""
    rng = np.random.default_rng(seed)
    bad = set()
    for ses in rng.choice(df["session"].unique(), size=n_sessions, replace=False):
        part = df[df["session"] == ses]
        full = build(part)
        times = np.sort(part["t_s"].unique())
        for cut in rng.choice(times[1:-1], size=n_cuts, replace=False):
            keep = (part["t_s"] <= cut).to_numpy()
            trunc = build(part[keep])
            a, b = full[keep], trunc
            for c in a.columns:
                x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
                if not np.array_equal(x, y, equal_nan=True):
                    bad.add(c)
    return sorted(bad)


if __name__ == "__main__":
    from features import FEATURE_SETS
    from harness.data import load_labelled
    from models.lgbm_ranker import build_features

    name = sys.argv[1] if len(sys.argv) > 1 else "raw"
    assert name in FEATURE_SETS, f"unknown feature set {name}"
    bad = check(lambda d: build_features(d, name), load_labelled())
    if bad:
        sys.exit(f"NOT CAUSAL ({len(bad)} columns read later snapshots): {bad[:20]}")
    print(f"{name}: causal (rows up to each cut identical under truncation)")
