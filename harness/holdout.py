"""Freeze a by-controller train / stop / eval split (complements the 5-fold CV in harness/folds.csv).

- eval  : 3 controllers who flew all three scenarios - the private LB's shape (one unseen controller, 3 sessions).
          Scored once per experiment; never used for fitting or for choosing anything.
- stop  : the other controllers who flew all three scenarios. The cost function is monitored here to choose
          the number of boosting rounds (early stopping).
- train : everyone else.

Run once (`python -m harness.holdout`); every holdout experiment reads harness/holdout.csv.
"""
import numpy as np
import pandas as pd

from .config import ROOT, SEED
from .folds import load as load_folds

HOLDOUT_FILE = ROOT / "harness" / "holdout.csv"
N_EVAL = 3


def build() -> pd.DataFrame:
    f = load_folds()
    per_atco = f.groupby("atco").agg(samples=("sample_id", "size"), scenarios=("scenario", "nunique"))
    full = per_atco.index[per_atco["scenarios"] == 3].to_numpy()
    eval_atcos = np.sort(np.random.default_rng(SEED).choice(full, size=N_EVAL, replace=False))
    role = pd.Series("train", index=per_atco.index, name="role")
    role[full] = "stop"
    role[eval_atcos] = "eval"
    return per_atco.join(role).reset_index()[["atco", "role", "samples", "scenarios"]]


def load() -> pd.DataFrame:
    if not HOLDOUT_FILE.exists():
        raise FileNotFoundError(f"{HOLDOUT_FILE} missing - run `python -m harness.holdout`")
    return pd.read_csv(HOLDOUT_FILE)


if __name__ == "__main__":
    split = build()
    if HOLDOUT_FILE.exists():
        if not pd.read_csv(HOLDOUT_FILE).equals(split):
            raise SystemExit(f"{HOLDOUT_FILE} exists and differs from a fresh build - refusing to overwrite")
        print(f"{HOLDOUT_FILE} already frozen and reproducible")
    else:
        split.to_csv(HOLDOUT_FILE, index=False)
        print(f"wrote {HOLDOUT_FILE}")
    print(split.groupby("role").agg(atcos=("atco", lambda a: sorted(a.tolist())), samples=("samples", "sum"))
          .to_string())
