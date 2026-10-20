"""Freeze the fold assignment: GroupKFold by controller over train + validation.

Run once (`python -m harness.folds`); every experiment reads harness/folds.csv. Never re-randomise.
"""
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import FOLDS_FILE, N_FOLDS
from .data import load_labelled, samples


def build() -> pd.DataFrame:
    s = samples(load_labelled())
    s["fold"] = -1
    # GroupKFold without shuffling is deterministic: it balances sample counts across folds
    for k, (_, idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(s, groups=s["atco"])):
        s.loc[idx, "fold"] = k
    return s[["sample_id", "session", "atco", "scenario", "episode", "fold"]]


def load() -> pd.DataFrame:
    if not FOLDS_FILE.exists():
        raise FileNotFoundError(f"{FOLDS_FILE} missing - run `python -m harness.folds`")
    return pd.read_csv(FOLDS_FILE)


if __name__ == "__main__":
    folds = build()
    if FOLDS_FILE.exists():
        old = pd.read_csv(FOLDS_FILE)
        if not old.equals(folds):
            raise SystemExit(f"{FOLDS_FILE} exists and differs from a fresh build - refusing to overwrite")
        print(f"{FOLDS_FILE} already frozen and reproducible")
    else:
        folds.to_csv(FOLDS_FILE, index=False)
        print(f"wrote {FOLDS_FILE}")
    summary = folds.groupby("fold").agg(samples=("sample_id", "size"), episodes=("episode", "nunique"),
                                        sessions=("session", "nunique"),
                                        atcos=("atco", lambda a: sorted(a.unique().tolist())))
    print(summary.to_string())
