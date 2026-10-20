"""Tier-1 heuristic (b0): P(target | task_index) from the training rows, tie-break nearest the sector boundary.

With the prior taken from train.csv alone this reproduces sample_submission.csv exactly - main() asserts it,
which proves loader, top-3 assembly and submission writer end to end.

    python -m models.heuristics
"""
import numpy as np
import pandas as pd

from harness import cv, folds as folds_mod, submission
from harness.config import DATA, LABEL
from harness.data import load_labelled, load_test, read_split

RUN_ID = "b0_type_prior"


def type_prior(train: pd.DataFrame) -> pd.Series:
    """Per-row hit rate of each task type: positives / rows."""
    return train.groupby("task_index")[LABEL].mean()


def fit_predict(train: pd.DataFrame, apply: pd.DataFrame) -> np.ndarray:
    prior = apply["task_index"].map(type_prior(train)).fillna(0.0)
    prior_rank = prior.rank(method="dense").to_numpy()
    # lexicographic: prior first, then smallest |time_to_sector| (capped at 3600, so it never crosses a prior step)
    return prior_rank * 1e4 - apply["time_to_sector"].abs().to_numpy()


def main() -> None:
    lab, test, folds = load_labelled(), load_test(), folds_mod.load()

    ref = pd.read_csv(DATA / "sample_submission.csv", dtype={"prediction": str})
    mine = submission.top3_from_scores(test, fit_predict(read_split("train.csv"), test))
    same = (mine.set_index("sample_id")["prediction"] == ref.set_index("sample_id")["prediction"]).mean()
    print(f"agreement with sample_submission.csv: {same:.4f}")
    assert same == 1.0, "harness does not reproduce the sample submission"

    print(f"[{RUN_ID}] leave-controller-out CV")
    oof = cv.run_cv(lab, folds, fit_predict)
    rep = cv.report(lab, oof, folds)
    cv.log_experiment(RUN_ID, "type prior (rows rate) + nearest |time_to_sector| = sample_submission logic", rep)
    cv.save_artifact(RUN_ID, "oof", lab, oof)

    score = fit_predict(lab, test)
    cv.save_artifact(RUN_ID, "test", test, score)
    submission.write(test, score, RUN_ID)


if __name__ == "__main__":
    main()
