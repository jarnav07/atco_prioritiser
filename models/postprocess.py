"""Tier-3 post-processing on artifacts: pool each task's probability over its label episode.

The files contain only labelled moments. Back-to-back snapshots (<= 4.5 s apart) of one session keep the same
label 94.6 % of the time, and the label changes after virtually every longer gap. So a run of back-to-back
snapshots is usually one episode with one answer, and averaging each task's per-sample softmax probability
over the run denoises the per-snapshot scores. Only timestamps are used - computable on test.

Measured (leave-controller-out, b1 OOF): +0.7 pt, inside the ~1 pt noise floor. Runs are not clean episodes:
13 % of them hold more than one label (the controller switched without a gap), and in 8 % of cases the true
task is missing from some snapshots of its run - so the mean is taken over the snapshots where the task is
present (treating absent as 0 cost 1.4 pt).

Caveat for deployment: live AWARE emits every 3 s whether or not a task is being worked, so the gap structure
(and the look-ahead within a run) is a property of this dataset, not of the real system.

    python -m models.postprocess --from b1_lgbm_raw
"""
import argparse

import numpy as np
import pandas as pd

from harness import cv, folds as folds_mod, submission
from harness.data import load_labelled, load_test, samples


def softmax_within_sample(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    s = pd.Series(score, index=df.index)
    z = np.exp(s - s.groupby(df["sample_id"]).transform("max"))
    return (z / z.groupby(df["sample_id"]).transform("sum")).to_numpy()


def episode_pool(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Mean probability of each (episode, task_name) over the snapshots of the episode where the task is listed."""
    ep = df["sample_id"].map(samples(df).set_index("sample_id")["episode"])
    p = pd.Series(softmax_within_sample(df, score), index=df.index)
    pooled = p.groupby([ep, df["task_name"]]).transform("mean")
    # the snapshot's own probability only breaks exact ties
    return (pooled + 1e-6 * p).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="run_id whose artifacts are pooled")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    run_id = args.run_id or f"{args.src}_epool"

    lab, test, folds = load_labelled(), load_test(), folds_mod.load()
    oof = episode_pool(lab, cv.load_artifact(args.src, "oof", lab))
    print(f"[{run_id}] episode pooling of {args.src}")
    rep = cv.report(lab, oof, folds)
    cv.log_experiment(run_id, f"{args.src} + per-(episode, task) mean softmax prob over back-to-back snapshots",
                      rep)
    cv.save_artifact(run_id, "oof", lab, oof)

    score = episode_pool(test, cv.load_artifact(args.src, "test", test))
    cv.save_artifact(run_id, "test", test, score)
    submission.write(test, score, run_id)


if __name__ == "__main__":
    main()
