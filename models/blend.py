"""Tier 5: equal-weight blend of runs on artifacts.

Each member's row scores are turned into a per-sample softmax probability (temperature 1 on the raw score,
which suits lambdarank and binary-logit scores alike) and averaged with equal weights - no weights are fitted,
so the blend's CV is not optimistically selected beyond the choice of members.

    python -m models.blend --runs b8_lgbm_f3f5 b8_binary_f3f5 --run-id b9_blend
"""
import argparse

import numpy as np

from harness import cv, folds as folds_mod, submission
from harness.data import load_labelled, load_test
from models.postprocess import softmax_within_sample


def blend(df, scores: list[np.ndarray]) -> np.ndarray:
    return np.mean([softmax_within_sample(df, s) for s in scores], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    lab, test, folds = load_labelled(), load_test(), folds_mod.load()
    oof = blend(lab, [cv.load_artifact(r, "oof", lab) for r in args.runs])
    print(f"[{args.run_id}] equal-weight softmax blend of {', '.join(args.runs)}")
    rep = cv.report(lab, oof, folds)
    cv.log_experiment(args.run_id, f"equal-weight per-sample softmax blend of {' + '.join(args.runs)}", rep)
    cv.save_artifact(args.run_id, "oof", lab, oof)
    score = blend(test, [cv.load_artifact(r, "test", test) for r in args.runs])
    cv.save_artifact(args.run_id, "test", test, score)
    submission.write(test, score, args.run_id)


if __name__ == "__main__":
    main()
