"""LightGBM lambdarank on causal features + the cross-controller consensus prior (features/f7_consensus.py).

The consensus features depend on other sessions' labels, so they are built inside each fold: held-out controllers
get them from the training-fold sessions; training rows from the other training-fold sessions (never their own);
test rows from all labelled sessions.

    python -m models.consensus_ranker --base raw --run-id c2_consensus_raw
"""
import argparse

import numpy as np
import pandas as pd

from features import FEATURE_SETS, f7_consensus as f7
from harness import cv, folds as folds_mod, submission
from harness.config import ARTIFACTS
from harness.data import load_labelled, load_test
from models.lgbm_ranker import NUM_ROUNDS, PRESETS, build_features, fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="raw", choices=sorted(FEATURE_SETS), help="causal base feature set")
    ap.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    ap.add_argument("--preset", default="base", choices=sorted(PRESETS))
    ap.add_argument("--inputs", action="store_true", help="add other controllers' HMI input timing (f7 build_inputs)")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    lab, test, folds = load_labelled(), load_test(), folds_mod.load()
    ref = f7.reference_arrivals(lab)
    u_lab, u_test = f7.scenario_clock(lab, ref), f7.scenario_clock(test, ref)
    X_lab, X_test = build_features(lab, args.base), build_features(test, args.base)
    print(f"[{args.run_id}] base={args.base} ({X_lab.shape[1]} cols) + f7 consensus; "
          f"scenario clock missing on {np.isnan(u_lab).mean():.2%} of labelled rows")
    pos_of = pd.Series(np.arange(len(lab)), index=lab.index)

    def consensus(query, u_q, pool, u_p) -> pd.DataFrame:
        parts = [f7.build(query, u_q, pool, u_p)]
        if args.inputs:
            parts.append(f7.build_inputs(query, u_q, f7.pool_inputs(pool, u_p)))
        return pd.concat(parts, axis=1)

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        tr, va = pos_of[train.index].to_numpy(), pos_of[valid.index].to_numpy()
        c_tr = consensus(train, u_lab[tr], train, u_lab[tr])
        c_va = consensus(valid, u_lab[va], train, u_lab[tr])
        Xtr = pd.concat([X_lab.loc[train.index], c_tr], axis=1)
        Xva = pd.concat([X_lab.loc[valid.index], c_va], axis=1)
        return fit(train, Xtr, rounds=args.rounds, preset=args.preset).predict(Xva)

    oof = cv.run_cv(lab, folds, fit_predict)
    rep = cv.report(lab, oof, folds)
    what = "f7 consensus + input timing" if args.inputs else "f7 consensus"
    cv.log_experiment(args.run_id, f"LightGBM lambdarank, base={args.base} + {what} (fold-aware), "
                      f"{args.rounds} rounds, preset={args.preset}", rep)
    cv.save_artifact(args.run_id, "oof", lab, oof)

    c_lab = consensus(lab, u_lab, lab, u_lab)
    c_test = consensus(test, u_test, lab, u_lab)
    X_full = pd.concat([X_lab, c_lab], axis=1)
    booster = fit(lab, X_full, rounds=args.rounds, preset=args.preset)
    score = booster.predict(pd.concat([X_test, c_test], axis=1))
    cv.save_artifact(args.run_id, "test", test, score)
    imp = pd.Series(booster.feature_importance("gain"), index=X_full.columns).sort_values(ascending=False)
    imp.rename("gain").to_csv(ARTIFACTS / f"{args.run_id}_importance.csv")
    print("top features by gain:\n" + (imp.head(12) / imp.sum()).round(3).to_string())
    submission.write(test, score, args.run_id)


if __name__ == "__main__":
    main()
