"""CatBoost YetiRank ranker - a second model family for the causal blend (different trees, different errors).

    python -m models.catboost_ranker --features raw --run-id c5_catboost_raw
"""
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from features import FEATURE_SETS
from harness import cv, folds as folds_mod, submission
from harness.config import ARTIFACTS, LABEL, SEED
from harness.data import load_labelled, load_test
from models.lgbm_ranker import build_features

PARAMS = dict(loss_function="YetiRank", iterations=300, learning_rate=0.08, depth=6, random_seed=SEED,
              thread_count=4, verbose=False)


def _pool(df: pd.DataFrame, X: pd.DataFrame, labelled: bool = True) -> Pool:
    return Pool(X, label=df[LABEL].to_numpy() if labelled else None, group_id=df["sample_id"].to_numpy(),
                cat_features=["task_index"])


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(task_index=X["task_index"].astype(int))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="raw", choices=sorted(FEATURE_SETS))
    ap.add_argument("--iterations", type=int, default=PARAMS["iterations"])
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    params = {**PARAMS, "iterations": args.iterations}

    lab, test, folds = load_labelled(), load_test(), folds_mod.load()
    X_lab, X_test = _prep(build_features(lab, args.features)), _prep(build_features(test, args.features))
    print(f"[{args.run_id}] CatBoost YetiRank, {X_lab.shape[1]} features")

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        m = CatBoostRanker(**params).fit(_pool(train, X_lab.loc[train.index]))
        return m.predict(_pool(valid, X_lab.loc[valid.index], labelled=False))

    oof = cv.run_cv(lab, folds, fit_predict)
    rep = cv.report(lab, oof, folds)
    cv.log_experiment(args.run_id, f"CatBoost YetiRank, {args.iterations} its, depth 6, features={args.features}", rep,
                      features=cv.features_hash(X_lab.columns))
    cv.save_artifact(args.run_id, "oof", lab, oof)

    m = CatBoostRanker(**params).fit(_pool(lab, X_lab))
    score = m.predict(_pool(test, X_test, labelled=False))
    cv.save_artifact(args.run_id, "test", test, score)
    imp = pd.Series(m.get_feature_importance(), index=X_lab.columns).sort_values(ascending=False)
    imp.rename("importance").to_csv(ARTIFACTS / f"{args.run_id}_importance.csv")
    submission.write(test, score, args.run_id)


if __name__ == "__main__":
    main()
