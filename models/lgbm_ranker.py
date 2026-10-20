"""Tier-1 workhorse (b1): LightGBM lambdarank, group = sample_id, optimising NDCG@3.

    python -m models.lgbm_ranker                       # raw features, CV + full refit + submission
    python -m models.lgbm_ranker --features raw --run-id b1_lgbm_raw
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import CATEGORICAL, FEATURE_SETS
from harness import cv, folds as folds_mod, submission
from harness.config import ARTIFACTS, LABEL, SEED

PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[3],
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    lambda_l2=1.0,
    seed=SEED,
    deterministic=True,
    force_col_wise=True,
    num_threads=4,
    verbose=-1,
)
NUM_ROUNDS = 300


def build_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    X = pd.concat([b(df) for b in FEATURE_SETS[feature_set]], axis=1)
    cv.check_features(X.columns)
    return X


def _groups(df: pd.DataFrame) -> np.ndarray:
    # rows are sorted by (sample_id, row_index), so each sample is one contiguous query
    return df.groupby("sample_id", sort=False).size().to_numpy()


def fit(df: pd.DataFrame, X: pd.DataFrame) -> lgb.Booster:
    ds = lgb.Dataset(X, label=df[LABEL].to_numpy(), group=_groups(df), categorical_feature=CATEGORICAL)
    return lgb.train(PARAMS, ds, num_boost_round=NUM_ROUNDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="raw", choices=sorted(FEATURE_SETS))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    run_id = args.run_id or f"b1_lgbm_{args.features}"

    from harness.data import load_labelled, load_test

    lab, test, folds = load_labelled(), load_test(), folds_mod.load()
    X_lab = build_features(lab, args.features)
    X_test = build_features(test, args.features)
    print(f"[{run_id}] {X_lab.shape[1]} features, {lab['sample_id'].nunique()} samples")

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        return fit(train, X_lab.loc[train.index]).predict(X_lab.loc[valid.index])

    oof = cv.run_cv(lab, folds, fit_predict)
    rep = cv.report(lab, oof, folds)
    cv.log_experiment(run_id, f"LightGBM lambdarank, {NUM_ROUNDS} rounds, features={args.features}", rep,
                      features=cv.features_hash(X_lab.columns))
    cv.save_artifact(run_id, "oof", lab, oof)

    booster = fit(lab, X_lab)
    score = booster.predict(X_test)
    cv.save_artifact(run_id, "test", test, score)
    imp = pd.Series(booster.feature_importance("gain"), index=X_lab.columns).sort_values(ascending=False)
    imp.rename("gain").to_csv(ARTIFACTS / f"{run_id}_importance.csv")
    print("top features by gain:\n" + (imp.head(15) / imp.sum()).round(3).to_string())
    submission.write(test, score, run_id)


if __name__ == "__main__":
    main()
