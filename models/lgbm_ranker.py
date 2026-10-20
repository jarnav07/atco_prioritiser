"""Tier-1 workhorse (b1): LightGBM ranker, group = sample_id.

    python -m models.lgbm_ranker                       # lambdarank, raw features, CV + full refit + submission
    python -m models.lgbm_ranker --features raw --run-id b1_lgbm_raw
    python -m models.lgbm_ranker --objective softmax --reg strong --rounds 1067   # cost function, models/cost.py
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import CATEGORICAL, FEATURE_SETS
from harness import cv, folds as folds_mod, submission
from harness.config import ARTIFACTS, LABEL, SEED
from models import cost

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

# lambdarank settings (named, so every run is reproducible from its CLI); chosen on the 5-fold CV
PRESETS = {
    "base": {},
    "l63": dict(num_leaves=63, min_data_in_leaf=50),
}

# same trees, but trained on the listwise softmax cross-entropy (models/cost.py) instead of lambdarank
SOFTMAX_PARAMS = {**{k: v for k, v in PARAMS.items() if k not in ("metric", "eval_at")},
                  "objective": cost.objective, "metric": "None"}

# regularisation settings for the cost-function model; the one with the lowest stop-set cost is selected
GRID = {
    "base":       {},
    "slow":       dict(learning_rate=0.02),
    "small":      dict(num_leaves=15, min_data_in_leaf=400),
    "small_slow": dict(num_leaves=15, min_data_in_leaf=400, learning_rate=0.02),
    "colsample":  dict(feature_fraction=0.5, lambda_l2=10.0),
    "strong":     dict(num_leaves=15, min_data_in_leaf=400, feature_fraction=0.5, lambda_l2=10.0,
                       learning_rate=0.02),
}


def build_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    X = pd.concat([b(df) for b in FEATURE_SETS[feature_set]], axis=1)
    cv.check_features(X.columns)
    return X


def _groups(df: pd.DataFrame) -> np.ndarray:
    # rows are sorted by (sample_id, row_index), so each sample is one contiguous query
    return df.groupby("sample_id", sort=False).size().to_numpy()


def fit(df: pd.DataFrame, X: pd.DataFrame, objective: str = "lambdarank", rounds: int = NUM_ROUNDS,
        reg: str = "base", preset: str = "base", seed: int = SEED) -> lgb.Booster:
    ds = lgb.Dataset(X, label=df[LABEL].to_numpy(), group=_groups(df), categorical_feature=CATEGORICAL)
    params = {**PARAMS, **PRESETS[preset]} if objective == "lambdarank" else {**SOFTMAX_PARAMS, **GRID[reg]}
    return lgb.train({**params, "seed": seed}, ds, num_boost_round=rounds)


class SeedBag:
    """Average of boosters that differ only in their seed (feature_fraction subsampling): damps the +-0.3 pt
    run-to-run noise that column subsampling alone causes on this CV."""

    def __init__(self, boosters: list):
        self.boosters = boosters

    def predict(self, X):
        return np.mean([b.predict(X) for b in self.boosters], axis=0)

    def feature_importance(self, kind: str = "gain"):
        return np.mean([b.feature_importance(kind) for b in self.boosters], axis=0)


def fit_bag(df: pd.DataFrame, X: pd.DataFrame, seeds: int = 1, **fit_args):
    """seeds == 1 is exactly fit(); otherwise a SeedBag of fits with seeds SEED, SEED + 1, ..."""
    if seeds == 1:
        return fit(df, X, **fit_args)
    boosters = [fit(df, X, seed=SEED + k, **fit_args) for k in range(seeds)]
    return SeedBag(boosters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="raw", choices=sorted(FEATURE_SETS))
    ap.add_argument("--objective", default="lambdarank", choices=["lambdarank", "softmax"])
    ap.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    ap.add_argument("--reg", default="base", choices=sorted(GRID), help="regularisation setting (softmax only)")
    ap.add_argument("--preset", default="base", choices=sorted(PRESETS), help="lambdarank setting")
    ap.add_argument("--seeds", type=int, default=1, help="seed bag size")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    run_id = args.run_id or f"b1_lgbm_{args.features}"
    fit_args = dict(objective=args.objective, rounds=args.rounds, reg=args.reg, preset=args.preset)

    from harness.data import load_labelled, load_test

    lab, test, folds = load_labelled(), load_test(), folds_mod.load()
    X_lab = build_features(lab, args.features)
    X_test = build_features(test, args.features)
    print(f"[{run_id}] {X_lab.shape[1]} features, {lab['sample_id'].nunique()} samples")

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        return fit_bag(train, X_lab.loc[train.index], args.seeds, **fit_args).predict(X_lab.loc[valid.index])

    oof = cv.run_cv(lab, folds, fit_predict)
    rep = cv.report(lab, oof, folds)
    desc = f"LightGBM {args.objective}, reg={args.reg}, {args.rounds} rounds, features={args.features}"
    if args.preset != "base":
        desc += f", preset={args.preset}"
    if args.seeds > 1:
        desc += f", seed bag of {args.seeds}"
    cv.log_experiment(run_id, desc, rep,
                      features=cv.features_hash(X_lab.columns))
    cv.save_artifact(run_id, "oof", lab, oof)

    booster = fit_bag(lab, X_lab, args.seeds, **fit_args)
    score = booster.predict(X_test)
    cv.save_artifact(run_id, "test", test, score)
    imp = pd.Series(booster.feature_importance("gain"), index=X_lab.columns).sort_values(ascending=False)
    imp.rename("gain").to_csv(ARTIFACTS / f"{run_id}_importance.csv")
    print("top features by gain:\n" + (imp.head(15) / imp.sum()).round(3).to_string())
    submission.write(test, score, run_id)


if __name__ == "__main__":
    main()
