"""Train on the `train` controllers, early-stop on the `stop` controllers, score once on the `eval` controllers.

    python -m models.holdout_eval                     # every objective x feature set below
    python -m models.holdout_eval --objective softmax --features raw

Objectives:
- lambdarank : the b1 setup, fixed 300 rounds (no early stopping) - the reference.
- softmax    : the cost function in models/cost.py; rounds chosen where the stop-set cost is lowest.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import CATEGORICAL, FEATURE_SETS
from harness import cv, holdout
from harness.data import load_labelled, samples
from harness.metric import hits_from_scores
from models import cost
from models.lgbm_ranker import GRID, NUM_ROUNDS, PARAMS, SOFTMAX_PARAMS, _groups, build_features

MAX_ROUNDS = 4000
PATIENCE = 100


def _dataset(df: pd.DataFrame, X: pd.DataFrame, reference=None) -> lgb.Dataset:
    return lgb.Dataset(X, label=df["target_index"].to_numpy(), group=_groups(df),
                       categorical_feature=CATEGORICAL, reference=reference)


def run(lab: pd.DataFrame, X: pd.DataFrame, role: pd.Series, objective: str, overrides: dict | None = None,
        verbose: bool = True) -> tuple[lgb.Booster, int, float]:
    """Returns (booster, rounds, stop-set cost at those rounds)."""
    tr, st = role.eq("train").to_numpy(), role.eq("stop").to_numpy()
    d_tr = _dataset(lab[tr], X[tr])
    if objective == "lambdarank":
        return lgb.train(PARAMS, d_tr, num_boost_round=NUM_ROUNDS), NUM_ROUNDS, float("nan")
    d_st = _dataset(lab[st], X[st], reference=d_tr)
    curve = {}
    booster = lgb.train(
        {**SOFTMAX_PARAMS, **(overrides or {})}, d_tr, num_boost_round=MAX_ROUNDS, valid_sets=[d_st],
        valid_names=["stop"], feval=[cost.feval_cost, cost.feval_top3],
        callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False),
                   lgb.record_evaluation(curve)])
    best = booster.best_iteration
    ce, hit = curve["stop"]["softmax_ce"], curve["stop"]["top3_hit"]
    if verbose:
        for it in sorted({50, 100, 200, 400, 800, best}):
            if it <= len(ce):
                print(f"    round {it:4d}: stop cost {ce[it - 1]:.4f}  stop top-3 {hit[it - 1]:.4f}"
                      + ("  <- best (lowest cost)" if it == best else ""))
    return booster, best, ce[best - 1]


def tune(lab: pd.DataFrame, role: pd.Series, feature_sets: list[str]) -> None:
    """Select (feature set, regularisation) by the stop-set cost; score only the winner on eval."""
    results = []
    X = {fs: build_features(lab, fs) for fs in feature_sets}
    for fs in feature_sets:
        for name, over in GRID.items():
            booster, rounds, stop_ce = run(lab, X[fs], role, "softmax", over, verbose=False)
            st = role.eq("stop").to_numpy()
            stop_hit = hits_from_scores(lab[st], booster.predict(X[fs][st], num_iteration=rounds)).mean()
            results.append(dict(features=fs, grid=name, rounds=rounds, stop_cost=round(stop_ce, 4),
                                stop_top3=round(stop_hit, 4)))
            print(results[-1], flush=True)
    table = pd.DataFrame(results).sort_values("stop_cost")
    print(table.to_string(index=False))
    win = table.iloc[0]
    print(f"selected by lowest stop-set cost: features={win.features} grid={win.grid} rounds={win.rounds}")
    evaluate(lab, X[win.features], role, "softmax", win.features, GRID[win.grid],
             run_id=f"ho_softmax_tuned_{win.features.replace('+', '_')}", tag=f"grid={win.grid}")


def evaluate(lab, X, role, obj, fs, overrides=None, run_id=None, tag="") -> dict:
    ev = role.eq("eval").to_numpy()
    eval_rows = lab[ev]
    run_id = run_id or f"ho_{obj}_{fs.replace('+', '_')}"
    print(f"[{run_id}]")
    booster, rounds, _ = run(lab, X, role, obj, overrides)
    score = booster.predict(X[ev], num_iteration=rounds)
    hits = hits_from_scores(eval_rows, score)
    atco_of = eval_rows.drop_duplicates("sample_id").set_index("sample_id")["atco"]
    per_atco = hits.groupby(atco_of.loc[hits.index].to_numpy()).mean()
    p, n_ep = hits.mean(), samples(eval_rows)["episode"].nunique()
    rep = dict(cv_mean=per_atco.mean(), cv_std=per_atco.std(ddof=1), pooled=p, se_episodes=np.sqrt(p * (1 - p) / n_ep),
               per_fold=" ".join(f"{a}:{h:.4f}" for a, h in per_atco.items()))
    ce = cost.cost_value(score, eval_rows["target_index"].to_numpy(), _groups(eval_rows))
    print(f"  eval top-3 {p:.4f} ± {rep['se_episodes']:.4f} (episode SE) | per controller {rep['per_fold']}"
          f" | eval cost {ce:.4f} | rounds {rounds}")
    how = " (early-stopped on stop-set cost)" if obj == "softmax" else " (fixed)"
    cv.log_experiment(run_id, f"holdout: {obj}, features={fs}, rounds={rounds}{how} {tag}".strip(), rep,
                      features=cv.features_hash(X.columns), split="holdout")
    return dict(run_id=run_id, eval_top3=round(p, 4), eval_cost=round(ce, 4), rounds=rounds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["lambdarank", "softmax"], nargs="*", default=["lambdarank", "softmax"])
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), nargs="*", default=["raw", "raw+f2"])
    ap.add_argument("--tune", action="store_true", help="select regularisation by stop-set cost, then eval once")
    args = ap.parse_args()

    lab = load_labelled()
    role = lab["atco"].map(holdout.load().set_index("atco")["role"])
    print("eval controllers", sorted(lab.loc[role.eq("eval"), "atco"].unique()),
          "| stop controllers", sorted(lab.loc[role.eq("stop"), "atco"].unique()))
    if args.tune:
        tune(lab, role, args.features)
        return

    rows = []
    for fs in args.features:
        X = build_features(lab, fs)
        for obj in args.objective:
            rows.append(evaluate(lab, X, role, obj, fs))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
