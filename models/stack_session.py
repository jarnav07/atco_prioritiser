"""Tier 3c: session-level stacking - one task is worked about once per session.

Two facts about the label sequence of a session (FINDINGS.md) that a per-snapshot ranker cannot see:
- the label changes across every gap between runs (adjacent runs share their label 0.2 % of the time);
- only ~3 % of runs carry a task that is also the label of another run of the same session.
A stage-1 ranker that is confident about task X in one run should therefore down-weight X in the other runs.
The b4 ranker (raw + f3) predicts the same task as the top pick of more than one run in 8 % of runs (truth 2.9 %).

Stage 1 = LightGBM lambdarank on `--features` (default raw+f3). Stage 2 = LightGBM lambdarank on the stage-1
scores turned into session context (per (run, task) mean probability, its rank in the run, the same task's best
run-mean in earlier / later runs of the session, how often it tops other runs, whether it tops the adjacent runs)
plus the stage-1 features.

Nested, leakage-free CV: inside each outer fold, the stage-2 training rows get stage-1 scores from an inner
leave-one-fold-out CV over the 4 training folds only, and the outer fold's rows get stage-1 scores from a model
fitted on those 4 folds. No label of the outer fold is ever seen by either stage. For the test file, stage 2 is
trained on the 5-fold OOF stage-1 scores of the labelled set and applied to test scores of a stage-1 fit on all
labelled rows. Stage-2 context features are computed within one session of the frame passed in (labels never
read), so they are computed the same way on test.

    python -m models.stack_session --features raw+f3 --run-id b7_stack_f3
"""
import argparse

import numpy as np
import pandas as pd

from features import FEATURE_SETS
from features.f3_temporal import sample_table
from harness import cv, folds as folds_mod, submission
from harness.config import ARTIFACTS, LABEL
from models import lgbm_ranker as lr
from models.postprocess import softmax_within_sample


def context_features(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Session context of stage-1 scores. Prefix s2_. Only scores, timestamps and task / callsign keys are read."""
    p = softmax_within_sample(df, score)
    s = sample_table(df).set_index("sample_id")
    run = df["sample_id"].map(s["run"]).to_numpy()
    sid = df["sample_id"]
    out = pd.DataFrame(index=df.index)
    out["s2_score"] = score
    out["s2_p"] = p
    out["s2_p_minus_max"] = p - pd.Series(p, index=df.index).groupby(sid).transform("max").to_numpy()
    out["s2_rank"] = pd.Series(score, index=df.index).groupby(sid).rank(ascending=False, method="min").to_numpy()

    # per (run, task): mean probability over the run's snapshots that list the task, and its rank in the run
    rt = pd.DataFrame({"run": run, "session": df["session"].to_numpy(), "task": df["task_name"].to_numpy(), "p": p})
    rt = rt.groupby(["run", "task"], sort=True).agg(session=("session", "first"), pm=("p", "mean"),
                                                    pmax=("p", "max")).reset_index()
    rt["rrank"] = rt.groupby("run")["pm"].rank(ascending=False, method="min")
    rt["top"] = (rt["rrank"] == 1).astype(float)
    rt["pm_minus_max"] = rt["pm"] - rt.groupby("run")["pm"].transform("max")
    # the same task in the other runs of its session (runs are numbered in time order)
    rt = rt.sort_values(["session", "task", "run"]).reset_index(drop=True)
    gk = rt.groupby(["session", "task"], sort=False)
    rt["pm_best_before"] = gk["pm"].transform(lambda x: x.shift(1).cummax())
    rt["pm_best_after"] = gk["pm"].transform(lambda x: x[::-1].shift(1).cummax()[::-1])
    rt["n_top_other"] = gk["top"].transform("sum") - rt["top"]
    rt["n_runs_listed"] = gk["pm"].transform("size")
    # adjacent runs (run +- 1 of the same session; NaN if the task is not listed there)
    by_rt = rt.set_index(["run", "task"])
    for tag, shift in (("prev", -1), ("next", 1)):
        idx = pd.MultiIndex.from_arrays([rt["run"] + shift, rt["task"]])
        adj = by_rt.reindex(idx)
        same_ses = adj["session"].to_numpy() == rt["session"].to_numpy()
        rt[f"pm_{tag}"] = np.where(same_ses, adj["pm"].to_numpy(), np.nan)
        rt[f"top_{tag}"] = np.where(same_ses, adj["top"].to_numpy(), np.nan)
    # aircraft level: best run-mean of any task of this aircraft in another run
    cs = rt["task"].str.extract(r"_([A-Z0-9]+)", expand=False)
    ac_best = rt.assign(cs=cs).groupby(["session", "cs", "run"])["pm"].max().rename("ac_pm").reset_index()
    ac_best["ac_top"] = ac_best["run"].map(rt.assign(cs=cs).loc[rt["top"] == 1].drop_duplicates("run").set_index("run")["cs"]).eq(
        ac_best["cs"]).astype(float)
    ac_tot = ac_best.groupby(["session", "cs"])["ac_top"].transform("sum")
    ac_best["ac_n_top_other"] = ac_tot - ac_best["ac_top"]
    rt = rt.assign(cs=cs).merge(ac_best[["session", "cs", "run", "ac_n_top_other"]], on=["session", "cs", "run"],
                                how="left")

    keys = pd.MultiIndex.from_arrays([run, df["task_name"].to_numpy()])
    cols = ["pm", "pmax", "rrank", "pm_minus_max", "pm_best_before", "pm_best_after", "n_top_other",
            "n_runs_listed", "pm_prev", "pm_next", "top_prev", "top_next", "ac_n_top_other"]
    back = rt.set_index(["run", "task"])[cols].reindex(keys)
    for c in cols:
        out[f"s2_{c}"] = back[c].to_numpy()
    out["s2_pm_over_best_other"] = out["s2_pm"] - np.fmax(out["s2_pm_best_before"], out["s2_pm_best_after"])
    return out


def expert_features(df: pd.DataFrame, score: np.ndarray, prefix: str) -> pd.DataFrame:
    """Score, within-sample probability / rank and run-mean probability of an extra stage-1 model."""
    p = softmax_within_sample(df, score)
    run = df["sample_id"].map(sample_table(df).set_index("sample_id")["run"]).to_numpy()
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}score"] = score
    out[f"{prefix}p"] = p
    out[f"{prefix}rank"] = pd.Series(score, index=df.index).groupby(df["sample_id"]).rank(
        ascending=False, method="min").to_numpy()
    out[f"{prefix}pm"] = pd.Series(p, index=df.index).groupby([run, df["task_name"].to_numpy()]).transform("mean")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="raw+f3", choices=sorted(FEATURE_SETS))
    ap.add_argument("--s2", default="full", choices=["full", "small"],
                    help="stage-2 inputs: context + stage-1 features (full) or context + task_index (small)")
    ap.add_argument("--rounds1", type=int, default=lr.NUM_ROUNDS, help="stage-1 boosting rounds")
    ap.add_argument("--preset1", default="base", choices=sorted(lr.PRESETS), help="stage-1 lambdarank setting")
    ap.add_argument("--rounds2", type=int, default=lr.NUM_ROUNDS, help="stage-2 boosting rounds")
    ap.add_argument("--preset2", default="base", choices=sorted(lr.PRESETS), help="stage-2 lambdarank setting")
    ap.add_argument("--expert", action="store_true", help="add the silent-target expert (needs f3 + f5)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    s1_args = dict(rounds=args.rounds1, preset=args.preset1)
    s2_args = dict(rounds=args.rounds2, preset=args.preset2)
    run_id = args.run_id or f"b7_stack_{args.features}_{args.s2}"

    from harness.data import load_labelled, load_test

    lab, test, folds = load_labelled(), load_test(), folds_mod.load()
    X_lab = lr.build_features(lab, args.features)
    X_test = lr.build_features(test, args.features)
    fold_of = lab["sample_id"].map(folds.set_index("sample_id")["fold"])
    print(f"[{run_id}] stage 1: {X_lab.shape[1]} features; stage 2: {args.s2}")

    # optional "silent expert": a second stage-1 ranker trained only on samples whose target is silent (its task
    # is still listed at the next run and its aircraft gets no input before then) - the hard 15 % of samples
    # where the main ranker is at 0.55. Applied to every row; its scores are extra stage-2 inputs.
    if args.expert:
        need = {"f3_in_next_run", "f5_n_act_to_next_run"}
        assert need <= set(X_lab.columns), f"--expert needs {need} (use a feature set with f3 and f5)"
        tgt = lab[LABEL].to_numpy() == 1
        silent_tgt = tgt & (X_lab["f3_in_next_run"].to_numpy() == 1) & (X_lab["f5_n_act_to_next_run"].to_numpy() == 0)
        silent_sample = lab["sample_id"].isin(set(lab.loc[silent_tgt, "sample_id"])).to_numpy()

    def stage1_fit(rows: pd.Index) -> dict:
        models = {"main": lr.fit(lab.loc[rows], X_lab.loc[rows], **s1_args)}
        if args.expert:
            r = rows[silent_sample[lab.index.get_indexer(rows)]]
            models["silent"] = lr.fit(lab.loc[r], X_lab.loc[r], **s1_args)
        return models

    def stage1_predict(models: dict, X: pd.DataFrame) -> dict:
        return {k: m.predict(X) for k, m in models.items()}

    def stage1_oof(rows: pd.Index) -> dict:
        """Leave-one-fold-out stage-1 scores over the folds present in `rows`."""
        oof = {}
        f = fold_of.loc[rows].to_numpy()
        for k in sorted(np.unique(f)):
            tr, va = rows[f != k], rows[f == k]
            for name, sc in stage1_predict(stage1_fit(tr), X_lab.loc[va]).items():
                oof.setdefault(name, pd.Series(np.nan, index=rows)).loc[va] = sc
        return {k: v.to_numpy() for k, v in oof.items()}

    def s2_inputs(df: pd.DataFrame, X: pd.DataFrame, s1: dict) -> pd.DataFrame:
        parts = [X if args.s2 == "full" else X[["task_index"]], context_features(df, s1["main"])]
        if "silent" in s1:
            parts.append(expert_features(df, s1["silent"], "s2e_"))
        return pd.concat(parts, axis=1)

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        s1_tr = stage1_oof(train.index)
        s1_va = stage1_predict(stage1_fit(train.index), X_lab.loc[valid.index])
        F_tr = s2_inputs(train, X_lab.loc[train.index], s1_tr)
        F_va = s2_inputs(valid, X_lab.loc[valid.index], s1_va)
        cv.check_features(F_tr.columns)
        return lr.fit(train, F_tr, **s2_args).predict(F_va)

    oof = cv.run_cv(lab, folds, fit_predict)
    rep = cv.report(lab, oof, folds)
    cv.log_experiment(run_id, f"session stacking: stage-1 lambdarank on {args.features}, stage-2 lambdarank on "
                              f"session context of stage-1 scores ({args.s2}); nested CV; stage 1 {s1_args}, stage 2 {s2_args}"
                              f"{'; + silent-target expert' if args.expert else ''}",
                      rep,
                      features=cv.features_hash(list(X_lab.columns) + ["s2"]))
    cv.save_artifact(run_id, "oof", lab, oof)

    # test: stage 2 trained on the 5-fold OOF stage-1 scores, applied to a stage-1 fit on all labelled rows
    s1_lab = stage1_oof(lab.index)
    s1_test = stage1_predict(stage1_fit(lab.index), X_test)
    F_lab, F_test = s2_inputs(lab, X_lab, s1_lab), s2_inputs(test, X_test, s1_test)
    booster = lr.fit(lab, F_lab, **s2_args)
    score = booster.predict(F_test)
    cv.save_artifact(run_id, "test", test, score)
    imp = pd.Series(booster.feature_importance("gain"), index=F_lab.columns).sort_values(ascending=False)
    imp.rename("gain").to_csv(ARTIFACTS / f"{run_id}_importance.csv")
    print("top features by gain:\n" + (imp.head(15) / imp.sum()).round(3).to_string())
    submission.write(test, score, run_id)


if __name__ == "__main__":
    main()
