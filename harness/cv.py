"""Leave-controller-out CV, reporting, experiment log and artifacts. Every experiment reports through here."""
import hashlib
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from .config import ARTIFACTS, EXPERIMENTS_FILE, FORBIDDEN_FEATURES, SEED
from .metric import hits_from_scores

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def check_features(cols) -> None:
    leaked = FORBIDDEN_FEATURES & set(cols)
    if leaked:
        raise ValueError(f"forbidden columns used as features: {sorted(leaked)}")


def run_cv(df: pd.DataFrame, folds: pd.DataFrame, fit_predict: FitPredict) -> np.ndarray:
    """Out-of-fold row scores. fit_predict(train_rows, valid_rows) -> scores for valid_rows."""
    fold_of = df["sample_id"].map(folds.set_index("sample_id")["fold"]).to_numpy()
    assert not np.isnan(fold_of).any(), "sample without a fold"
    oof = np.full(len(df), np.nan)
    for k in np.unique(fold_of):
        va = fold_of == k
        oof[va] = fit_predict(df[~va], df[va])
        print(f"  fold {int(k)}: {hits_from_scores(df[va], oof[va]).mean():.4f}", flush=True)
    return oof


def report(df: pd.DataFrame, score: np.ndarray, folds: pd.DataFrame) -> dict:
    hits = hits_from_scores(df, score)
    f = folds.set_index("sample_id").loc[hits.index]
    rows = []
    for k, part in f.groupby("fold"):
        h = hits.loc[part.index]
        n_ep = part["episode"].nunique()
        p = h.mean()
        rows.append(dict(fold=k, atcos=" ".join(map(str, sorted(part["atco"].unique()))), samples=len(h),
                         episodes=n_ep, hit=p, se_episodes=np.sqrt(p * (1 - p) / n_ep)))
    table = pd.DataFrame(rows)
    p_all = hits.mean()
    rep = dict(cv_mean=table["hit"].mean(), cv_std=table["hit"].std(ddof=1), pooled=p_all,
               se_episodes=np.sqrt(p_all * (1 - p_all) / f["episode"].nunique()),
               per_fold=" ".join(f"{h:.4f}" for h in table["hit"]))
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"CV mean {rep['cv_mean']:.4f} ± {rep['cv_std']:.4f} (fold std) | pooled {p_all:.4f} "
          f"± {rep['se_episodes']:.4f} (episode SE) | mean-0.5·std {rep['cv_mean'] - 0.5 * rep['cv_std']:.4f}")
    return rep


def features_hash(cols) -> str:
    return hashlib.md5(",".join(sorted(cols)).encode()).hexdigest()[:8]


def log_experiment(run_id: str, desc: str, rep: dict, features: str = "", seed: int = SEED) -> None:
    row = dict(run_id=run_id, desc=desc, cv_mean=round(rep["cv_mean"], 4), cv_std=round(rep["cv_std"], 4),
               pooled=round(rep["pooled"], 4), se_episodes=round(rep["se_episodes"], 4), per_fold=rep["per_fold"],
               public_lb="", features_hash=features, seed=seed,
               utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    log = pd.read_csv(EXPERIMENTS_FILE, dtype=str) if EXPERIMENTS_FILE.exists() else pd.DataFrame(columns=list(row))
    if run_id in set(log["run_id"]):
        # a rerun replaces its row but keeps a public_lb that was filled in by hand
        row["public_lb"] = log.loc[log["run_id"] == run_id, "public_lb"].fillna("").iloc[0]
        log = log[log["run_id"] != run_id]
    log = pd.concat([log, pd.DataFrame([row]).astype(str)], ignore_index=True)
    log.to_csv(EXPERIMENTS_FILE, index=False)


def save_artifact(run_id: str, kind: str, df: pd.DataFrame, score: np.ndarray) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    out = df[["sample_id", "row_index"]].assign(score=np.asarray(score, dtype=float))
    out.to_parquet(ARTIFACTS / f"{run_id}_{kind}.parquet", index=False)


def load_artifact(run_id: str, kind: str, df: pd.DataFrame) -> np.ndarray:
    art = pd.read_parquet(ARTIFACTS / f"{run_id}_{kind}.parquet")
    merged = df[["sample_id", "row_index"]].merge(art, on=["sample_id", "row_index"], how="left")
    assert merged["score"].notna().all(), f"artifact {run_id}_{kind} does not cover every row"
    return merged["score"].to_numpy()
