"""Top-3 hit rate. Mirrors data/scorer.py exactly (checked by `python -m harness.metric`)."""
import numpy as np
import pandas as pd

from .config import K, LABEL


def _ids(value) -> list[int]:
    """space-separated ids -> ints, duplicates removed, order kept (same parsing as the official scorer)"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out = []
    for tok in str(value).split():
        i = int(float(tok))
        if i not in out:
            out.append(i)
    return out


def top3_hit(labels_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """labels_df: rows with sample_id, row_index, target_index. pred_df: sample_id, prediction ("12 3 7")."""
    truth = labels_df.loc[labels_df[LABEL] == 1].set_index("sample_id")["row_index"]
    pred = pred_df.set_index("sample_id")["prediction"].reindex(truth.index)
    return float(np.mean([int(t in _ids(p)[:K]) for t, p in zip(truth.values, pred.values)]))


def positions(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """0-based position of every row within its sample: score descending, ties -> lower row_index first."""
    order = np.lexsort((df["row_index"].to_numpy(), -np.asarray(score, dtype=float), df["sample_id"].to_numpy()))
    pos = np.empty(len(df), dtype=np.int64)
    sid = df["sample_id"].to_numpy()[order]
    start = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    run = np.diff(np.r_[start, len(sid)])
    pos[order] = np.arange(len(sid)) - np.repeat(start, run)
    return pos


def hits_from_scores(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """Per-sample 0/1 hit, indexed by sample_id. df must carry target_index."""
    pos = positions(df, score)
    is_pos = df[LABEL].to_numpy() == 1
    return pd.Series((pos[is_pos] < K).astype(int), index=df["sample_id"].to_numpy()[is_pos], name="hit")


def top3_from_scores(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """sample_id, prediction - the submission frame for any row-level score."""
    pos = positions(df, score)
    top = df.loc[pos < K, ["sample_id", "row_index"]].assign(pos=pos[pos < K]).sort_values(["sample_id", "pos"])
    pred = top.groupby("sample_id", sort=True)["row_index"].agg(lambda r: " ".join(map(str, r)))
    return pred.rename("prediction").reset_index()


if __name__ == "__main__":
    # self-check against the official scorer on validation.csv with a deliberately mediocre score
    import importlib.util

    from .config import DATA
    from .data import load_labelled

    spec = importlib.util.spec_from_file_location("official_scorer", DATA / "scorer.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    lab = load_labelled()
    score = -lab["time_to_sector"].abs().to_numpy()
    sub = top3_from_scores(lab, score)
    sol = lab.loc[lab[LABEL] == 1, ["sample_id", "row_index"]].rename(columns={"row_index": "target"})
    ours_sub, ours_rows = top3_hit(lab, sub), hits_from_scores(lab, score).mean()
    theirs = official.score(sol, sub, "sample_id")
    print(f"official {theirs:.6f}  ours(sub) {ours_sub:.6f}  ours(rows) {ours_rows:.6f}")
    assert abs(theirs - ours_sub) < 1e-12 and abs(theirs - ours_rows) < 1e-12
    print("metric matches data/scorer.py")
