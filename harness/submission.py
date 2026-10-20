"""Submission writer + validator. Nothing is written to submissions/ without passing validate()."""
import pandas as pd

from .config import K, SUBMISSIONS
from .metric import _ids, top3_from_scores


class InvalidSubmission(Exception):
    pass


def validate(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(sub.columns) != ["sample_id", "prediction"]:
        raise InvalidSubmission(f"columns must be exactly sample_id,prediction - got {list(sub.columns)}")
    if sub["sample_id"].duplicated().any():
        raise InvalidSubmission("duplicate sample_id")
    expected = set(test["sample_id"].unique())
    got = set(sub["sample_id"])
    if got != expected:
        raise InvalidSubmission(f"{len(expected - got)} sample_ids missing, {len(got - expected)} unexpected")
    seq_len = test.drop_duplicates("sample_id").set_index("sample_id")["seq_len"]
    for sid, pred in zip(sub["sample_id"], sub["prediction"]):
        try:
            ids = _ids(pred)
        except ValueError:
            raise InvalidSubmission(f"sample {sid}: non-numeric id in {pred!r}") from None
        if len(ids) < min(K, seq_len[sid]):
            raise InvalidSubmission(f"sample {sid}: only {len(ids)} distinct ids")
        if any(i < 0 or i >= seq_len[sid] for i in ids):
            raise InvalidSubmission(f"sample {sid}: row_index out of range in {pred!r}")


def write(test: pd.DataFrame, score, run_id: str) -> pd.DataFrame:
    sub = top3_from_scores(test, score)
    validate(sub, test)
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / f"{run_id}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path} ({len(sub)} samples, validated)")
    return sub
