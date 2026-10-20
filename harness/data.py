"""Loaders. Every frame comes back sorted by (sample_id, row_index) so each sample is a contiguous block."""
import pandas as pd

from .config import DATA, EPISODE_GAP_S, ID_COLS, LABEL


def read_split(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - unzip engage-ktn-2026-competition.zip into {DATA}/")
    df = pd.read_csv(path, dtype={"timestamp": str, "session": str, "task_name": str})
    df = df.sort_values(["sample_id", "row_index"], kind="stable").reset_index(drop=True)
    df["t_s"] = pd.to_timedelta(df["timestamp"]).dt.total_seconds()
    return df


def load_labelled() -> pd.DataFrame:
    """train.csv + validation.csv pooled (37 sessions, 18 controllers)."""
    df = pd.concat([read_split("train.csv"), read_split("validation.csv")], ignore_index=True)
    df = df.sort_values(["sample_id", "row_index"], kind="stable").reset_index(drop=True)
    per_sample = df.groupby("sample_id")[LABEL].sum()
    assert (per_sample == 1).all(), "every labelled sample must have exactly one positive row"
    return df


def load_test() -> pd.DataFrame:
    return read_split("test.csv")


def raw_feature_cols(df: pd.DataFrame) -> list[str]:
    """The 87 provided feature inputs: everything that is not an identifier, the label or a derived helper."""
    return [c for c in df.columns if c not in ID_COLS and c not in (LABEL, "t_s")]


def samples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample: sample_id, session, atco, scenario, t_s, episode."""
    s = df.drop_duplicates("sample_id")[["sample_id", "session", "atco", "scenario", "t_s"]].copy()
    s = s.sort_values(["session", "t_s", "sample_id"]).reset_index(drop=True)
    new_run = s["session"].ne(s["session"].shift()) | (s["t_s"].diff() > EPISODE_GAP_S)
    # an "episode" here is a run of back-to-back labelled snapshots; it is computable on test (timestamps only)
    s["episode"] = new_run.cumsum() - 1
    return s.sort_values("sample_id").reset_index(drop=True)
