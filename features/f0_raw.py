"""The 87 provided feature inputs, untouched. Missing stays NaN (empty means "does not apply", never 0)."""
import pandas as pd

from harness.data import raw_feature_cols


def build(df: pd.DataFrame) -> pd.DataFrame:
    return df[raw_feature_cols(df)]
