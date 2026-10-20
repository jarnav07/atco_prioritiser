"""Feature sets. A builder is a pure function build(df) -> DataFrame, index-aligned with df, that only looks
across rows inside one sample_id (or one session for temporal features). FEATURE_SETS maps a name to builders."""
from . import f0_raw

FEATURE_SETS = {
    "raw": [f0_raw.build],
}

CATEGORICAL = ["task_index"]
