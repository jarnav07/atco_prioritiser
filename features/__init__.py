"""Feature sets. A builder is a pure function build(df) -> DataFrame, index-aligned with df, that only looks
across rows inside one sample_id (or one session for temporal features). FEATURE_SETS maps a name to builders."""
from . import f0_raw, f2_within_sample, f3_temporal, f4_state_changes, f5_task_context

FEATURE_SETS = {
    "raw": [f0_raw.build],
    "raw+f2": [f0_raw.build, f2_within_sample.build],
    "raw+f3": [f0_raw.build, f3_temporal.build],
    "raw+f3+f4": [f0_raw.build, f3_temporal.build, f4_state_changes.build],
    "raw+f2+f3": [f0_raw.build, f2_within_sample.build, f3_temporal.build],
    "raw+f3+f5": [f0_raw.build, f3_temporal.build, f5_task_context.build],
}

CATEGORICAL = ["task_index"]
