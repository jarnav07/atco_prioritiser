"""Feature sets. A builder is a pure function build(df) -> DataFrame, index-aligned with df, that only looks
across rows inside one sample_id (or one session for temporal features). FEATURE_SETS maps a name to builders."""
from . import f0_raw, f2_within_sample, f3_temporal, f4_state_changes, f5_task_context, f6_causal

FEATURE_SETS = {
    "raw": [f0_raw.build],
    "raw+f2": [f0_raw.build, f2_within_sample.build],
    "raw+f3": [f0_raw.build, f3_temporal.build],
    "raw+f3+f4": [f0_raw.build, f3_temporal.build, f4_state_changes.build],
    "raw+f2+f3": [f0_raw.build, f2_within_sample.build, f3_temporal.build],
    "raw+f3+f5": [f0_raw.build, f3_temporal.build, f5_task_context.build],
    "raw+f3+f4+f5": [f0_raw.build, f3_temporal.build, f4_state_changes.build, f5_task_context.build],
    # no look-ahead: only the current and earlier snapshots (for the deployment question)
    "raw+f3past": [f0_raw.build, f3_temporal.build_past],
    # causal (current + earlier snapshots only; checked by `python -m harness.causal <set>`)
    "raw+f6": [f0_raw.build, f6_causal.build],
    "raw+f3past+f6": [f0_raw.build, f3_temporal.build_past, f6_causal.build],
}

CATEGORICAL = ["task_index"]
