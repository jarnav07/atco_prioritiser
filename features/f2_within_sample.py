"""Tier 2: within-snapshot relational features, following the briefing deck's hints.

- "Aircraft vs task": most columns describe the aircraft, so compare each task with the other tasks of its own
  aircraft (task age is what separates them) and with the other tasks of its own type.
- "Assume vs entry coordination": explicit twin features (assume wins 73 % of twins).
- "Rank within the list": percentile ranks of every continuous column within the snapshot and within
  (snapshot, task type) - generalising the host's l_rank_*_type columns.
- Slide 25 (BIX72G transferred before the nearer EWG5C): the oldest task of a type is the answer 63 % of the
  time, so age relative to the oldest open task of the same type is spelt out.

Every value only looks at rows of the same sample_id. Callsigns are used as a grouping key, never as a value.
"""
import numpy as np
import pandas as pd

CONTINUOUS = [
    "distance_to_sector", "time_to_sector", "distance_to_MD", "time_to_MD", "current_FL", "exit_FL",
    "non_conformance_FL_delta", "non_conformance_agreed_route_deviation_fl",
    "non_conformance_agreed_route_deviation_horizontally", "distance_to_zone", "x_state_age", "x_task_age",
    "k_clearedFLDelta", "k_levelClearanceAgeS", "k_headingClearanceAgeS", "k_groundSpeed", "k_timeToExitPoint",
    "k_distanceToExitPoint", "k_outCoordRevision", "k_cfl_minus_fl", "a_since_flight",
]
SIGNED = ["distance_to_sector", "time_to_sector"]   # negative = inside the sector, distance to the exit
SIBLING_TYPES = {2: "entry_coord", 3: "exit_coord", 4: "assume", 5: "transfer", 10: "exit_cond", 12: "zone",
                 13: "rtr"}


def callsign(df: pd.DataFrame) -> pd.Series:
    """First callsign in task_name (conflict tasks name two aircraft; the row's features describe the first)."""
    return df["task_name"].str.extract(r"_([A-Z0-9]+)", expand=False)


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    sid = df["sample_id"]
    ac = callsign(df)
    typ = df["task_index"]
    base = df[CONTINUOUS].copy()
    for c in SIGNED:
        base[f"abs_{c}"] = df[c].abs()

    # percentile ranks within the snapshot and within (snapshot, type); NaN stays NaN
    r_all = base.groupby(sid).rank(method="min", pct=True)
    r_typ = base.groupby([sid, typ]).rank(method="min", pct=True)
    for c in base.columns:
        out[f"f2_{c}_pct"] = r_all[c]
        out[f"f2_{c}_tpct"] = r_typ[c]

    # distance from the extreme of the same type: 0 = oldest / nearest open task of this type
    age = df["x_task_age"]
    out["f2_type_age_minus_max"] = age - age.groupby([sid, typ]).transform("max")
    abs_t = base["abs_time_to_sector"]
    out["f2_type_abs_tts_minus_min"] = abs_t - abs_t.groupby([sid, typ]).transform("min")
    out["f2_a_since_flight_minus_min"] = df["a_since_flight"] - df["a_since_flight"].groupby(sid).transform("min")

    # the aircraft's own tasks: which is oldest, and what else is open on it
    out["f2_ac_age_rank"] = age.groupby([sid, ac]).rank(method="min", ascending=False)
    out["f2_ac_age_minus_max"] = age - age.groupby([sid, ac]).transform("max")
    held = pd.crosstab([sid, ac], typ).clip(upper=1)
    held_rows = held.reindex(pd.MultiIndex.from_arrays([sid, ac])).to_numpy()
    cols = list(held.columns)
    for t, name in SIBLING_TYPES.items():
        out[f"f2_ac_has_{name}"] = held_rows[:, cols.index(t)] if t in cols else np.zeros(len(df))

    # assume / entry-coordination twin: which side of the pair, and how much older than its sibling
    is_twin = df["l_twin_AE"].eq(1) & typ.isin([2, 4])
    twin_age = pd.DataFrame({"sid": sid, "ac": ac, "typ": typ, "age": age})[is_twin]
    pair = twin_age.pivot_table(index=["sid", "ac"], columns="typ", values="age", aggfunc="max")
    pair = pair.reindex(columns=[2, 4])
    keys = pd.MultiIndex.from_arrays([sid, ac])
    sibling_age = np.where(typ.eq(4), pair[2].reindex(keys).to_numpy(), pair[4].reindex(keys).to_numpy())
    out["f2_twin_is_assume"] = (is_twin & typ.eq(4)).astype(int)
    out["f2_twin_age_diff"] = np.where(is_twin, age.to_numpy() - sibling_age, np.nan)

    # snapshot size, for interactions only (constant within a sample)
    out["f2_n_aircraft"] = ac.groupby(sid).transform("nunique")
    out["f2_seq_len"] = df["seq_len"]

    return pd.DataFrame(out, index=df.index)
