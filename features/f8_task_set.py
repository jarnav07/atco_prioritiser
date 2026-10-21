"""Tier 3e: changes to the aircraft's set of open tasks around the snapshot (look-ahead).

Silent tasks (worked without an input or a closure before the next run) are most of b8's misses. Their aircraft's
task list still tends to change soon after (a task added or removed within 120 s for 45 % of silent targets vs 24 %
of their rivals - evaluation-subagent prototype). Features (prefix f8_), per aircraft over its consecutive
appearances in the session: time to the next / since the last change of its task set, whether a change falls within
60 / 120 s, and the within-snapshot rank of the time to the next change.

Look-ahead: uses future snapshots of the same session. Never reads `target_index`.
"""
import numpy as np
import pandas as pd

from .f3_temporal import callsigns, sample_table


def build(df: pd.DataFrame) -> pd.DataFrame:
    s = sample_table(df).set_index("sample_id")
    g = df["sample_id"].map(s["g"]).to_numpy().astype(np.int64)
    t = df["t_s"].to_numpy()
    cs1, _ = callsigns(df)
    ac = pd.factorize(df["session"] + "|" + cs1.fillna("?"))[0].astype(np.int64)
    task = pd.factorize(df["session"] + "|" + df["task_name"])[0].astype(np.int64)

    rows = pd.DataFrame({"ac": ac, "g": g, "t": t, "task": task})
    sets = rows.groupby(["ac", "g"], sort=True).agg(tasks=("task", frozenset), t=("t", "first")).reset_index()
    prev = sets.groupby("ac")["tasks"].shift(1)
    changed = np.array([p is not None and p == p and a != p for a, p in zip(sets["tasks"], prev)])
    t_chg = sets["t"].where(changed)
    nxt = t_chg.groupby(sets["ac"]).shift(-1).groupby(sets["ac"]).bfill()   # first change after this appearance
    last = t_chg.groupby(sets["ac"]).ffill()                                 # last change at or before it
    tab = pd.DataFrame({"n": nxt.to_numpy(), "l": last.to_numpy()},
                       index=pd.MultiIndex.from_arrays([sets["ac"], sets["g"]]))
    at = tab.reindex(pd.MultiIndex.from_arrays([ac, g]))

    tt = at["n"].to_numpy() - t
    out = pd.DataFrame(index=df.index)
    out["f8_tt_set_change"] = tt
    out["f8_t_since_set_change"] = t - at["l"].to_numpy()
    out["f8_set_change_60"] = np.where(np.isnan(tt), 0.0, (tt <= 60).astype(float))
    out["f8_set_change_120"] = np.where(np.isnan(tt), 0.0, (tt <= 120).astype(float))
    out["f8_tt_set_change_rk"] = out["f8_tt_set_change"].groupby(df["sample_id"]).rank(method="min")
    return out
