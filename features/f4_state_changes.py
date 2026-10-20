"""Tier 3b: aircraft state changes around the snapshot - the *effect* of working a task.

f3 recovers HMI inputs from `a_since_flight`, but many tasks are finished by inputs the six `a_last_*` flags do not
name, or leave their trace in the aircraft state instead: a heading clearance locks a zone-conflict aircraft
(`k_isClearedHeadingAssigned`, `k_isLocked`), an exit-conditions or conflict task ends with a new cleared level
(`k_clearedFL`, `k_levelClearanceAgeS` reset), an exit coordination moves `k_outCoordRevision` /
`k_coordinationToolActivation`. Measured on targets whose task is still listed at the next run (the hard cases),
P(state changed before the next run | target) vs (| other task of the same type): zone heading/lock 0.60 vs
0.01-0.04, exit conditions cleared FL 0.64 vs 0.02, conflict cleared FL 0.51 vs 0.03, exit coordination tool
0.28 vs 0.08.

All state columns are aircraft-level (identical across an aircraft's tasks in a snapshot, checked), so the
aircraft's state is tracked over the snapshots that list it. Features (prefix f4_):
- per column: did it change between this snapshot and the aircraft's first appearance in the next run;
- per group of columns (clearance / coordination / conditions): time to the next and since the last change
  on this aircraft over its consecutive appearances, and whether the next change falls before the next run;
- how many columns change before the next run, and since the previous run.

Look-ahead: uses future snapshots of the same session (see f3 / FINDINGS.md). Never reads `target_index`.
"""
import numpy as np
import pandas as pd

from .f3_temporal import callsigns, sample_table

AGES = ["x_state_age", "k_levelClearanceAgeS", "k_headingClearanceAgeS"]   # a drop below elapsed time = reset
GROUPS = {
    "clr": ["k_clearedFL", "k_isClearedFLAssigned", "k_levelClearanceAgeS", "k_isClearedHeadingAssigned",
            "k_headingClearanceAgeS", "k_isLocked"],
    "coord": ["k_out_NOTIFIED", "k_out_COORDINATED", "k_outCoordRevision", "k_outCoordMismatch",
              "k_coordinationToolActivation", "x_isc_notified", "x_isc_coordinated", "x_transfer_in",
              "x_request_in", "x_state_age", "exit_FL"],
    "cond": ["k_exitConditionsAreMet", "k_aircraftNeedsToBeTransferred", "zone_conflict_ind", "conflict_ind",
             "monitoring_rule_activated", "qos_candidate", "within_sector", "x_in_sector_h", "is_assumed",
             "k_transferToolActivation"],
}
COLS = [c for cols in GROUPS.values() for c in cols]


def _changed(a: pd.DataFrame, b: pd.DataFrame, dt: np.ndarray) -> pd.DataFrame:
    """Column-wise change from state a (earlier) to state b (dt seconds later); NaN where b is unknown."""
    out = {}
    for c in COLS:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        if c in AGES:
            ch = (y < x + dt - 2) | (np.isnan(x) & ~np.isnan(y))
        else:
            ch = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        out[c] = ch.astype(float)
    return pd.DataFrame(out, index=a.index)


def build(df: pd.DataFrame) -> pd.DataFrame:
    s = sample_table(df)
    si = s.set_index("sample_id")
    g = df["sample_id"].map(si["g"]).to_numpy()
    t = df["t_s"].to_numpy()
    t_of_g = s["t_s"].to_numpy()
    cs1, _ = callsigns(df)
    ac = pd.factorize(df["session"] + "|" + cs1.fillna("?"))[0].astype(np.int64)

    # aircraft state per (g, aircraft): first row (state columns are aircraft-level)
    st = pd.DataFrame({"g": g, "ac": ac, "t": t}, index=df.index).join(df[COLS])
    st = st.drop_duplicates(["g", "ac"]).sort_values(["ac", "g"]).reset_index(drop=True)
    lookup = pd.MultiIndex.from_arrays([st["g"], st["ac"]])

    def state_at(g_other) -> pd.DataFrame:
        idx = pd.MultiIndex.from_arrays([np.nan_to_num(np.asarray(g_other, float), nan=-1).astype(np.int64), ac])
        return st.set_index(["g", "ac"]).reindex(idx)[COLS].reset_index(drop=True).set_axis(df.index)

    now = df[COLS]
    out = {}
    nxt_g = df["sample_id"].map(si["next_g0"]).to_numpy(dtype=float)
    nxt = state_at(nxt_g)
    dt_next = np.where(np.isnan(nxt_g), np.nan, t_of_g[np.nan_to_num(nxt_g, nan=0).astype(int)] - t)
    ch_next = _changed(now, nxt, dt_next)
    # unknown when the aircraft is not listed in the next run's first snapshot
    ac_listed = pd.MultiIndex.from_arrays([np.nan_to_num(nxt_g, nan=-1).astype(np.int64), ac]).isin(lookup)
    ch_next[~ac_listed] = np.nan
    for c in COLS:
        out[f"f4_chg_nr_{c}"] = ch_next[c].to_numpy()
    out["f4_n_chg_nr"] = ch_next.sum(axis=1, min_count=1).to_numpy()
    for grp, cols in GROUPS.items():
        out[f"f4_chg_nr_{grp}"] = ch_next[cols].max(axis=1).to_numpy()

    prv_g = df["sample_id"].map(si["prev_g1"]).to_numpy(dtype=float)
    dt_prev = np.where(np.isnan(prv_g), np.nan, t - t_of_g[np.nan_to_num(prv_g, nan=0).astype(int)])
    ch_prev = _changed(state_at(prv_g), now, dt_prev)
    ac_listed_p = pd.MultiIndex.from_arrays([np.nan_to_num(prv_g, nan=-1).astype(np.int64), ac]).isin(lookup)
    ch_prev[~ac_listed_p] = np.nan
    out["f4_n_chg_pr"] = ch_prev.sum(axis=1, min_count=1).to_numpy()
    for grp, cols in GROUPS.items():
        out[f"f4_chg_pr_{grp}"] = ch_prev[cols].max(axis=1).to_numpy()

    # change events along each aircraft's consecutive appearances: time to next / since last change, per group
    prev_state = st.groupby("ac")[COLS].shift(1)
    dt_app = (st["t"] - st.groupby("ac")["t"].shift(1)).to_numpy()
    ch_app = _changed(prev_state, st[COLS], dt_app)
    first = st["ac"].ne(st["ac"].shift(1)).to_numpy()
    ch_app[first] = 0.0
    key_rows = pd.MultiIndex.from_arrays([g.astype(np.int64), ac])
    next_g0_of_row = nxt_g
    for grp, cols in GROUPS.items():
        tchg = st["t"].where(ch_app[cols].max(axis=1).to_numpy() > 0)
        nxt_t = tchg.groupby(st["ac"]).shift(-1)
        nxt_t = nxt_t.groupby(st["ac"]).bfill()
        last_t = tchg.groupby(st["ac"]).ffill()
        tab = pd.DataFrame({"n": nxt_t.to_numpy(), "l": last_t.to_numpy()}, index=lookup).reindex(key_rows)
        tt_next = tab["n"].to_numpy() - t
        out[f"f4_tt_chg_{grp}"] = tt_next
        out[f"f4_t_since_chg_{grp}"] = t - tab["l"].to_numpy()
        t_nr = np.where(np.isnan(next_g0_of_row), np.inf, t_of_g[np.nan_to_num(next_g0_of_row, nan=0).astype(int)])
        out[f"f4_chg_{grp}_before_nr"] = np.where(np.isnan(tt_next), np.nan, (t + tt_next <= t_nr).astype(float))

    # per column: time to the next change at any later appearance (not only before the next run). Silent
    # coordination work shows up late: e.g. monitoring_rule_activated changes later for 46 % of silent
    # entry-coordination targets vs 9 % of the other entry-coordination tasks of their snapshot.
    for c in COLS:
        tchg = st["t"].where(ch_app[c].to_numpy() > 0)
        nxt_t = tchg.groupby(st["ac"]).shift(-1).groupby(st["ac"]).bfill()
        out[f"f4_tt_{c}"] = pd.Series(nxt_t.to_numpy(), index=lookup).reindex(key_rows).to_numpy() - t

    res = pd.DataFrame(out, index=df.index)
    sid = df["sample_id"]
    for c in ["f4_tt_chg_clr", "f4_tt_chg_coord", "f4_tt_chg_cond"]:
        res[f"{c}_rk"] = res[c].groupby(sid).rank(method="min")
    return res
