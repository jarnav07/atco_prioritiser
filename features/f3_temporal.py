"""Tier 3: session-temporal context - what happens to each task and each aircraft before and after this snapshot.

The files hold only labelled moments: a sample exists while the controller works a task, so a session is a
sequence of *runs* of back-to-back snapshots (<= 4.5 s apart) separated by gaps. Two consequences make the
session timeline informative about the label:

- A worked task is usually *finished*: an assumed aircraft's assume task closes, a transferred aircraft's
  transfer task closes. So the target task tends to vanish from the list at the end of its run
  (target absent from the next run's first snapshot 62 % vs 7 % for any task).
- Working a task usually means an HMI input on its aircraft, which appears in *later* snapshots as a reset of
  `a_since_flight` (and the `a_last_*` flags say which input). Every snapshot therefore reveals the time of the
  last input on each listed aircraft; collected over the session they give the full input log per aircraft,
  and for each row the time to the *next* input on its aircraft (soonest of the snapshot for ~50 % of targets).

Feature families (prefix f3_):
- run geometry: position in the run, run length, time to run end / from run start, gaps to neighbours;
- task presence: in previous / next snapshot, in the last snapshot of the previous run / first snapshot of the
  next run, time until the task's presence streak ends / it vanishes, first / last time seen in the session,
  share of the run's snapshots that list it;
- aircraft input log: time to the next / since the previous HMI input on this aircraft (both callsigns of a
  conflict task), which input it is, whether it falls inside this run, number of inputs in the next 60 s;
- within-snapshot ranks of the key temporal values (soonest next input, soonest vanish).

**Look-ahead.** Most of these use *future* snapshots of the same session. That is legal here (the whole test
file is given and no label is read), but a live AWARE system would not have them - see FINDINGS.md.

Leakage audit: only `session`, `t_s` (timestamps), `task_name` (as a key), `a_since_flight` and `a_last_*` are
read; `target_index` never is. Everything is computed within one session of the frame passed in, so building
on the labelled file and on the test file separately gives identical semantics.
"""
import numpy as np
import pandas as pd

from harness.config import EPISODE_GAP_S

ACTIONS = ["a_last_assume_control", "a_last_force_assume_control", "a_last_assign_level_clearance",
           "a_last_assign_lateral_clearance", "a_last_decontrol", "a_last_initiate_transfer_of_control"]
NO_INPUT = 3600  # a_since_flight sentinel: no input in the last hour


def callsigns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """First and (conflict tasks only) second callsign of task_name `<type>_<CS1>[_<CS2>]`."""
    parts = df["task_name"].str.extract(r"_([A-Z0-9]+)(?:_([A-Z0-9]+))?$")
    return parts[0], parts[1]


def sample_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample in session-time order with run geometry and neighbour pointers (positions g)."""
    s = df.drop_duplicates("sample_id")[["sample_id", "session", "t_s"]]
    s = s.sort_values(["session", "t_s", "sample_id"]).reset_index(drop=True)
    s["g"] = np.arange(len(s))
    new_ses = s["session"].ne(s["session"].shift())
    gap_prev = s["t_s"].diff().where(~new_ses)
    s["gap_prev"] = gap_prev
    s["gap_next"] = gap_prev.shift(-1).where(~new_ses.shift(-1, fill_value=True))
    s["run"] = (new_ses | (gap_prev > EPISODE_GAP_S)).cumsum() - 1
    r = s.groupby("run")
    s["run_pos"] = r.cumcount()
    s["run_len"] = r["g"].transform("size")
    s["run_t0"] = r["t_s"].transform("min")
    s["run_t1"] = r["t_s"].transform("max")
    s["run_g0"] = r["g"].transform("min")
    s["run_g1"] = r["g"].transform("max")
    # pointers to the neighbouring runs (same session), -1 if none
    runs = s.groupby("run").agg(session=("session", "first"), g0=("g", "min"), g1=("g", "max"), t0=("t_s", "min"),
                                t1=("t_s", "max"))
    same_next = runs["session"].eq(runs["session"].shift(-1))
    same_prev = runs["session"].eq(runs["session"].shift(1))
    runs["next_g0"] = runs["g0"].shift(-1).where(same_next)
    runs["next_t0"] = runs["t0"].shift(-1).where(same_next)
    runs["prev_g1"] = runs["g1"].shift(1).where(same_prev)
    runs["prev_t1"] = runs["t1"].shift(1).where(same_prev)
    s = s.join(runs[["next_g0", "next_t0", "prev_g1", "prev_t1"]], on="run")
    s["ses_t0"] = s.groupby("session")["t_s"].transform("min")
    return s


def input_events(df: pd.DataFrame, cs: pd.Series) -> pd.DataFrame:
    """Every HMI input any snapshot reveals: (session, cs, act_t, code). code 1..6 = ACTIONS order, 0 = other."""
    seen = df["a_since_flight"] < NO_INPUT
    ev = pd.DataFrame({"session": df["session"], "cs": cs, "act_t": (df["t_s"] - df["a_since_flight"]).round(),
                       "code": (df[ACTIONS].fillna(0).to_numpy() * np.arange(1, 7)).max(axis=1)})[seen]
    return ev.drop_duplicates(["session", "cs", "act_t"]).sort_values("act_t").reset_index(drop=True)


def _asof(keys: pd.DataFrame, ev: pd.DataFrame, direction: str, exact: bool) -> pd.DataFrame:
    """For each (session, cs, t_s) key: the nearest input event before/after t_s on that aircraft."""
    q = keys.assign(_i=np.arange(len(keys))).dropna(subset=["cs"]).sort_values("t_s")
    m = pd.merge_asof(q, ev, left_on="t_s", right_on="act_t", by=["session", "cs"], direction=direction,
                      allow_exact_matches=exact)
    return m.set_index("_i").reindex(np.arange(len(keys)))[["act_t", "code"]]


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    s = sample_table(df)
    si = s.set_index("sample_id")
    row = si.loc[df["sample_id"]].reset_index(drop=True)
    row.index = df.index
    t = df["t_s"]
    g = row["g"].to_numpy()

    # run geometry (sample-level; only useful in interaction with task-level values)
    out["f3_run_pos"] = row["run_pos"]
    out["f3_run_len"] = row["run_len"]
    out["f3_run_t_to_end"] = row["run_t1"] - t
    out["f3_run_t_from_start"] = t - row["run_t0"]
    out["f3_gap_prev"] = row["gap_prev"]
    out["f3_gap_next"] = row["gap_next"]
    out["f3_gap_next_run"] = row["next_t0"] - row["run_t1"]
    out["f3_ses_t"] = t - row["ses_t0"]

    # task presence across the session's snapshots: key = (g, task code) within one session
    task_code = pd.factorize(df["session"] + "|" + df["task_name"])[0].astype(np.int64)
    n_codes = task_code.max() + 1
    present = np.unique(g.astype(np.int64) * n_codes + task_code)

    def present_at(g_other) -> np.ndarray:
        """1/0: is this row's task listed in snapshot g_other (NaN where there is no such snapshot)."""
        g_other = np.asarray(g_other, dtype=float)
        ok = ~np.isnan(g_other)
        res = np.full(len(df), np.nan)
        res[ok] = np.isin(g_other[ok].astype(np.int64) * n_codes + task_code[ok], present)
        return res

    out["f3_in_prev"] = np.where(row["gap_prev"].notna(), present_at(g - 1), np.nan)
    out["f3_in_next"] = np.where(row["gap_next"].notna(), present_at(g + 1), np.nan)
    out["f3_in_prev_run"] = present_at(row["prev_g1"])    # last snapshot of the previous run
    out["f3_in_next_run"] = present_at(row["next_g0"])    # first snapshot of the next run
    out["f3_in_run_end"] = present_at(row["run_g1"])      # last snapshot of this run
    out["f3_in_run_start"] = present_at(row["run_g0"])    # first snapshot of this run

    # presence streaks over consecutive snapshots of the session (a streak may span a gap)
    tk = pd.DataFrame({"code": task_code, "g": g, "t": t.to_numpy()}, index=df.index).sort_values(["code", "g"])
    brk = (tk["code"].diff().ne(0)) | (tk["g"].diff().ne(1))
    tk["streak"] = brk.cumsum()
    st = tk.groupby("streak")
    tk["st_g0"] = st["g"].transform("min")
    tk["st_g1"] = st["g"].transform("max")
    tk = tk.reindex(df.index)
    t_of_g = s["t_s"].to_numpy()
    ses_of_g = s["session"].to_numpy()
    g1 = tk["st_g1"].to_numpy()
    g0 = tk["st_g0"].to_numpy()
    out["f3_streak_t_to_end"] = t_of_g[g1] - t.to_numpy()       # last consecutive snapshot still listing it
    out["f3_streak_t_from_start"] = t.to_numpy() - t_of_g[g0]
    nxt = np.minimum(g1 + 1, len(s) - 1)
    vanishes = (g1 + 1 < len(s)) & (ses_of_g[nxt] == df["session"].to_numpy())
    out["f3_t_to_vanish"] = np.where(vanishes, t_of_g[nxt] - t.to_numpy(), np.nan)  # first snapshot without it
    out["f3_vanish_in_run"] = np.where(vanishes, (g1 + 1 <= row["run_g1"].to_numpy()).astype(float), np.nan)
    out["f3_vanish_at_run_end"] = np.where(vanishes, (g1 == row["run_g1"].to_numpy()).astype(float), np.nan)
    first_seen = t.groupby(task_code).transform("min")
    last_seen = t.groupby(task_code).transform("max")
    out["f3_task_t_since_first"] = t - first_seen
    out["f3_task_t_to_last"] = last_seen - t
    # share of this run's snapshots that list the task
    run_task = pd.Series(1, index=df.index).groupby([row["run"].to_numpy(), task_code]).transform("sum")
    out["f3_run_task_frac"] = run_task / row["run_len"]

    # aircraft-level input log
    cs1, cs2 = callsigns(df)
    ev = input_events(df, cs1)  # a row's a_ columns describe its first aircraft
    for tag, cs in (("", cs1), ("2", cs2)):
        keys = pd.DataFrame({"session": df["session"].to_numpy(), "cs": cs.to_numpy(), "t_s": t.to_numpy()})
        nx = _asof(keys, ev, "forward", exact=False)
        pv = _asof(keys, ev, "backward", exact=True)
        out[f"f3_tt_next_act{tag}"] = nx["act_t"].to_numpy() - t.to_numpy()
        out[f"f3_t_since_act{tag}"] = t.to_numpy() - pv["act_t"].to_numpy()
        if tag == "":
            out["f3_next_act_code"] = nx["code"].to_numpy()
            out["f3_prev_act_code"] = pv["code"].to_numpy()
    tt = pd.Series(out["f3_tt_next_act"], index=df.index)
    out["f3_next_act_in_run"] = np.where(tt.notna(), (tt <= out["f3_run_t_to_end"] + 3).astype(float), np.nan)
    out["f3_next_act_before_next_run"] = np.where(
        tt.notna(), (tt <= (row["next_t0"] - t).fillna(np.inf)).astype(float), np.nan)
    # number of inputs on this aircraft in the next 60 s
    ev_key = ev["session"] + "|" + ev["cs"]
    ev_by = dict(tuple(ev.assign(k=ev_key).groupby("k")["act_t"]))
    key1 = (df["session"] + "|" + cs1.fillna("")).to_numpy()
    n60 = np.zeros(len(df))
    tv = t.to_numpy()
    for k in np.unique(key1):
        if k not in ev_by:
            continue
        times = np.sort(ev_by[k].to_numpy())
        idx = np.flatnonzero(key1 == k)
        n60[idx] = np.searchsorted(times, tv[idx] + 60, side="right") - np.searchsorted(times, tv[idx], side="right")
    out["f3_n_act_next60"] = n60

    # aircraft presence: last time any task of this aircraft is listed, and in the next run's first snapshot
    ac_code = pd.factorize(df["session"] + "|" + cs1.fillna(""))[0].astype(np.int64)
    out["f3_ac_t_to_last"] = t.groupby(ac_code).transform("max") - t
    n_ac = ac_code.max() + 1
    ac_present = np.unique(g.astype(np.int64) * n_ac + ac_code)
    ng = row["next_g0"].to_numpy(dtype=float)
    ok = ~np.isnan(ng)
    v = np.full(len(df), np.nan)
    v[ok] = np.isin(ng[ok].astype(np.int64) * n_ac + ac_code[ok], ac_present)
    out["f3_ac_in_next_run"] = v

    res = pd.DataFrame(out, index=df.index)

    # within-snapshot ranks (1 = soonest); NaN stays NaN
    sid = df["sample_id"]
    for c in ["f3_tt_next_act", "f3_t_to_vanish", "f3_streak_t_to_end", "f3_task_t_to_last"]:
        res[f"{c}_rk"] = res[c].groupby(sid).rank(method="min")
        res[f"{c}_trk"] = res[c].groupby([sid, df["task_index"]]).rank(method="min")
    return res
