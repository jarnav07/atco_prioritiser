"""Tier 3d: separating the tasks of one aircraft, and tying an HMI input to the task it closes.

After f3 most remaining misses are (a) several tasks of the same aircraft that f3 sees identically (the aircraft
gets the next input, several of its tasks vanish together), and (b) tasks that are worked without closing.
Features (prefix f5_), all within one session and computed from f3's building blocks:
- per aircraft in the snapshot: number of its tasks, how many vanish at the next run, rank of this task's
  vanish time among them, is it the only one that vanishes;
- does the next input match the task type (assume <- assume / force assume, transfer <- initiate transfer,
  zone / return-to-route <- lateral clearance, exit conditions / conflicts / QoS <- level clearance);
- vanish time minus time to the next input (the input that precedes the closure closed the task);
- inputs on the aircraft inside this run's window and before the next run;
- tasks of the aircraft that appear by the next run (a worked task often spawns the next one, e.g. assume ->
  exit coordination) and that appeared since the previous run.

Look-ahead: uses future snapshots of the same session (see f3 / FINDINGS.md). Never reads `target_index`.
"""
import numpy as np
import pandas as pd

from . import f3_temporal as f3

# task_index -> input codes (f3.ACTIONS order, 1-based) that close / work it
MATCH = {4: {1, 2}, 5: {6}, 12: {4}, 13: {4}, 10: {3}, 6: {3, 4}, 1: {3, 4}, 0: {3, 4}, 8: {3}, 7: {3, 4}}


def build(df: pd.DataFrame, base: pd.DataFrame | None = None) -> pd.DataFrame:
    b = f3.build(df) if base is None else base
    sid = df["sample_id"]
    cs1, _ = f3.callsigns(df)
    key = [sid, cs1.fillna("?")]
    out = pd.DataFrame(index=df.index)

    vanish_nr = 1.0 - b["f3_in_next_run"]
    out["f5_ac_n_tasks"] = sid.groupby(key).transform("size")
    out["f5_ac_n_vanish_nr"] = vanish_nr.groupby(key).transform("sum")
    out["f5_ac_only_vanisher"] = ((vanish_nr == 1) & (out["f5_ac_n_vanish_nr"] == 1)).astype(float)
    out["f5_ac_vanish_rk"] = b["f3_t_to_vanish"].groupby(key).rank(method="min")
    out["f5_ac_age_rk"] = df["x_task_age"].groupby(key).rank(method="min", ascending=False)

    code = b["f3_next_act_code"]
    match = np.array([c in MATCH.get(t, set()) if c == c else False
                      for t, c in zip(df["task_index"].to_numpy(), code.to_numpy())], dtype=float)
    out["f5_next_act_matches_type"] = np.where(code.isna(), np.nan, match)
    out["f5_match_tt"] = np.where(match == 1, b["f3_tt_next_act"], np.nan)
    out["f5_match_tt_rk"] = out["f5_match_tt"].groupby(sid).rank(method="min")
    out["f5_vanish_minus_act"] = b["f3_t_to_vanish"] - b["f3_tt_next_act"]

    # inputs on this aircraft inside [run start, run end + 3 s] and in (t, next run start]
    s = f3.sample_table(df).set_index("sample_id")
    ev = f3.input_events(df, cs1)
    ev_by = {k: np.sort(v.to_numpy()) for k, v in ev.groupby(ev["session"] + "|" + ev["cs"])["act_t"]}
    k1 = (df["session"] + "|" + cs1.fillna("?")).to_numpy()
    t = df["t_s"].to_numpy()
    r0 = df["sample_id"].map(s["run_t0"]).to_numpy()
    r1 = df["sample_id"].map(s["run_t1"]).to_numpy() + 3
    nr = df["sample_id"].map(s["next_t0"]).fillna(np.inf).to_numpy()
    n_run = np.zeros(len(df))
    n_to_nr = np.zeros(len(df))
    for k in np.unique(k1):
        times = ev_by.get(k)
        if times is None:
            continue
        i = np.flatnonzero(k1 == k)
        n_run[i] = np.searchsorted(times, r1[i], side="right") - np.searchsorted(times, r0[i], side="left")
        n_to_nr[i] = np.searchsorted(times, nr[i], side="right") - np.searchsorted(times, t[i], side="right")
    out["f5_n_act_in_run"] = n_run
    out["f5_n_act_to_next_run"] = n_to_nr

    # snapshot context (constant within a sample): is this a run with visible closures / inputs at all?
    # A silent run (nothing vanishes, no input) calls for prior-style ranking instead.
    out["f5_s_n_vanish_nr"] = vanish_nr.groupby(sid).transform("sum")
    has_act = pd.Series((n_run > 0).astype(float), index=df.index)
    out["f5_s_n_act_in_run"] = has_act.groupby(sid).transform("sum")
    out["f5_s_n_act_to_nr"] = pd.Series((n_to_nr > 0).astype(float), index=df.index).groupby(sid).transform("sum")

    # the aircraft's task set at the next run / previous run: tasks gained
    g = df["sample_id"].map(s["g"]).to_numpy().astype(np.int64)
    ac = pd.factorize(df["session"] + "|" + cs1.fillna("?"))[0].astype(np.int64)
    task = pd.factorize(df["session"] + "|" + df["task_name"])[0].astype(np.int64)
    rows = pd.DataFrame({"g": g, "ac": ac, "task": task})
    tasks_at = rows.groupby(["g", "ac"])["task"].agg(frozenset)
    for tag, col in (("nr", "next_g0"), ("pr", "prev_g1")):
        other = df["sample_id"].map(s[col]).to_numpy()
        ok = ~np.isnan(other)
        here = tasks_at.reindex(pd.MultiIndex.from_arrays([g, ac])).to_numpy()
        there = tasks_at.reindex(pd.MultiIndex.from_arrays([np.where(ok, other, -1).astype(np.int64), ac])).to_numpy()
        if tag == "nr":
            n_new = [len(b_ - a_) if isinstance(b_, frozenset) else np.nan for a_, b_ in zip(here, there)]
        else:
            n_new = [len(a_ - b_) if isinstance(b_, frozenset) else np.nan for a_, b_ in zip(here, there)]
        out[f"f5_ac_new_tasks_{tag}"] = np.where(ok, n_new, np.nan)
    return out
