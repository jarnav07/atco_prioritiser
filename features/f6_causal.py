"""Tier 3c: causal session context - only the current and earlier snapshots of the session (no look-ahead).

The final evaluation has to work snapshot by snapshot, so every value here is computable at time t from rows with
t_s <= t (`python -m harness.causal raw+f6` proves it by truncating sessions). Families (prefix f6_):

- run-relative timing: time since this run (= the current labelled episode) started, the task's age and its
  coordination-state age at the run start, whether the coordination state changed / an input landed on the aircraft
  during this run (targets 5.4x more often than other tasks for a state change inside the run);
- the aircraft's past input log, recovered from the `a_since_flight` values of earlier snapshots: inputs in the
  last 60 / 300 s, time since the last input of each kind;
- the aircraft's past state changes (clearance / coordination / conditions groups), backward half of f4;
- the controller's own style in this session, learned without labels from their earlier inputs: how far from the
  sector boundary they assumed / transferred aircraft so far (running median), and how this row's aircraft compares.

Caveat: the run structure itself comes from the dataset keeping only labelled moments; a live 3 s feed has none.
"""
import numpy as np
import pandas as pd

from . import f4_state_changes as f4
from .f3_temporal import ACTIONS, NO_INPUT, callsigns, sample_table

ASSUME_CODES = (1, 2)   # assume_control, force_assume_control (f3.ACTIONS order, 1-based)
TRANSFER_CODE = 6       # initiate_transfer_of_control
STYLE_MAX_DELAY = 10.0  # only inputs seen within 10 s of happening give a clean "where was the aircraft" reading


def _events(df: pd.DataFrame, cs: pd.Series) -> pd.DataFrame:
    """One row per (session, aircraft, input time): kind, when first revealed, aircraft geometry at that snapshot."""
    seen = df["a_since_flight"] < NO_INPUT
    ev = pd.DataFrame({
        "session": df["session"], "cs": cs, "act_t": (df["t_s"] - df["a_since_flight"]).round(),
        "code": (df[ACTIONS].fillna(0).to_numpy() * np.arange(1, 7)).max(axis=1),
        "rev_t": df["t_s"], "tts": df["time_to_sector"], "ttx": df["k_timeToExitPoint"],
    })[seen]
    ev = ev.sort_values("rev_t").drop_duplicates(["session", "cs", "act_t"], keep="first")
    return ev.reset_index(drop=True)


def _running_median(ev: pd.DataFrame, value: str, keys: pd.DataFrame) -> np.ndarray:
    """For each (session, t_s) key: median of `value` over events revealed at or before t_s in that session."""
    out = np.full(len(keys), np.nan)
    for ses, e in ev.groupby("session"):
        e = e.sort_values("rev_t")
        med = e[value].expanding().median().to_numpy()
        idx = np.flatnonzero(keys["session"].to_numpy() == ses)
        pos = np.searchsorted(e["rev_t"].to_numpy(), keys["t_s"].to_numpy()[idx], side="right") - 1
        ok = pos >= 0
        out[idx[ok]] = med[pos[ok]]
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    s = sample_table(df).set_index("sample_id")
    t = df["t_s"].to_numpy()
    run_t0 = df["sample_id"].map(s["run_t0"]).to_numpy()
    t_run = t - run_t0

    # run-relative timing
    out["f6_t_run"] = t_run
    out["f6_run_pos"] = df["sample_id"].map(s["run_pos"]).to_numpy()
    out["f6_gap_prev_run"] = df["sample_id"].map(s["run_t0"]).to_numpy() - df["sample_id"].map(s["prev_t1"]).to_numpy()
    out["f6_task_age_at_start"] = df["x_task_age"].to_numpy() - t_run
    out["f6_state_age_at_start"] = df["x_state_age"].to_numpy() - t_run
    out["f6_state_chg_in_run"] = (df["x_state_age"].to_numpy() <= t_run).astype(float)
    a = df["a_since_flight"].to_numpy(dtype=float)
    out["f6_input_age_at_start"] = np.where(a < NO_INPUT, a - t_run, np.nan)
    out["f6_input_in_run"] = ((a <= t_run + 1) & (a < NO_INPUT)).astype(float)

    # the aircraft's past input log (events with act_t <= t on the row's own aircraft are all revealed by t)
    cs1, _ = callsigns(df)
    ev = _events(df, cs1)
    key = (df["session"] + "|" + cs1.fillna("?")).to_numpy()
    ev_key = (ev["session"] + "|" + ev["cs"]).to_numpy()
    n60, n300, n_all = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    since = {c: np.full(len(df), np.nan) for c in ("assume", "transfer", "level", "lateral")}
    kinds = {"assume": ASSUME_CODES, "transfer": (TRANSFER_CODE,), "level": (3,), "lateral": (4,)}
    for k in np.unique(key):
        m = ev_key == k
        if not m.any():
            continue
        e = ev[m].sort_values("act_t")
        times, codes = e["act_t"].to_numpy(), e["code"].to_numpy()
        i = np.flatnonzero(key == k)
        hi = np.searchsorted(times, t[i], side="right")
        n60[i] = hi - np.searchsorted(times, t[i] - 60, side="right")
        n300[i] = hi - np.searchsorted(times, t[i] - 300, side="right")
        n_all[i] = hi
        for name, cset in kinds.items():
            kt = times[np.isin(codes, cset)]
            if len(kt):
                j = np.searchsorted(kt, t[i], side="right") - 1
                ok = j >= 0
                since[name][i[ok]] = t[i][ok] - kt[j[ok]]
    out["f6_n_inputs_60"] = n60
    out["f6_n_inputs_300"] = n300
    out["f6_n_inputs_session"] = n_all
    for name, v in since.items():
        out[f"f6_t_since_{name}"] = v

    # the aircraft's past state changes (backward half of f4; f4.build is causal for these columns)
    f4_all = f4.build(df)
    for c in ["f4_n_chg_pr"] + [f"f4_chg_pr_{g}" for g in f4.GROUPS] + [f"f4_t_since_chg_{g}" for g in f4.GROUPS]:
        out[f"f6_{c[3:]}"] = f4_all[c].to_numpy()

    # controller style so far in this session: geometry of the aircraft at their earlier assume / transfer inputs
    fresh = ev[(ev["rev_t"] - ev["act_t"]) <= STYLE_MAX_DELAY]
    keys = pd.DataFrame({"session": df["session"].to_numpy(), "t_s": t})
    med_assume = _running_median(fresh[fresh["code"].isin(ASSUME_CODES)], "tts", keys)
    med_transfer = _running_median(fresh[fresh["code"] == TRANSFER_CODE], "ttx", keys)
    out["f6_style_assume_tts"] = med_assume
    out["f6_style_transfer_ttx"] = med_transfer
    out["f6_tts_minus_style"] = df["time_to_sector"].to_numpy() - med_assume
    out["f6_ttx_minus_style"] = df["k_timeToExitPoint"].to_numpy() - med_transfer

    res = pd.DataFrame(out, index=df.index)
    sid = df["sample_id"]
    for c in ["f6_tts_minus_style", "f6_ttx_minus_style", "f6_state_age_at_start", "f6_t_since_chg_coord",
              "f6_t_since_chg_clr"]:
        res[f"{c}_trk"] = res[c].abs().groupby([sid, df["task_index"]]).rank(method="min")
    return res
