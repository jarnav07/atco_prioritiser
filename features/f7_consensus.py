"""Tier 4: cross-controller consensus on the shared traffic - when do *other* controllers work on this task?

All controllers fly the same three scenarios and share the overall order of work (deck: Kendall W = 0.98), so the
labelled sessions of other controllers give a prior on when each task (same scenario, same task_name) gets worked.
Measured alone, with the scenario clock below and leave-one-controller-out pools: top-3 0.48 at a +-60 s window
(type prior 0.39, chance 0.09).

Two pieces, both causal for the session being predicted:
- `scenario_clock`: timestamps are wall-clock, so each session is shifted onto a reference clock. An aircraft's
  sector-arrival time (t_s + time_to_sector) is fixed by the traffic; the offset is the running median, over the
  aircraft seen *so far* in the session, of (first arrival estimate - the reference session's arrival).
- `build(query, pool)`: features from the pool's labelled targets (other sessions only - a row never sees its own
  session's labels). The caller decides the pool: training-fold sessions in CV, all labelled sessions for test.

Deployment caveat: this needs recordings of other controllers on the same traffic, which a live sector never has.
"""
import numpy as np
import pandas as pd

from .f3_temporal import callsigns

ARR_RANGE = (1.0, 900.0)   # time_to_sector window where t_s + time_to_sector is a clean arrival estimate
WINDOWS = (30, 60, 120)


def _first_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    cs, _ = callsigns(df)
    m = df["time_to_sector"].between(*ARR_RANGE)
    a = pd.DataFrame({"session": df["session"][m], "scenario": df["scenario"][m], "cs": cs[m],
                      "t": df["t_s"][m], "arr": (df["t_s"] + df["time_to_sector"])[m]})
    return a.sort_values("t").drop_duplicates(["session", "cs"], keep="first")


def reference_arrivals(labelled: pd.DataFrame) -> dict:
    """(scenario, callsign) -> arrival time on the scenario's reference clock (its first labelled session)."""
    a = _first_arrivals(labelled)
    ref = a.groupby("scenario")["session"].min()
    a = a[a["session"].to_numpy() == a["scenario"].map(ref).to_numpy()]
    return {(sc, cs): v for sc, cs, v in zip(a["scenario"], a["cs"], a["arr"])}


def scenario_clock(df: pd.DataFrame, ref_arr: dict) -> np.ndarray:
    """u = t_s - offset(t), offset estimated from the aircraft seen up to t (NaN before the first one)."""
    a = _first_arrivals(df)
    a["d"] = a["arr"] - [ref_arr.get((sc, cs), np.nan) for sc, cs in zip(a["scenario"], a["cs"])]
    a = a.dropna(subset=["d"]).sort_values("t")
    u = np.full(len(df), np.nan)
    ses_all = df["session"].to_numpy()
    t_all = df["t_s"].to_numpy()
    for ses, e in a.groupby("session"):
        med = e["d"].expanding().median().to_numpy()
        idx = np.flatnonzero(ses_all == ses)
        pos = np.searchsorted(e["t"].to_numpy(), t_all[idx], side="right") - 1
        ok = pos >= 0
        u[idx[ok]] = t_all[idx[ok]] - med[pos[ok]]
    return u


def build(query: pd.DataFrame, u_query: np.ndarray, pool: pd.DataFrame, u_pool: np.ndarray) -> pd.DataFrame:
    """Consensus features for every query row from the pool's targets in *other* sessions of the same scenario."""
    is_t = pool["target_index"].to_numpy() == 1
    P = pd.DataFrame({"key": (pool["scenario"].astype(str) + "|" + pool["task_name"]).to_numpy()[is_t],
                      "ses": pool["session"].to_numpy()[is_t], "u": u_pool[is_t]}).dropna(subset=["u"])
    n_ses = pool.groupby("scenario")["session"].nunique()
    qkey = (query["scenario"].astype(str) + "|" + query["task_name"]).to_numpy()
    qses = query["session"].to_numpy()
    out = {f"f7_n{w}": np.zeros(len(query)) for w in WINDOWS}
    out["f7_near_dt"] = np.full(len(query), np.nan)
    out["f7_frac_ses_ever"] = np.zeros(len(query))
    by = {k: (g["u"].to_numpy(), g["ses"].to_numpy()) for k, g in P.groupby("key")}
    for k, idx in pd.Series(np.arange(len(query))).groupby(qkey):
        hit = by.get(k)
        if hit is None:
            continue
        pu, ps = hit
        i = idx.to_numpy()
        other = qses[i][:, None] != ps[None, :]                       # never the query's own session
        dt = u_query[i][:, None] - pu[None, :]
        for w in WINDOWS:
            out[f"f7_n{w}"][i] = ((np.abs(dt) <= w) & other).sum(1)
        adt = np.where(other, np.abs(dt), np.inf)
        j = adt.argmin(1)
        near = dt[np.arange(len(i)), j]
        out["f7_near_dt"][i] = np.where(np.isfinite(adt.min(1)), near, np.nan)
        # share of the other pool sessions of this scenario that ever worked this task
        n_ever = np.array([len(set(ps[o])) for o in other])
        out["f7_frac_ses_ever"][i] = n_ever
    res = pd.DataFrame(out, index=query.index)
    # normalise counts by the number of *other* pool sessions of the scenario (pool sizes differ train vs valid)
    in_pool = query["session"].isin(set(pool["session"])).to_numpy()
    denom = query["scenario"].map(n_ses).fillna(0).to_numpy() - in_pool
    denom = np.where(denom > 0, denom, np.nan)
    for c in [f"f7_n{w}" for w in WINDOWS] + ["f7_frac_ses_ever"]:
        res[c] = res[c] / denom
    res.loc[np.isnan(u_query), [f"f7_n{w}" for w in WINDOWS] + ["f7_near_dt"]] = np.nan
    sid = query["sample_id"]
    res["f7_n60_rk"] = (-res["f7_n60"]).groupby(sid).rank(method="min")
    res["f7_abs_near_rk"] = res["f7_near_dt"].abs().groupby(sid).rank(method="min")
    return res


# task_index -> HMI input codes (f3.ACTIONS order, 1-based) that work / close it
MATCH = {4: (1, 2), 5: (6,), 12: (4,), 13: (4,), 10: (3,), 6: (3, 4), 1: (3, 4), 0: (3, 4), 8: (3,), 7: (3, 4)}


def pool_inputs(pool: pd.DataFrame, u_pool: np.ndarray) -> pd.DataFrame:
    """Every HMI input the pool sessions reveal, on the scenario clock: (scenario, session, cs, code, u)."""
    from .f3_temporal import ACTIONS, NO_INPUT
    cs, _ = callsigns(pool)
    seen = (pool["a_since_flight"] < NO_INPUT).to_numpy() & ~np.isnan(u_pool)
    ev = pd.DataFrame({"scenario": pool["scenario"].to_numpy(), "ses": pool["session"].to_numpy(),
                       "cs": cs.to_numpy(), "u": np.round(u_pool - pool["a_since_flight"].to_numpy()),
                       "code": (pool[ACTIONS].fillna(0).to_numpy() * np.arange(1, 7)).max(axis=1)})[seen]
    return ev.drop_duplicates(["ses", "cs", "u"]).reset_index(drop=True)


def build_inputs(query: pd.DataFrame, u_query: np.ndarray, ev: pd.DataFrame) -> pd.DataFrame:
    """When did *other* controllers act on this aircraft, relative to now on the scenario clock?

    For the input kind that works this task type (assume -> assume, transfer -> initiate transfer, zone / RTR ->
    lateral, conditions / conflicts / QoS -> level) and for any input: the median over other sessions of the time
    to their nearest such input, and the share of other sessions with one in the next 60 s. This is the same-traffic
    stand-in for f3's look-ahead "time to the next input", taken from other recordings instead of the future.
    """
    cs, _ = callsigns(query)
    n = len(query)
    out = {k: np.full(n, np.nan) for k in ("f7_in_match_med_dt", "f7_in_match_next60", "f7_in_any_med_dt",
                                           "f7_in_any_next60")}
    qses, ti = query["session"].to_numpy(), query["task_index"].to_numpy()
    key = (query["scenario"].astype(str) + "|" + cs.fillna("?")).to_numpy()
    ev_key = (ev["scenario"].astype(str) + "|" + ev["cs"]).to_numpy()
    groups = {k: g for k, g in ev.groupby(ev_key)}
    for k, idx in pd.Series(np.arange(n)).groupby(key):
        g = groups.get(k)
        if g is None:
            continue
        i = idx.to_numpy()
        for tag, codes_of in (("match", lambda t: MATCH.get(t)), ("any", lambda t: (1, 2, 3, 4, 5, 6, 0))):
            for t_type in np.unique(ti[i]):
                codes = codes_of(t_type)
                if codes is None:
                    continue
                gg = g[g["code"].isin(codes)]
                if gg.empty:
                    continue
                j = i[ti[i] == t_type]
                sessions = gg["ses"].unique()
                dts = np.full((len(j), len(sessions)), np.nan)
                for c, s in enumerate(sessions):
                    us = np.sort(gg.loc[gg["ses"] == s, "u"].to_numpy())
                    dt = us[None, :] - u_query[j][:, None]                 # >0: their input is ahead of now
                    nearest = np.take_along_axis(dt, np.abs(dt).argmin(1)[:, None], 1)[:, 0]
                    dts[:, c] = np.where(qses[j] == s, np.nan, nearest)  # never the row's own session
                n_other = (~np.isnan(dts)).sum(1)
                with np.errstate(all="ignore"):
                    out[f"f7_in_{tag}_med_dt"][j] = np.nanmedian(np.where(n_other[:, None] > 0, dts, 0), 1)
                    nxt = ((dts >= 0) & (dts <= 60)).sum(1)
                    out[f"f7_in_{tag}_next60"][j] = np.where(n_other > 0, nxt / np.maximum(n_other, 1), np.nan)
    res = pd.DataFrame(out, index=query.index)
    res.loc[np.isnan(u_query)] = np.nan
    sid = query["sample_id"]
    res["f7_in_match_med_rk"] = res["f7_in_match_med_dt"].abs().groupby(sid).rank(method="min")
    res["f7_in_any_next60_rk"] = (-res["f7_in_any_next60"]).groupby(sid).rank(method="min")
    return res
