# ENGAGE KTN 2026 — AWARE Task-Prioritisation Challenge
## Competition brief & agent execution plan

**Event:** Engage #2 Hackathon, University of Zagreb, Borongaj Campus (Building 69), 20–21 Oct 2026
**Challenge owner:** Tomislav Radišić, Faculty of Transport and Traffic Sciences, University of Zagreb
**Funding/partners:** AWARE project + Engage 2, SESAR 3 JU, Horizon Europe (GA 101167442, 101114648); CANSO sponsors prizes
**Kaggle:** `engage-ktn-2026-competition` — Community Prediction Competition, private, 5 files, 148.75 MB
**Prize:** Airspace World 2027 Pack (Lisbon trip, ≤€750/person + €3,000/team travel, €100 voucher/participant, CEO breakfast, "Future Skies: Tomorrow's Voices" slot)
**Sources:** `AWARE_prioritisation_challenge.pdf` (34-slide briefing deck) + Kaggle Overview / Data / Rules / Leaderboard pages.

---

## 0. Hard logistics — read this first

### Agenda (from the deck)

| Tue 20 Oct | | Wed 21 Oct | |
|---|---|---|---|
| 12:00–13:45 | check-in, welcome, challenge guidelines | 07:30–10:00 | breakfast (canteen) |
| 13:45–14:00 | Q&A | **08:00–13:00** | **challenge session (5 h)** |
| **14:00–21:00** | **challenge session (7 h)** | 13:00–13:30 | jury evaluation of results |
| 20:00 / 21:00 | buses Campus → hotel / city centre | 13:30–14:30 | **top-3 interviews with the jury** |
| — | dinner = lunch boxes at the venue | 15:00–15:30 | winner announced |

**Actual working time is ~12 hours in two blocks (7 h Tue + 5 h Wed), not a continuous 20-hour run.** Public transport from campus is poor in the evening — the organised buses are the plan. Build the schedule around an overnight break where nothing gets trained.

### Deadline — two documents disagree, take the earlier one

| Source | Close |
|---|---|
| Briefing deck | "Kaggle: submissions open Tue 14:00 and close **Wed 13:00**" |
| Kaggle Rules page §2.3.b | "21 October 2026, **14:00 CEST** (12:00 UTC)" |
| Kaggle countdown (observed Tue ~17:00) | ≈ Wed 13:00 CEST |

**Treat Wed 21 Oct, 13:00 CEST (11:00 UTC) as the hard deadline.** The deck and the live countdown agree; the jury evaluates at 13:00. Verify on the Kaggle Overview → Timeline page early, and have the final submission in by **12:30 CEST** regardless.

### Submission budget — also disagrees

- **5 submissions per team per day.** Kaggle's day rolls at **02:00 Zagreb time (00:00 UTC)** → the budget refreshes overnight, so you get 5 on Tue and 5 on Wed.
- Final submissions: the deck says **"select up to 2"**, the Kaggle Rules page says **one (1)**. Select whatever the Kaggle UI actually permits, and **select manually** — if none is selected, Kaggle auto-picks the best *public* score, which is the wrong objective here (see §2.5).

### Team mechanics (deck, slide "Teams on Kaggle")

Teams of 2–4 as registered; solo applicants are matched on site. **Form the complete Kaggle team before anyone submits** — Kaggle sums the submissions of members who merge, and merges between registered teams are not allowed. Team name must match the registration. No private sharing of code, models, predictions or data between teams.

### Winning is not purely the score

Top 3 teams **by private leaderboard** become finalists; they present to the jury (13:30–14:30 Wed) and **the jury picks the winner**. Rule 2.8.a: on request, finalists must hand over training + inference code that reproduces the submission from the data, plus a method description and environment spec — a condition of receiving the prize. So reproducibility and a clear 10-minute story are deliverables, not afterthoughts.

---

## 1. The problem

An en-route air traffic controller (ATCO) works one sector — **ESMM_23**, Swedish upper airspace. The AWARE system maintains, every 3 seconds, the list of the controller's **open tasks** (assume this aircraft, transfer that one, resolve this conflict, avoid that military zone…). About **50 tasks are open at any moment**; the controller works one of them for **5–40 seconds**, then switches. The choice is not simply the most urgent task — transfers, entries, conflicts and zone tasks are interleaved.

AWARE already knows *which tasks are open*. **What it cannot do is rank them the way a controller does.** That is the challenge: for each labelled moment, predict which open task the controller is working on. Downstream it feeds EASA **Level 2A** (AI advises on the task the ATCO is on) and **Level 2B** (AI executes queued routine tasks after 30 s unless dismissed).

Note what the real AWARE system uses that **you do not get**: gaze/eye-tracking. The dataset gives you traffic context + HMI input logs only. The intent signal has to come from the traffic situation and what the controller recently typed.

### How the labels were made (matters for noise)

1. After each session the controller replayed their own screen recording and marked the **start and end** of every task they worked on (task type + callsign).
2. Labels were quality-controlled and corrected by an **independent subject-matter expert**.
3. The recordings were replayed through AWARE, which emitted its task list every 3 s; each label was matched to the corresponding task in that list.

So the labels are self-report + expert correction, episode-based, then discretised to a 3 s grid.

### Data collection

Sept 2025, Zagreb: **23 UkSATSE controllers**, three **1-hour** scenarios each, eye-tracking + HMI logs + self-annotation. 19 controllers have annotated sessions → **42 annotated sessions, 5,043 task labels**.

### Shape of the data

| File | Content | Samples | Rows | Sessions |
|---|---|---|---|---|
| `train.csv` | labelled, has `target_index` | 11,923 | 432,578 | 34 |
| `validation.csv` | labelled — `019sc1`, `019sc2`, `020sc1` | 1,252 | 42,110 | 3 |
| `test.csv` | no `target_index` — `020sc2`, `020sc3`, `023sc1`, `023sc2`, `023sc3` | 1,863 | 66,409 | 5 |
| `sample_submission.csv` | valid baseline (scores 0.352) | 1,863 | — | — |
| `scorer.py` | official offline scorer | — | — | — |
| `FEATURES.md` | every column explained | — | — | — |

- **Sample** = one 3 s snapshot = one `sample_id`. **Row** = one task in it, keyed by `row_index` (0-based). `seq_len` ∈ [9, 59], mean ≈ 36.
- Exactly one row per sample has `target_index = 1`.
- 96 columns = 87 feature inputs + identifiers + label. Six feature groups: basic task features **25**, task type **1** (14 classes), coordination state `x_` **8**, aircraft state & exit `k_` **32**, ATCO action history `a_` **12**, list context `l_` **9**.
- **Effective sample size is far below 15,038: those samples come from about 4,150 label episodes.** A 1-hour session at 3 s gives ~1,200 snapshots, of which only ~350 are labelled moments.

### Metric

**Top-3 hit rate** — mean over samples of `1[true task ∈ first three distinct predicted row_index]`. Every sample counts equally.

| Anchor | Score |
|---|---|
| Random 3 of ~36 | 0.094 |
| `sample_submission.csv` (type prior, then nearest to sector boundary) | 0.352 |
| Public LB, observed Tue evening | 0.931 / 0.925 / 0.890 |

### Submission format

```csv
sample_id,prediction
12797,9 6 41
12798,9 6 41
```

Space-separated `row_index`, **best first**. Only the **first three distinct** ids count; extras ignored, repeats collapse, fewer than three is legal but only hurts. A missing `sample_id`, duplicated `sample_id` or non-numeric id ⇒ **invalid**. Offline: `python scorer.py --labelled validation.csv my_val_submission.csv`

Minimal working pipeline (from the deck, as a shape to fill in):

```python
import pandas as pd
train = pd.read_csv('train.csv')   # target_index == 1 marks the answer
test  = pd.read_csv('test.csv')    # same sample_id == one snapshot

test['score'] = -test['time_to_sector'].abs()      # <- replace with the model

top3 = (test.sort_values('score', ascending=False)
            .groupby('sample_id').row_index
            .apply(lambda r: ' '.join(map(str, r[:3]))))
top3.rename('prediction').reset_index().to_csv('submission.csv', index=False)
```

---

## 2. The seven facts that decide the approach

**(1) The unit of prediction is the sample, not the row.** Exactly one positive per variable-size group; the metric is recall@3 *within* the group. The deck says it outright: *"think ranking, not classification."* Score rows → rank within `sample_id` → take top 3. Train with a **group-wise softmax (listwise cross-entropy)** or **LambdaRank with `eval_at=3`**, not independent binary logloss — the latter spends capacity on absolute calibration against a 1:36 imbalance when only within-group order matters.

**(2) Most columns describe the aircraft, not the task.** All tasks of one aircraft in one snapshot share nearly every column. The deck calls task type *"the only column that always differs between the tasks of one aircraft"* — in practice `x_task_age` and the `l_` columns also differ (the deck's own worked example shows a twin pair with task ages 1270 s vs 496 s). **Raw features alone cannot separate two tasks of the same aircraft.** Everything you build must be (a) within-sample relational, (b) task-type conditional, or (c) temporal. This is the central engineering insight.

**(3) The assume/entry twin has a known prior — use it.** An aircraft often holds both `assume_aircraft` and `entry_coordination` with identical aircraft features (`l_twin_AE = 1`). The deck: **controllers chose `assume` in about three out of four such cases.** With three slots this is nearly free: when the model is confident about the aircraft but not the type, spend two slots on the pair, **assume first**. Because the three candidates are mutually exclusive outcomes, top-3 by *calibrated* within-group probability is the optimal selection policy — so the work is calibration, not clever selection.

**(4) `transfer_aircraft` + `assume_aircraft` are half of all samples.** The label distribution is heavily imbalanced but broad across 14 types (`aircraft_request` does not occur at all; `entry_conditions` is very rare). Compute `P(target | task_index)` early — it alone, with a sensible tie-break, is a respectable baseline and a necessary sanity reference.

**(5) Samples are near-duplicates, and the host quantified the damage.** Snapshots are 3 s apart; consecutive samples usually share the label. In the host's own tests **a random split inflated scores by 25–45 points.** Two consequences:
- **Validate by session, and preferably by controller. Never by random rows.**
- **Exploit it in modelling:** track each task identity `(session, task_name)` across the session and smooth scores along time. Episodes last 5–40 s ⇒ **2–13 snapshots**, so the smoothing window must be short (EWMA half-life of ~2–4 snapshots); anything longer smears across switches. This is legitimate at test time — it smooths predictions, not labels.

**(6) There is a hard human ceiling, and it is low.** From the deck's inter-/intra-rater analysis:
- At the same moment, **two controllers work on the same task only 6.5 % of the time** (same *aircraft* 12 %).
- The overall *order* of work is strongly shared (**Kendall W = 0.98**), and agreement is 84–99 % when tasks are minutes apart — but only 50–66 % when they compete within the same minute, and **tasks due within 30 s of each other are ordered at chance level**.
- Within a controller, style is only partly stable: ICC 0.66 for how early they assume, 0.64 for transfer, and the same controller's assume timing still varies by **~40 s** between sessions.
- The deck's conclusion: *"Expect noise: no model can reach 100 %"*, and *"robust models beat over-fitted ones."*

**This kills the naive "cross-session consensus" idea.** Because the same traffic is flown by many controllers, it is tempting to look up what other controllers did at the same moment of the same scenario. At 6.5 % moment-level agreement that lookup is near-worthless as a direct predictor. What *is* usable is the coarse, scenario-level ordering (W = 0.98) — i.e. a prior on *which tasks come due in what order*, not *who was on what at second T*. Downgrade this from a headline strategy to a low-priority, heavily-validated experiment. See §4 Tier 4.

**(7) The generalisation axis is the controller, and the public LB does not test it honestly.**
- **Public LB = controller 020** (`020sc2`, `020sc3`, 915 samples) — but **`020sc1` is in `validation.csv`**, so 020 is a *partly seen* controller.
- **Private LB = controller 023** (3 sessions, 948 samples) — appears nowhere in train or validation.

The public score is therefore optimistically biased and measures something different from the private score. **Worse, both are tiny.** 948 private samples ≈ **~260 independent episodes**; at p ≈ 0.9 the binomial standard error is **≈ 1.9 points**, so a 1σ band is ±2 and 2σ is ±4. The observed public gap between 1st (0.931) and 2nd (0.925) is **pure noise.** Practical consequence: do not chase sub-1-point gains, do not read the leaderboard as a ranking of method quality, and **select the final submission on cross-validation, not on the public LB.**

---

## 3. Column map and data hygiene

| Family | Examples | Varies within an aircraft? | Use |
|---|---|---|---|
| IDs | `sample_id`, `row_index`, `seq_len`, `session`, `atco`, `scenario`, `timestamp`, `task_name` | — | grouping/keys — **not raw features** |
| Task type (1) | `task_index` 0–13 | **yes** | categorical; interacts with everything |
| Basic task (25) | `distance_to_sector`, `time_to_sector`, `within_sector`, `approaching_sector`, `current_FL`, `exit_FL`, `is_assumed`, `is_not_assumed`, `is_transferred`, `conflict_ind`, `distance_to_MD`, `time_to_MD`, `multiple_conflicts_ind`, `HMI_conflict_safety_alert`, `CPA_entry_location`, `CPA_exit_location`, `what_if_potential_conflict`, `qos_candidate`, `monitoring_rule_activated`, `non_conformance_*`, `zone_conflict_ind`, `distance_to_zone` | no | base signal; **rank/normalise within sample** |
| Coordination in (8) `x_` | `x_isc_notified`, `x_isc_coordinated`, `x_transfer_in`, `x_request_in`, `x_entry_coord_not_done`, `x_in_sector_h`, `x_state_age`, **`x_task_age`** | `x_task_age` **yes** | `x_task_age` is a key twin separator |
| Aircraft state & exit (32) `k_` | `k_isClimbing…`, `k_clearedFL`, `k_cfl_minus_fl`, `k_distanceToExitPoint`, `k_timeToExitPoint`, `k_exitConditionsAreMet`, `k_aircraftNeedsToBeTransferred`, `k_out_NOTIFIED/COORDINATED`, `k_outCoordMismatch`, `k_*ToolActivation` | no | mostly **empty for unassumed aircraft — missingness is itself a feature** |
| ATCO action history (12) `a_` | `a_since_flight`, `a_last_*` (6), `a_since_any`, `a_n60`, `a_n300`, **`a_is_last_cs`**, **`a_is_last2_cs`** | no | **strongest family** — "the controller just typed on this aircraft" |
| List context (9) `l_` | `l_n_type`, `l_frac_type`, `l_n_flight`, **`l_twin_AE`**, `l_rank_f1_type`, `l_rank_absf1_type`, `l_rank_f0_type`, `l_rank_absf1_all`, `l_rank_age_type` | **yes** | the host already did *some* within-sample ranking — **extend that pattern to every numeric column** |

### Hygiene rules (non-negotiable)

- **Empty ≠ 0.** "Empty = does not apply, never 0. Many `k_` columns are empty for aircraft not yet assumed." Never blanket-`fillna(0)`. Use NaN-native trees, or impute + an explicit `_isnan` indicator.
- **Caps are sentinels:** `time_to_sector` and `x_state_age` capped at 3600; `a_since_flight`/`a_since_any` = 3600 means "none in the last hour". Add `== 3600` flags.
- **Signs carry meaning:** `distance_to_sector` / `time_to_sector` negative = inside the sector (distance to the *exit* boundary). Feed both signed value and `abs()`.
- **Redundancy:** `k_cfl_minus_fl` and `k_clearedFLDelta` are the same quantity with opposite sign.
- **Known dirt:** a few `non_conformance_agreed_route_deviation_horizontally` values > 100 km "may be data errors" — clip/winsorise.
- **`task_index` is a category**, not a quantity ("code 12 is not more than code 4"). Use native-categorical / one-hot / target-prior encoding.
- **The nearest-to-boundary heuristic is explicitly not the answer.** The deck's worked example: *"The controller transfers BIX72G although EWG5C is closer to the exit."*

### Leakage / anti-generalisation

- **Never feed `atco` or `session` as features** — test controllers are unseen, so these can only teach per-controller bias. Grouping/CV only.
- **`scenario` and `timestamp` are a judgement call.** Same scenario = identical traffic, flown by other controllers, and all three test scenarios appear in training. The deck: *"memorising one controller's choices does not help."* Gate scenario-conditioned features behind **leave-controller-out** CV; include only if they help held-out *controllers*, not just the public LB.
- **Hand-labelling the test set is banned** (competition rule 2.6.a and Kaggle foundational rule 4.b). **External data is banned** — only the provided data. **LLMs, open-source libraries and AutoML are explicitly allowed**, provided they bring in no external data.

---

## 4. Modelling ladder

### Tier 0 — Harness (before any modelling)
Loader, fold-assignment file, metric wrapper around `scorer.py`, submission writer + validator, one reproducible entry point. Freeze it. Every experiment reports one number from one function.

### Tier 1 — Honest baselines (floor 0.352; expect 0.6–0.8)
1. Reproduce `sample_submission` logic locally to prove the harness.
2. Prior-only: rank by `P(target | task_index)`, tie-break on `l_rank_absf1_type`.
3. **LightGBM `lambdarank`**, `group = sample_id`, `eval_at=[3]`, 87 raw features with `task_index` categorical. The workhorse — get it on the board on Day 1.

### Tier 2 — Within-sample relational features (biggest single jump)
For each numeric column `c`, **within each `sample_id`**: rank and percentile (ties → lowest rank, matching the host's `l_` convention), `c − mean`, `c / max|c|`, z-score. Then the same **within `(sample_id, task_index)`** — generalising `l_rank_*_type` to every column. Plus: rank among tasks of the *same aircraft*; per-type counts/shares; `seq_len` interactions; is this the aircraft's only task; does this aircraft hold the most tasks; explicit twin pairing off `l_twin_AE`.

### Tier 3 — Temporal / session features (second biggest, and cheap)
Identity key `(session, task_name)`. Per key across the session: snapshots alive (cross-check `x_task_age`); present in previous snapshot; time since appearance; deltas of key features vs 1/3/5 snapshots back (did the conflict just appear? did it just cross the boundary?); session time index; workload trajectory (`a_n60`, `a_n300`).
Then **post-hoc smoothing**: EWMA each key's score along time before taking top 3, with half-life tuned on held-out sessions in the **2–4 snapshot** range (episodes are 5–40 s). Also consider explicit episode modelling: predict "is this the same task as the previous snapshot's argmax" and use it as a stickiness prior.

### Tier 4 — Scenario-level priors (LOW priority — see fact 6)
Moment-level cross-controller consensus is near chance (6.5 %) and will mislead. What may survive: a coarse, scenario-level ordering prior (W = 0.98), e.g. "at roughly this stage of this scenario, task type X with this geometry tends to be due". Build it as a *feature*, never a predictor; validate leave-controller-out **and** leave-scenario-out before shipping. **Expect to drop this.** Timebox it to one hour and kill it on the first flat result.

### Tier 5 — Diversity and blending
- **Set/listwise neural model:** the input is literally a set of ≤59 items with one correct answer — a small set-transformer / DeepSets with a softmax over the group is architecturally correct and errs differently from a GBDT.
- **CatBoost** (native categoricals) and **XGBoost `rank:pairwise`** for cheap GBDT diversity.
- **Two-stage:** predict the task *type* first, then the aircraft within type. Or per-type specialist rankers.
- **Blend at score level** after per-sample softmax normalisation (average normalised probabilities or ranks); weights by CV. Given the ±2-point noise floor, blending for *robustness* is worth more than squeezing the single best model.

### Tier 6 — Selection policy
With calibrated mutually-exclusive within-group probabilities, plain top-3 is optimal — so invest in calibration. Then hand-inspect two cases: twin pairs (is the pair covered, assume first?), and samples where all three picks are one aircraft or one type (genuine confidence, or a degenerate mode?).

---

## 5. Validation protocol

1. **Pool `train.csv` + `validation.csv`** (37 labelled sessions, 13,175 samples) — explicitly permitted and you need the extra controllers.
2. **GroupKFold by `atco`**, 5 folds. Controller, not session — the private LB is an unseen controller. Whole sessions stay together as a consequence.
3. **Write the fold assignment to one file.** Every experiment reads it. No re-randomising, ever.
4. Report **mean ± across-fold spread** plus a per-fold table. Controller-to-controller variance *is* the quantity of interest; a feature that helps 3 folds and wrecks 2 is a trap.
5. **Size your error bars on episodes, not samples.** ~4,150 episodes over 15,038 samples ⇒ divide effective n by ~3.6. A fold with 2,500 samples carries ~700 episodes ⇒ SE ≈ 1.1 points. **Treat anything under ~1 point as noise.**
6. **Secondary check:** leave-one-controller-out on 2–3 controllers who flew all three scenarios, mimicking the private LB's shape.
7. **Public LB is a third signal, weighted least** (915 samples, partly-seen controller). A public/CV gap that widens while CV is flat means you are fitting controller 020.
8. One `experiments.csv`: `run_id, desc, cv_mean, cv_std, per_fold, public_lb, features_hash, seed`.

---

## 6. How an agent should split and approach this

### 6.1 Principles

- **Harness before models.** One metric function, one fold file, one submission validator. Nothing starts until these exist and the 0.352 baseline reproduces.
- **One artifact format.** Every model writes `(sample_id, row_index, score)` parquet for out-of-fold rows and for test. Blending, smoothing and selection then operate on artifacts, so they can be built in parallel with the models.
- **Features as pure functions.** `build(df) -> DataFrame`, owner-prefixed columns (`f2_*`, `f3_*`, `f4_*`), no cross-row computation outside `sample_id` or `session`. Makes the leakage audit mechanical.
- **Every family earns its place under leave-controller-out CV**, with a delta larger than the ~1-point noise floor.
- **Reproducibility is a rule.** Fixed seeds, pinned versions, a single `make submission` path from the first hour — rule 2.8.a requires handing it over.
- **Robustness beats peak.** The host says it, and the ±2-point private noise band proves it.

### 6.2 Workstreams

Nine streams. A1 → A2 are serial; A3–A6 run in parallel against the frozen harness; A7–A8 consume artifacts; A9 runs continuously.

| ID | Workstream | Deliverable | Depends on |
|---|---|---|---|
| **A0** | **Orchestrator** — schedule, 5/day budget, experiment log, final-submission choice, jury-story owner | decisions + `experiments.csv` | — |
| **A1** | **Harness** — loaders, `atco` fold file, `scorer.py` wrapper, submission writer/validator, repro entry point | frozen `harness/`, 0.352 reproduced | — |
| **A2** | **EDA & label analysis** — `P(target\|task_index)`, episode segmentation and count, episode-length distribution, `seq_len`, missingness map, `l_twin_AE` frequency + the 3:1 assume split verified, label persistence across consecutive snapshots, per-controller/per-scenario label mix, dirty-value audit | `FINDINGS.md` of numbers that justify every later feature | A1 |
| **A3** | **Baseline ranker** — LightGBM `lambdarank` on raw features | first real submission + OOF artifact | A1 |
| **A4** | **Within-sample features** (Tier 2) | `f4_*` module + CV delta | A1, A2 |
| **A5** | **Temporal features + smoothing** (Tier 3) | `f5_*` module, tuned EWMA half-life, CV delta | A1, A2 |
| **A6** | **Scenario priors** (Tier 4, timeboxed 1 h) | ship/drop verdict with evidence | A1, A2 |
| **A7** | **Model zoo & blending** — CatBoost, XGBoost, set-transformer w/ group softmax; per-sample normalisation; CV-weighted blend; calibration | blended test scores + calibration report | A3–A6 |
| **A8** | **Post-processing & selection** — temporal smoothing, twin coverage (assume first), top-3 policy, CSV assembly | `submission.csv` candidates | A7 |
| **A9** | **Verification, compliance & jury pack** — submission validity, leakage audit (no `atco`/`session` creeping in), fold integrity, clean-clone reproduction, MIT/GPL-3/BSD-3 headers, method description, **finalist slides** | sign-off checklist + presentation | all |

### 6.3 Interface contract

```
data/                              raw CSVs, read-only
harness/folds.csv                  sample_id, session, atco, scenario, fold      <- frozen
harness/metric.py                  top3_hit(labels_df, pred_df) -> float
harness/validate_sub.py            asserts every test sample_id present exactly once, ids numeric
features/f4_within_sample.py       build(df) -> DataFrame   # index-aligned, f4_* only
artifacts/<run_id>_oof.parquet     sample_id, row_index, score
artifacts/<run_id>_test.parquet    sample_id, row_index, score
experiments.csv                    run_id, desc, cv_mean, cv_std, per_fold, public_lb, seed
```

A stream reads only `data/`, `harness/` and its own module. Cross-stream communication happens through artifacts and `experiments.csv` — never through a shared mutable notebook.

### 6.4 Schedule against the real agenda

**Day 1 — Tue 14:00–21:00 (7 h). Goal: a legitimate model on the board and all feature work underway.**

| Time | Focus | Exit criterion |
|---|---|---|
| 14:00–15:00 | A1 harness; A2 starts EDA. Team formed on Kaggle **before any submission**. | metric reproduces, folds frozen, 0.352 reproduced |
| 15:00–16:30 | A3 `lambdarank` on raw features. **Submission #1** (1/5). | CV recorded; LB anchor |
| 16:30–19:00 | A4 and A5 in parallel; A6 timeboxed and reported. | each reports a signed CV delta > 1 pt or is dropped |
| 19:00–20:30 | Merge winning features, retrain. **Submission #2** (2/5). Start A7 (second model family). | CV improvement confirmed |
| 20:30–21:00 | **Checkpoint:** commit everything, write the state of play, queue overnight-safe long runs only if infrastructure allows. Catch the bus. | clean repo, written handover note |

**Day 2 — Wed 08:00–13:00 (5 h). Budget resets at 02:00 → 5 fresh submissions. Goal: blend, verify, select.**

| Time | Focus | Exit criterion |
|---|---|---|
| 08:00–09:30 | A7 model zoo + blend; A8 smoothing and twin policy. | blend beats best single model on CV |
| 09:30–10:30 | **Submissions #1–2 of the day** on the two best *distinct* configurations. | two live candidates with public scores |
| 10:30–11:30 | Retrain the chosen config on all 37 labelled sessions; A9 full audit. | clean-clone reproduction passes |
| 11:30–12:15 | **Final submission** (keep ≥1 slot spare). | uploaded and validated |
| 12:15–12:30 | **Select the final submission(s) manually, by CV.** Freeze code. | selection confirmed in the UI |
| 12:30–13:00 | Method description + jury slides; rehearse the 10-minute story. | pack ready before jury evaluation |

Never leave a late discovery with no slot to test it: hold one submission in reserve on each day.

### 6.5 Final-submission decision rule

Choose by **best leave-controller-out CV mean penalised by spread** (`mean − 0.5·std`), breaking ties toward the simpler, better-calibrated, more-blended model. Use the public LB only to confirm nothing is catastrophically broken. **Do not pick the public-LB maximum:** public = controller 020 (partly in your training data), private = controller 023 (not), and the public differences between close submissions are inside the noise band. If the UI allows two final submissions, pick the CV-best and a genuinely *different* model family rather than two near-identical variants — that is a hedge against the ±2-point private wobble. **If you select nothing, Kaggle picks your best public score for you.**

### 6.6 The jury, not just the leaderboard

Finalists present at 13:30. Prepare, in this order: the ranking framing and why it is a ranking problem; the controller-wise validation and the 25–45-point trap it avoids; the two or three features that actually moved CV, with numbers; the human-noise ceiling (6.5 % inter-rater) and what it implies for a deployable Level 2A advisor; reproducibility (one command, pinned environment, open licence); and honest limits. A clear account of *why* the model generalises to a new controller will read better to this jury than a 0.005 leaderboard edge.

---

## 7. Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| Random-split illusion (host measured +25–45 pts) | CV ≫ public LB | GroupKFold by `atco`, frozen fold file, A9 audit |
| Chasing noise | decisions made on < 1-point deltas | episode-based error bars; ignore sub-noise gains |
| Overfitting the public LB | public rises while CV flat/falls | weight CV; cap LB-driven decisions |
| Deadline confusion (13:00 vs 14:00 CEST) | — | assume 13:00; everything in by 12:30 |
| Auto-selected final submission | nothing selected in the UI | A0 checklist, done by 12:30 Wed |
| Submissions counted against a half-formed team | Kaggle merge sums entries | form the full Kaggle team before the first submission |
| `fillna(0)` corrupting semantics | odd importances, silent loss | NaN-native models + `_isnan` flags; A9 greps for it |
| `atco`/`session` leaking in | implausible fold scores, id-like column high in importance | A9 column + importance audit before every submission |
| Scenario memorisation (Tier 4) | gains only on same-scenario folds | leave-controller-out **and** leave-scenario-out; drop by default |
| Invalid submission burning a slot | Kaggle rejects the file | `validate_sub.py` inside the submit script |
| Non-reproducible winning run | cannot hand code to the Organisers ⇒ disqualified, place passes down | single `make submission`, pinned env, fixed seeds, clean-clone test Wed morning |
| Grinding past the ceiling | CV plateaus | stop featuring; spend the time on blending, calibration and the jury story |
| External data temptation | any non-provided data | banned (rule 2.6.a); LLMs/libraries/AutoML are fine |
| Overnight gap treated as work time | plan assumes 20 continuous hours | the schedule above; hard checkpoint at 20:30 Tue |

---

## 8. Quick reference — the 14 task types and their drivers

| Code | Type | Group | Likely drivers |
|---|---|---|---|
| 4 | `assume_aircraft` | entry | `x_transfer_in`, `is_not_assumed`, `x_state_age`, `a_last_assume_control` |
| 2 | `entry_coordination` | entry | `x_isc_notified`, `x_isc_coordinated`, `x_entry_coord_not_done` |
| 9 | `entry_conditions` | entry | very rare — check the prior, consider excluding |
| 5 | `transfer_aircraft` | exit | `k_aircraftNeedsToBeTransferred`, `k_transferToolActivation`, `k_isAtExitFL` |
| 3 | `exit_coordination` | exit | `k_out_NOTIFIED`, `k_out_COORDINATED`, `k_outCoordMismatch`, `k_coordinationToolActivation` |
| 10 | `exit_conditions` | exit | `k_hasToClimbToExitFL`, `k_hasToDescendToExitFL`, `exit_FL`, `k_isAtExitFL` |
| 6 | `conflict_resolution` | conflict | `conflict_ind`, `time_to_MD`, `HMI_conflict_safety_alert`, `multiple_conflicts_ind` |
| 0 | `entry_conflict_resolution` | conflict | `CPA_entry_location`, `approaching_sector`, `time_to_MD` |
| 1 | `exit_conflict_resolution` | conflict | `CPA_exit_location`, `k_distanceToExitPoint` |
| 12 | `zone_conflict` | military zone | `zone_conflict_ind`, `distance_to_zone` |
| 13 | `return_to_route` | route | `non_conformance_agreed_route_deviation_horizontally`, `monitoring_rule_activated` |
| 7 | `non_conformance_resolution` | route | `monitoring_rule_activated`, `non_conformance_*`, `k_isFlightLevelBust` |
| 8 | `qos_improvement` | QoS | `qos_candidate` |
| 11 | `aircraft_request` | requests | **does not occur in this data** |

Across all types, the `a_` input-log family (`a_since_flight`, `a_is_last_cs`, `a_is_last2_cs`) carries the "what is the controller doing right now" signal, `x_task_age` separates an aircraft's competing tasks, and the `l_` family is the only group that already compares a task against its rivals. `transfer_aircraft` + `assume_aircraft` are half the labels.

### Acronyms

ATC/ATCO/ATM · AWARE (artificial situational awareness system + project) · CFL cleared flight level · CPA closest point of approach · ESMM Malmö ACC (sector ESMM_23) · FL flight level · HMI human-machine interface · ICC intraclass correlation · ISC inter-sector coordination · MD minimum distance · NM nautical mile (1,852 m) · QoS quality of service · STCA short-term conflict alert · TRA temporary reserved area (military) · UkSATSE Ukrainian air traffic service provider · UTC

---

**Sources:** `AWARE_prioritisation_challenge.pdf` (challenge briefing deck, 34 slides) · [Kaggle Overview](https://www.kaggle.com/competitions/engage-ktn-2026-competition/overview) · [Data & features](https://www.kaggle.com/competitions/engage-ktn-2026-competition/data) · [Rules](https://www.kaggle.com/competitions/engage-ktn-2026-competition/rules) · [Leaderboard](https://www.kaggle.com/competitions/engage-ktn-2026-competition/leaderboard)
