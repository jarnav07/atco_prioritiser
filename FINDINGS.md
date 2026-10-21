# Findings (A2 - EDA & label analysis)

Numbers computed on train + validation pooled (37 sessions, 18 controllers, 13,175 samples, 474,688 rows)
unless stated. They justify the harness and baseline choices; extend this file as features are added.

## Splits

| | samples | sessions | controllers | scenarios |
|---|---|---|---|---|
| train.csv | 11,923 | 34 | 1–11, 14–18 | 1, 2, 3 |
| validation.csv | 1,252 | 3 | 19, 20 | 1, 2 |
| test.csv | 1,863 | 5 | 20 (public: `020sc2`, `020sc3`), 23 (private) | 1, 2, 3 |

Controller 20 is partly seen (`020sc1` is in validation), controller 23 is unseen. Hence CV is
**GroupKFold by `atco`** over the pooled labelled set (`harness/folds.csv`, 5 folds, 3–4 controllers each).

## The sample submission

`sample_submission.csv` is exactly: per-row hit rate of the task type in train.csv (positives / rows), ties
broken by smallest `|time_to_sector|`, then lowest `row_index` - 100 % agreement on all 1,863 test samples.
`models/heuristics.py` asserts this, which proves loader → scores → top-3 → CSV end to end.

## Labels by task type

| code | type | labels | share | rows | P(target \| row of this type) |
|---|---|---|---|---|---|
| 5 | transfer_aircraft | 3,527 | 26.8 % | 30,231 | 11.7 % |
| 4 | assume_aircraft | 3,366 | 25.5 % | 78,904 | 4.3 % |
| 12 | zone_conflict | 2,121 | 16.1 % | 53,109 | 4.0 % |
| 2 | entry_coordination | 1,068 | 8.1 % | 141,540 | 0.8 % |
| 13 | return_to_route | 937 | 7.1 % | 16,542 | 5.7 % |
| 3 | exit_coordination | 745 | 5.7 % | 60,207 | 1.2 % |
| 10 | exit_conditions | 448 | 3.4 % | 36,656 | 1.2 % |
| 6 | conflict_resolution | 390 | 3.0 % | 11,072 | 3.5 % |
| 8 | qos_improvement | 257 | 2.0 % | 29,918 | 0.9 % |
| 1 | exit_conflict_resolution | 197 | 1.5 % | 6,080 | 3.2 % |
| 7 | non_conformance_resolution | 86 | 0.7 % | 1,446 | 5.9 % |
| 0 | entry_conflict_resolution | 31 | 0.2 % | 8,852 | 0.4 % |
| 9 | entry_conditions | 2 | 0.0 % | 131 | 1.5 % |
| 11 | aircraft_request | 0 | – | 0 | – |

Transfer + assume = 52 % of labels.

## Structure

- `seq_len` 9–59, mean 36.0 (test 35.6). Random top-3 = 0.091.
- **Assume/entry twin prior verified:** of 3,929 positives on an `l_twin_AE = 1` aircraft that are assume or
  entry coordination, **73.2 % are assume**.
- `l_rank_absf1_type` is the within-type rank of `|time_to_sector|` (`f1`), `l_rank_f0_type` of
  `distance_to_sector`. So "prior + `l_rank_absf1_type`" is the sample submission, not a new baseline.
- `a_is_last_cs = 1` lifts P(target) from 2.8 % (base row rate) to 8.3 %; but only 11 % of positives have it.
- `k_` columns are 61 % empty for unassumed aircraft vs 10 % for assumed - missingness carries signal, kept
  as NaN (LightGBM routes it natively).
- `non_conformance_agreed_route_deviation_horizontally` > 100 km on 365 rows (max 167 km) - the "may be data
  errors" values. Left as is for the tree baseline.

## Time structure: only labelled moments are in the files

Snapshots are 3 s apart, but **samples exist only while a labelled task is being worked**, so a session has
gaps. Label persistence between consecutive samples of a session:

| gap to previous sample | pairs | same label |
|---|---|---|
| ≤ 4 s | 9,053 | **94.6 %** |
| 4–6 s | 241 | 2.1 % |
| > 6 s | 2,595 | ~0 % |

A run of back-to-back snapshots (gap ≤ 4.5 s) is computable on test from timestamps alone. 3,128 runs in the
labelled set (mean 4.2 snapshots, median 3), 401 in test. But runs are **not** clean episodes:

- 13.4 % of runs contain more than one label (27 % of samples sit in such runs): controllers switch tasks
  without a gap.
- In 8.3 % of cases the true task is missing from some snapshots of its own run.
- The b1 hit rate falls with run length: 0.83 on single-snapshot runs, 0.56–0.61 on runs of 5+ snapshots.

Episode pooling of b1 scores (`models/postprocess.py`, mean softmax probability per (run, task) over the
snapshots where the task is listed) gives +0.7 pt - inside the noise floor. Treating absent as probability 0
*costs* 1.4 pt. Deployment caveat: live AWARE emits every 3 s regardless, so this gap signal does not exist
there.

## Session-temporal context: the missing signal family (0.659 → 0.923)

**Read this first: the whole gain is look-ahead.** The features below read *later* snapshots of the session.
Restricted to the current and earlier snapshots they score 0.653 (`b11_lgbm_f3past`), no better than b1 (0.659).
Legal for the competition (no label is read; the test file is given whole), unavailable to a live system - see
"Look-ahead" below.

The 0.66 → 0.93 gap to the leaderboard was a missing family, not tuning: **what happens to each task and each
aircraft in the later snapshots of the same session.** A worked task is usually *finished* - an assumed
aircraft's assume task closes, a transferred aircraft leaves - and working it usually means an HMI input on its
aircraft, which later snapshots reveal as a reset of `a_since_flight` (with `a_last_*` saying which input). Every
snapshot thus discloses the time of the last input on each listed aircraft; collected over a session they give
the input log per aircraft, so each row can be given the time to the *next* input on its aircraft.

Raw signal on the labelled set (target row vs any row):

| signal | target | any row |
|---|---|---|
| task absent from the next snapshot | 19.7 % | 2.0 % |
| task absent from the first snapshot of the next run | **62.5 %** | 7.3 % |
| median time until the task is last seen in the session | 9 s | 297 s |
| median time to the next HMI input on the task's aircraft (when there is one) | 10 s | 215 s |
| that input comes before the next run starts | **80.8 %** | 11.8 % |
| the aircraft gets the soonest next input of its snapshot (ties included) | 50.4 % of samples | - |

Single-feature rankers: "task last seen soonest" 0.561, "soonest next input on the aircraft" 0.479 (b0: 0.390).
Per target type: assume - input within 30 s 91 %, gone at the next run 99 %; transfer - gone 99 % (its input is
rarely observed: the aircraft leaves the list); zone conflict - input within 30 s 69 % (84 % of them lateral
clearances), task usually persists (12 % gone); return to route 79 % (93 % lateral); exit conditions 77 % (83 %
level); entry coordination 20 % / 9 % gone; exit coordination 37 % / 10 % gone. For tasks that persist, the
aircraft *state* moves instead: before the next run, zone-conflict targets get `k_isClearedHeadingAssigned` /
`k_isLocked` changes 60 % vs 1–4 % for other zone tasks, exit-conditions targets a new `k_clearedFL` 64 % vs 2 %,
conflict targets 51 % vs 3 %, exit-coordination targets a coordination-tool change 28 % vs 8 %.

### Feature modules (all keyed within one session, built on each file separately)

- `features/f3_temporal.py` (41 columns): run geometry (position, length, time to run end, gaps); task presence
  in the previous / next snapshot, last snapshot of the previous run, first snapshot of the next run; presence
  streak end, time to vanish, first / last seen; the per-aircraft input log - time to the next / since the last
  input (both callsigns of a conflict), which input, inside this run?, before the next run?, inputs in the next
  60 s; aircraft still listed at the next run; within-snapshot and within-type ranks of the key times.
- `features/f4_state_changes.py` (74 columns): 27 aircraft-state columns (clearances, coordination flags,
  exit / zone / conflict conditions; checked identical across an aircraft's tasks in a snapshot) - did each change
  by the next run, since the previous run, and the time to the next change of each column and column group at any
  later appearance.
- `features/f5_task_context.py` (16 columns): separating the tasks of one aircraft (how many of its tasks vanish,
  rank of this one), does the next input match the task type (assume ↔ assume, transfer ↔ transfer, zone / RTR ↔
  lateral, exit conditions / conflicts / QoS ↔ level), inputs inside the run window and before the next run, tasks
  the aircraft gains by the next run, and snapshot context (how many tasks vanish / get an input at all).

| run_id | features | CV mean ± fold std | Δ vs b1 |
|---|---|---|---|
| `b4_lgbm_f3` | raw + f3, 300 rounds | 0.909 ± 0.028 | +25.0 pt |
| `b5_lgbm_f3f4` | raw + f3 + f4 v1 (group-level change flags only) | 0.909 ± 0.021 | +25.0 |
| scan | raw + f3 + f5, 300 / 600 / 900 rounds | 0.914 / 0.917 / 0.918 | +25.5 / +25.8 / +25.9 |
| **`b8_lgbm_f345`** | **raw + f3 + f4 + f5, 600 rounds** | **0.923 ± 0.020** | **+26.4** |

b8 by target type (b1 → b4 → b8): transfer .86 → .99 → .99, assume .83 → .98 → .99, zone .72 → .91 → .94, entry
coordination .40 → .72 → .75, return to route .49 → .88 → .90, exit coordination .09 → .63 → .66, exit conditions
.16 → .84 → .87, conflict .42 → .82 → .87, QoS .17 → .87 → .91. Per controller b8 ranges 0.77 (controller 1, one
session full of rare types) to 0.97.

### Leakage audit (CV jumped above 0.85, so this was checked before anything else)

- No builder reads `target_index`: f3, f4, f5 and the stack's stage-2 context are **bit-identical** when built on
  a frame with `target_index` dropped or randomly permuted (asserted).
- Every feature is computed within one session (`session`, `t_s`, `task_name`, callsigns are grouping keys only;
  `cv.check_features` still refuses them as inputs); folds are by controller, so a session never straddles train
  and validation. Test features are built from `test.csv` alone.
- Test looks like the labelled data on these features: task gone at the next run 7.3 % vs 8.2 % of rows, input
  before the next run 11.8 % vs 12.7 %, the same gap distribution (median 3 s, 75 % 4 s).
- Sanity check only (not used in any model): the test top-1 type mix moved from b1's 876 transfers / 6 exit
  coordinations to b8's 573 / 73, towards the counts implied by the deck (516 / 112).

### Look-ahead: this is a retrospective labeller, not a live predictor

**Almost every f3/f4/f5 column uses future snapshots of the session.** That is legal here (the whole test file
is given and no label is read), but a live AWARE system would not have them. Measured: the backward-looking
subset of f3 (`raw+f3past`: run position, gaps to the past, presence in the previous snapshot / run, time since
first seen, time since the last input) scores **0.653 ± 0.024 (`b11_lgbm_f3past`) - no better than b1 (0.659)**.
So the whole +26 pt is look-ahead: the model recognises the *effects* of the controller's work (the task closes,
the input lands, the clearance changes) after the fact. Even the backward-looking columns lean on the dataset's
run structure (samples exist only while a labelled task is worked), which a live 3 s feed would not have. For the
jury: the method answers "which task was the controller on at time t, given the next minutes of the log" - useful
for automatic annotation of recordings, and a delayed (t + Δ) advisor; a real-time Level 2A advisor would need the
past-only number (≈ b1) and gaze data.

### What is left: silent tasks

Call a target **silent** when its task is still listed at the next run and its aircraft gets no input before
then: 1,918 samples (15 %). Hit rate on silent targets b1 0.36 → b4 0.53 → b8 0.61, on the rest 0.71 → 0.973 →
0.975. The per-column "time to the next change at any later appearance" (f4 v2) is what moved silent targets
(+8 pt; e.g. `monitoring_rule_activated` changes later for 46 % of silent entry-coordination targets vs 9 % of
their same-type rivals). Entry + exit coordination are about half of b8's ≈1,020 misses (≈520); their timelines
show no state change at all around the labelled moment (task age 60–150 s, nothing in `k_out_*`, `x_isc_*`, cleared
level or tool flags), so only priors can rank them - which is where b1-style models top out.

**Last look-ahead attempts (deadline day), vs b8 0.923:**

| Run | Result |
|---|---|
| b8 + f8 aircraft task-set changes (`d2_lgbm_f3458`, 600 rounds) | 0.9226 ± 0.0233 - flat (folds .958 .901 .904 .918 .932); stopped at the 1 h limit before its test refit, so no submission file |
| b8 + f7 cross-controller consensus (`d1_consensus_f345`) | 3 of 5 folds before the 1 h limit: .949 .904 .917 vs b8 .952 .898 .911 (+0.3 pt, noise); not logged |
| blends of b8 with the causal consensus models | 0.905-0.923 - no gain |

Both runs shared 4 cores and ran ~3x slower than alone; b8 stays the submission.

## Causal track: current and earlier snapshots only (0.659 → 0.673)

If predictions must be made snapshot by snapshot, everything that reads later snapshots (f3 / f4 / f5 look-ahead)
is out. `harness/causal.py` makes the rule mechanical: build features on whole sessions and on sessions cut at
random times; every row up to the cut must get identical values (raw+f3past, raw+f2, raw+f6 and the scenario
clock pass; full f3 fails with its 29 look-ahead columns).

**Past-only signals are weak** (labelled set, targets vs other tasks):

| Signal | Targets | Others | Note |
|---|---|---|---|
| input on the aircraft since this run started | 6.5 % | 1.5 % | P(target \| flag) 0.11; inputs land at the *end* of an episode (1.4 % at run position 0, 19.5 % at 6+) |
| coordination state changed during this run | 2.8 % | 0.5 % | lift 5.4, tiny coverage |
| coordination state changed 0–10 s before the run started | 3.6 % | 1.0 % | lift 3.5 |
| task appeared shortly before the run started | – | – | no lift (0.5–0.8) |
| same aircraft as the previous episode | 6.2 % | – | the episode sequence carries little |
| tool flags (`k_transferToolActivation` on transfer tasks) | 39 % | 14 % | already raw columns |

b1's misses by type: exit coordination (hit 9 %, 675 misses), entry coordination (40 %, 645), zone conflict (72 %,
601), assume (83 %, 566), transfer (86 %, 500), return to route (49 %, 479), exit conditions (16 %, 377).

**Cross-controller consensus.** All controllers fly the same three scenarios; timestamps are wall-clock, so
`f7_consensus.scenario_clock` aligns sessions by aircraft sector-arrival times (t_s + time_to_sector): residuals
of ~1 s median across sessions. Estimated causally (running median over the aircraft seen so far). On that clock,
"how many other controllers worked this task within ±60 s" alone gives top-3 **0.48** (leave-one-controller-out
pool; type prior 0.39, chance 0.09); other controllers' input timing on the aircraft alone gives 0.24.

| Experiment (5-fold CV) | Result |
|---|---|
| raw + f6 (run timing, past inputs / state changes per aircraft, controller assume / transfer style) | 0.656 |
| raw + f3 past-only + f6 | 0.655 |
| raw + f7 consensus (fold-aware: held-out controllers only see training-fold labels) | 0.654 |
| raw + f7 consensus + other controllers' input timing | 0.658 |
| causal prefix pooling of b1 probabilities within the run (mean / EWMA, mixed or not) | 0.652–0.657 (raw 0.658) |
| **equal-weight blend b1 + raw+f6 + raw+f7 (one model per signal family)** | **0.673**, +1.4 pt, 95 % CI +0.7 to +2.2, 16 / 18 controllers better |
| best of all 127 blend subsets of 7 causal models | 0.674 (selected from 127 looks - not adopted) |
| 4-member blend adding the input-timing consensus model | 0.675 (noise) |

No causal feature family beats raw features alone - the snapshot already holds the geometry the consensus
mostly encodes - but the three models err differently, and the blend is the only clean causal step so far.
Caveats: f7 needs recordings of other controllers on the same traffic (a live sector never repeats), and f6's run
timing relies on the files keeping only labelled moments.

**Where the causal model fails: choosing the type, not the task.** On the c3 blend's out-of-fold scores:

| Diagnostic | Value |
|---|---|
| c3 top-3 hit | 0.673 |
| true task *type* among the types of its top 3 | 0.756 |
| oracle type: rank only the tasks of the true type | **0.975** |
| oracle aircraft: rank only the tasks of the true aircraft | 0.987 |
| implied type mass (softmax of scores) vs label share | transfer 0.07 vs 0.27; entry coordination 0.28 vs 0.08 |

Angles tried on that diagnosis (all causal, 5-fold CV unless noted):

| Angle | Result |
|---|---|
| two-stage: snapshot-level type classifier (per-type counts / urgency / flags, last input) x within-type softmax | type top-1 0.43, top-2 0.66; combined 0.50–0.53 - worse than the ranker's own type choice |
| diversified top 3 (force a second type / aircraft into the picks) | 0.628–0.672 (plain 0.673); the top 3 already span 3 types 47 % / 3 aircraft 86 % of the time |
| episode-level training weights (1/run length, 1/sqrt) | 0.650 / 0.652 (b1 0.659) - the metric weights snapshots, so long episodes matter |
| other controllers' type choice at the same scenario time | top-1 type 0.33–0.36 (always "transfer" 0.27) |
| CatBoost YetiRank, raw / raw+f6 (`models/catboost_ranker.py`) | 0.648 / 0.641; adding it to the c3 blend costs 0.6–0.7 pt |

Which *type* of task a controller picks next is largely their own choice: neither the snapshot nor the other
controllers on the same traffic predict it well (consistent with the deck's 6.5 % same-task agreement). Within the
right type the model is already near-perfect, so causal gains have to come from information about the controller's
current focus that these files do not carry (the real AWARE system uses gaze).

## From the briefing deck, checked against the data

`AWARE_prioritisation_challenge.pdf` (34 slides; not committed - slide 33 carries the private Kaggle invite).

| Deck claim | Check |
|---|---|
| 15,038 samples, 541,097 rows (slide 26) | 13,175 + 1,863 samples; 474,688 + 66,409 rows ✓ |
| A sample is a snapshot "every 3 s **while the controller works on a task**" (slide 26) | ✓ - the gap structure above is by design |
| ≈ 4,150 label episodes (slide 20) | 3,655 in the labelled set (new episode = gap > 4.5 s or label change) ⇒ ≈ 495 in test, in 401 runs |
| Assume chosen in ~3 of 4 twin cases (slide 24) | 73.2 % ✓ |
| BIX72G transferred before the nearer EWG5C (slide 25) | the answer is the **oldest open task of its type** 63.3 % of the time vs **nearest boundary** 57.6 % |
| Controllers return to tasks (slide 16) | only 1.9 % of labelled episodes revisit a task already worked in the session |
| Same task at the same moment across controllers 6.5 %; pairs < 30 s apart ordered at chance (slides 22–23) | not checkable from the files; sets the ceiling |
| Up to 2 final submissions; deadline Wed 13:00 CEST; daily limit resets 02:00 (slide 30) | - |

**Slide 20 reveals the test label mix.** Its per-type counts cover all 15,038 samples, so subtracting the
labelled counts gives the test counts exactly (they sum to 1,863):

| transfer | assume | zone | RTR | exit coord | entry coord | exit cond | conflict | exit conflict | QoS | entry conflict | non-conf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 516 | 410 | 373 | 142 | 112 | 99 | 82 | 65 | 29 | 25 | 7 | 3 |

Versus train, test has more zone conflicts (20.0 % vs 16.1 %) and fewer assume (22.0 % vs 25.5 %) and entry
coordination (5.3 % vs 8.1 %). Every team has the deck, but **this is test-label information: it is not used
in any model here.** Use it only as a sanity check on the predicted type mix of a submission.

## What did not work

| Idea (deck hint) | Result, leave-controller-out |
|---|---|
| Force the assume/entry twin pair into the top 3 (assume first) when the top pick is a twin | 0.634 vs 0.658 (−2.5 pt). The model already hits 70 % of twin labels; only 224 misses have the sibling in the top 3, and the forced pair pushes out a better third candidate |
| Within-snapshot relational features (`f2_within_sample`: snapshot / type pct ranks, age vs oldest of type and of aircraft, sibling types, twin, snapshot size) | 0.641 vs 0.659 (−1.8 pt), worse on all 5 folds |
| … with stronger regularisation, or the `rank_xendcg` objective | 0.641 / 0.640 at 500 rounds - still below raw at every checkpoint from round 25 |
| Cost function = listwise softmax cross-entropy (`models/cost.py`), rounds and regularisation chosen by lowest cost on the stop controllers | hold-out eval +1.2 pt (0.641 vs 0.629) but 5-fold CV −1.3 pt (0.646 vs 0.659; 95 % CI −2.2 to −0.4, better on 2 of 18 controllers). Better calibrated (eval cost 2.19 vs 2.24), not better ranked |
| Episode pooling on the temporal model (`b4_lgbm_f3_epool`) | 0.906 vs 0.909 - the run-level f3 features already carry what pooling did |
| f2 relational features on top of f3 (`b6_lgbm_f2f3`) | 0.905 vs 0.909 (worse on 3 of 5 folds) |
| f2 group by group on top of raw + f3 + f5 (scan rows `scan_raw+f3+f5+f2*`, at 300 / 600 / 900 rounds; reference 0.914 / 0.917 / 0.918) | snapshot pct ranks 0.913 / 0.915 / 0.915; type pct ranks 0.912 / 0.914 / 0.914; age deltas 0.915 / 0.913 / 0.916; sibling types 0.913 / 0.916 / 0.916; twin 0.914 / 0.917 / 0.918; snapshot size 0.912 / 0.917 / 0.917. No group helps (all within −0.35 … +0.05 pt); the two 23-column percentile-rank groups are the most negative, consistent with the original −1.8 pt being many weak, noisy columns rather than one harmful group |
| f4 v1: only "did the aircraft state change by the next run" flags (`b5_lgbm_f3f4`) | 0.909, flat: the flags are NaN whenever the aircraft leaves, duplicating f3's vanish signal, and clearance changes coincide with the inputs f3 already sees. v2 adds per-column time to the next change at any later appearance - that is what helps (b8) |
| Session stacking (`models/stack_session.py`): stage 2 on stage-1 scores turned into session context - (run, task) mean probability, best run-mean of the same task in earlier / later runs, how often it tops other runs - with nested CV | on raw + f3 (`b7_stack_f3`) 0.913 vs 0.909 (+0.4, 4 of 5 folds); on raw + f3 + f4 + f5 with a silent-target expert (`b10_stack_f345_expert`) 0.9232 vs 0.9229 - nothing left once stage 1 is strong. The stage-1 model predicts one task as top of more than one run 8 % of the time vs 2.9 % in truth, but fixing that rarely changes the top 3 |
| LightGBM settings on raw + f3 + f5 (scan rows): 15 leaves, feature_fraction 0.5, lambdarank truncation 10, `rank_xendcg`, binary logloss | all ≤ base at 300–1,200 rounds (best 0.914–0.915 vs 0.918). 63 leaves / 50 min rows is +0.4 on the full set in the same column order (0.924 at 900 rounds) - noise-level vs b8 |
| Seed bag of 3 (`b12_lgbm_f345_bag3`) | 0.9224 vs 0.9229 |
| `row_index` (the list order) | AWARE lists tasks by callsign, reverse-alphabetical; target relative position 0.531 vs 0.514 for all rows - a callsign proxy, not used |
| Label-type sequence across runs | P(type of the next run's label \| this run's) ≈ the marginal, except exit coordination → exit conditions (0.22 vs 0.03) - not used |
| Within-run switches | 27 % of samples sit in runs with > 1 label; first segments hit 0.835 (b7). Their task vanishes inside the run 44 % of the time, which the soonest-vanish / soonest-input ranks already encode |

Learning curves of the raw lambdarank baseline (5-fold CV): 0.630 / 0.638 / 0.652 / 0.656 / **0.659** / 0.652 at
25 / 50 / 100 / 200 / 300 / 500 rounds - the fixed 300 is at the peak. Stronger regularisation (15 leaves, 400
min rows, 0.5 colsample, L2 10) tops out at 0.649; `rank_xendcg` at 0.648.

## Noise floor

5 folds of 2,469–2,757 samples carry 575–723 runs each ⇒ per-fold SE ≈ 1.8–2.1 pt; pooled SE ≈ 0.85 pt.
Fold-to-fold spread of b1 is ±2.1 pt. **Treat any change under ~1 pt as noise.**

At the 0.92 level the pooled episode SE is ≈ 0.48 pt, but run-to-run noise is of the same size: the same
features in a different column order (which changes LightGBM's `feature_fraction` draws) gave 0.9196 (scan) vs
0.9229 (b8). Differences under ~0.5 pt between single runs at this level are not evidence.

Scan rows (`scan_*` in `experiments.csv`) are learning curves from a scratch script on cached features (same frozen
folds, same metric); 34 of the 52 kept only mean ± fold std (no per-fold scores). **Selection bias:** 52 scan
configurations were read off the same 5 folds, so the best scan value (0.924, 63 leaves at 900 rounds) is
optimistically selected by roughly the run-to-run noise (~0.3–0.5 pt); none of them was adopted. The selected
model's settings (base lambdarank, 600 rounds) were fixed from the raw + f3 + f5 learning curve and confirmed by the
CLI run b8; b8's 0.923 carries a little of that selection too (rounds picked on the same folds), on the order of
0.1–0.2 pt.
