# ATCO prioritiser - ENGAGE KTN 2026 AWARE task-prioritisation

For each 3 s snapshot of an en-route controller's open-task list (sector ESMM_23), rank the tasks so the one
the controller is actually working on lands in the top 3. Metric: top-3 hit rate per sample.
The full brief and plan are in [`docs/agent-plan.md`](docs/agent-plan.md); data facts in
[`FINDINGS.md`](FINDINGS.md).

This repo holds **Tier 0 + Tier 1** of that plan (a frozen harness and honest baselines, validated
leave-controller-out - the private LB is an unseen controller), Tier-2 and cost-function experiments, and the
**Tier-3 session-temporal features that take the CV from 0.659 to 0.923** - see FINDINGS.md for the evidence and
the look-ahead caveat (they read later snapshots of the session; a past-only variant scores like b1).

## Validation: two splits, both by controller

- **5-fold CV** (`harness/folds.csv`) - GroupKFold by `atco` over train + validation pooled. The main yardstick:
  every decision below is made on it.
- **Train / stop / eval hold-out** (`harness/holdout.csv`) - eval = controllers 5, 17, 18 (seeded pick among the
  five who flew all three scenarios: the private LB's shape), stop = 15, 16 (the other two), train = the other 13.
  The cost function is monitored on *stop* to choose rounds and regularisation; *eval* is scored once.

## Results - 5-fold CV (GroupKFold by controller)

| run_id | model | CV mean ± fold std | mean − 0.5·std | per fold |
|---|---|---|---|---|
| `b0_type_prior` | type prior + nearest boundary (= `sample_submission`, 0.352 public) | 0.390 ± 0.033 | 0.374 | .444 .371 .362 .399 .375 |
| `b1_lgbm_raw` | LightGBM lambdarank, 87 raw features | 0.659 ± 0.021 | 0.648 | .694 .663 .651 .642 .644 |
| `b1_lgbm_raw_epool` | b1 + per-episode probability pooling | 0.666 ± 0.026 | 0.653 | .710 .665 .659 .644 .652 |
| `b2_lgbm_f2` | b1 + 62 within-snapshot relational features (deck hints) | 0.641 ± 0.029 | 0.627 | .680 .658 .640 .617 .611 |
| `b3_softmax_raw` | cost function (listwise softmax CE), regularisation picked on the hold-out | 0.646 ± 0.027 | 0.632 | .685 .649 .654 .622 .619 |
| `b4_lgbm_f3` | b1 + 41 session-temporal features (f3: task vanishes, next HMI input on the aircraft, run geometry) | 0.909 ± 0.028 | 0.895 | .955 .893 .885 .894 .919 |
| `b4_lgbm_f3_epool` | b4 + per-episode probability pooling | 0.906 ± 0.032 | 0.891 | .960 .891 .878 .895 .909 |
| `b5_lgbm_f3f4` | b4 + f4 v1 (state-change flags by the next run only) | 0.909 ± 0.021 | 0.898 | .943 .891 .894 .910 .906 |
| `b6_lgbm_f2f3` | b4 + f2 relational features | 0.905 ± 0.026 | 0.892 | .941 .884 .878 .899 .921 |
| `b7_stack_f3` | session stacking on b4 scores (nested CV) | 0.913 ± 0.025 | 0.900 | .954 .896 .893 .902 .920 |
| **`b8_lgbm_f345`** | **raw + f3 + f4 (aircraft state changes) + f5 (task context), 600 rounds** | **0.923 ± 0.020** | **0.913** | .952 .898 .911 .923 .931 |
| `b10_stack_f345_expert` | session stacking on b8-type stage 1 + silent-target expert (nested CV) | 0.923 ± 0.025 | 0.911 | .958 .898 .899 .929 .932 |
| `b11_lgbm_f3past` | b1 + the past-only part of f3 (no look-ahead) | 0.653 ± 0.024 | 0.641 | .689 .663 .641 .627 .644 |
| `b12_lgbm_f345_bag3` | b8, seed bag of 3 | 0.922 ± 0.024 | 0.911 | .957 .893 .910 .921 .931 |

Pooled episode-level SE is ≈ 0.85 pt at b1's level and ≈ 0.5 pt at b8's, and a single run moves by ~0.3 pt
with column order / seed, so only b4 (+25 pt) and b8 (+1.4 pt over b4) are real steps. **`b8_lgbm_f345` is the
selected submission** (`make submission`); b10 and b12 tie it and are more complex. All runs - including the
`scan_*` learning-curve / parameter rows - are logged in [`experiments.csv`](experiments.csv) (`split` = `cv5`
or `holdout`); fill in `public_lb` by hand after each Kaggle upload.

## Results - hold-out (eval controllers 5, 17, 18)

| run_id | model | eval top-3 | eval cost | rounds |
|---|---|---|---|---|
| `ho_lambdarank_raw` | b1 setup, fixed 300 rounds | 0.629 | 2.241 | 300 |
| `ho_softmax_raw` | cost function, early-stopped on stop-set cost | 0.627 | 2.217 | 148 |
| `ho_lambdarank_raw_f2` | b1 setup + relational features | 0.622 | 2.236 | 300 |
| `ho_softmax_raw_f2` | cost function + relational features | 0.605 | 2.259 | 91 |
| `ho_softmax_tuned_raw` | cost function, regularisation with the lowest stop-set cost of 12 (`strong`) | 0.641 | 2.186 | 1067 |

Eval SE ≈ 1.75 pt. The cost function gives **better-calibrated** probabilities (lower eval cost) and, once
regularised, +1.2 pt on these three controllers - but the same model on the 5-fold CV (`b3_softmax_raw`) is
**−1.3 pt vs b1** (paired episode bootstrap 95 % CI −2.2 to −0.4; better on only 2 of 18 controllers, and worse
on 5, 17 and 18 too). A single 3-controller hold-out is too small to select on; the 5-fold CV decides, and b1
stays. The cost function is kept for calibrated per-sample probabilities (blending, Tier 5).

## Reproduce

```bash
# 1. data: unzip engage-ktn-2026-competition.zip into data/ (train.csv, validation.csv, test.csv,
#    sample_submission.csv, scorer.py). data/ is git-ignored.
make setup        # pinned deps, Python 3.13
make submission   # metric check -> folds check -> b8 CV + full refit -> submissions/submission.csv (~6 min, 4 cores)
make all          # also b0, b1, episode pooling, b4 (raw + f3) and the past-only variant b11
make holdout      # freeze/verify harness/holdout.csv, lambdarank vs cost function on the hold-out (~1 min)
make tune         # pick regularisation by stop-set cost, score the winner once on eval (~4 min)
```

Runs are deterministic (`seed=42`, `deterministic=True`, fixed threads): a rerun gives bit-identical files.

## Layout

```
data/                          raw competition files (read-only, not in git)
harness/config.py              paths, ID columns, forbidden feature columns, K, seeds
harness/data.py                loaders (sorted by sample_id, row_index), run/episode segmentation
harness/folds.py + folds.csv   frozen GroupKFold-by-atco assignment - never re-randomise
harness/holdout.py + .csv      frozen train / stop / eval controller split
harness/metric.py              top-3 hit; `python -m harness.metric` checks it equals data/scorer.py
harness/cv.py                  run_cv / report (per-fold + episode SE) / experiments.csv / artifacts
harness/submission.py          top-3 CSV writer + validator; validate_sub.py is the CLI
features/                      feature builders: build(df) -> DataFrame; FEATURE_SETS registry
  f2_within_sample.py          within-snapshot relational features (hurt; kept for the record)
  f3_temporal.py               session timeline: task presence before / after, per-aircraft HMI input log, runs
  f4_state_changes.py          aircraft state changes (clearances, coordination, conditions) around the snapshot
  f5_task_context.py           tasks of one aircraft, input-type match, inputs in the run, snapshot context
models/heuristics.py           b0 - reproduces sample_submission exactly (asserted)
models/lgbm_ranker.py          LightGBM lambdarank, group = sample_id, NDCG@3 (--objective softmax for the cost function,
                               --preset / --rounds / --seeds); b1, b4, b8 ...
models/stack_session.py        session-level stacking with nested CV (+ --expert: silent-target ranker)
models/blend.py                equal-weight softmax blend of artifacts (not used by the selected submission)
models/cost.py                 the cost function: listwise softmax cross-entropy (objective + eval metrics)
models/holdout_eval.py         train / early-stop on cost / eval-once runner; --tune selects regularisation
models/postprocess.py          episode pooling on artifacts
artifacts/<run_id>_{oof,test}.parquet   sample_id, row_index, score (+ _importance.csv)
submissions/<run_id>.csv       validated Kaggle files; submission.csv = the selected one
```

## Conventions

- **Ranking, not classification.** Score rows, rank within `sample_id`, take the top 3.
- **One artifact format.** Every model writes `(sample_id, row_index, score)` for OOF and test, so blending and
  post-processing work on artifacts, not on models.
- **Never feed `atco`, `session`, `scenario`, `timestamp`, `task_name` or `sample_id` as features.**
  `cv.check_features` refuses them.
- **Empty ≠ 0.** No `fillna(0)`; LightGBM handles NaN natively.
- **Every change earns its place by CV delta > ~1 pt**, not public LB (public = controller 020, partly seen).

### Adding a feature set

```python
# features/f2_within_sample.py
def build(df):            # only look across rows of the same sample_id (or session, for temporal features)
    ...
    return out            # index-aligned with df, columns prefixed f2_

# features/__init__.py
FEATURE_SETS["raw+f2"] = [f0_raw.build, f2_within_sample.build]
```

Then `python -m models.lgbm_ranker --features raw+f2 --run-id b2_lgbm_f2`.

## Next steps (from the plan, informed by the experiments)

1. **Silent tasks** are what is left: targets whose task neither closes nor gets an input before the next run
   (15 % of samples, hit 0.61 vs 0.975 elsewhere); entry + exit coordination are half of b8's misses. Only priors
   can rank them - e.g. a coordination-specific ranker on the b1-style features, or the aircraft task-set change
   timing (`l_n_flight` changes within 120 s for 45 % of silent targets vs 24 % of rivals; prototyped, not
   validated).
2. **A look-ahead horizon curve for the jury**: cap every future-looking feature at t + Δ (Δ = 3, 15, 30, 60 s)
   to show what a delayed advisor would reach; past-only is 0.653.
3. 63 leaves / 50 min rows is +0.4 pt in the scan (noise-level); a paired multi-seed check could adopt it.
4. Before the final: pick an open licence and add method description (rule 2.8.a).
