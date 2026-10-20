# ATCO prioritiser - ENGAGE KTN 2026 AWARE task-prioritisation baseline

For each 3 s snapshot of an en-route controller's open-task list (sector ESMM_23), rank the tasks so the one
the controller is actually working on lands in the top 3. Metric: top-3 hit rate per sample.
The full brief and plan are in [`docs/agent-plan.md`](docs/agent-plan.md); data facts in
[`FINDINGS.md`](FINDINGS.md).

This repo is **Tier 0 + Tier 1** of that plan (a frozen harness and honest baselines, validated
leave-controller-out - the private LB is an unseen controller), plus first Tier-2 and cost-function experiments.

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
| **`b1_lgbm_raw`** | **LightGBM lambdarank, 87 raw features** | **0.659 ± 0.021** | **0.648** | .694 .663 .651 .642 .644 |
| `b1_lgbm_raw_epool` | b1 + per-episode probability pooling | 0.666 ± 0.026 | 0.653 | .710 .665 .659 .644 .652 |
| `b2_lgbm_f2` | b1 + 62 within-snapshot relational features (deck hints) | 0.641 ± 0.029 | 0.627 | .680 .658 .640 .617 .611 |
| `b3_softmax_raw` | cost function (listwise softmax CE), regularisation picked on the hold-out | 0.646 ± 0.027 | 0.632 | .685 .649 .654 .622 .619 |

Pooled episode-level SE is ≈ 0.85 pt and fold spread ≈ 2 pt, so b1 vs b1+epool (+0.7 pt) is noise. Per the plan's
tie rule (simpler wins) **`b1_lgbm_raw` is the baseline submission**. All runs are logged in
[`experiments.csv`](experiments.csv) (`split` = `cv5` or `holdout`); fill in `public_lb` by hand after each
Kaggle upload.

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
make submission   # metric check -> folds check -> b1 CV + full refit -> submissions/submission.csv (~2 min, 4 cores)
make all          # also b0 and the episode-pooling variant
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
models/heuristics.py           b0 - reproduces sample_submission exactly (asserted)
models/lgbm_ranker.py          b1 - LightGBM lambdarank, group = sample_id, NDCG@3 (--objective softmax for the cost function)
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

1. **Find out why the relational features hurt.** `f2_within_sample` (62 features) loses 1.8 pt on every fold
   and under every objective / regularisation tried, from round 25 on - it is not late overfitting. Next:
   add its groups one at a time (snapshot pct ranks, type pct ranks, type/aircraft age deltas, sibling types,
   twin, snapshot size) to find the harmful group rather than dropping the whole idea.
2. **Tier 3 - temporal**: per `(session, task_name)` deltas vs previous snapshots, time since task appeared,
   and short-window smoothing. Whole-run pooling only bought +0.7 pt because 13 % of runs hold more than one
   label - see FINDINGS.md.
3. CatBoost / set-model diversity and a CV-weighted blend on per-sample probabilities (the cost-function model
   is the calibrated member).
4. Before the final: pick an open licence and add method description (rule 2.8.a).
