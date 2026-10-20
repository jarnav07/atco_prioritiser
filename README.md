# ATCO prioritiser - ENGAGE KTN 2026 AWARE task-prioritisation baseline

For each 3 s snapshot of an en-route controller's open-task list (sector ESMM_23), rank the tasks so the one
the controller is actually working on lands in the top 3. Metric: top-3 hit rate per sample.
The full brief and plan are in [`docs/agent-plan.md`](docs/agent-plan.md); data facts in
[`FINDINGS.md`](FINDINGS.md).

This repo is **Tier 0 + Tier 1** of that plan: a frozen harness and honest baselines, validated
leave-controller-out (the private LB is an unseen controller).

## Results (GroupKFold by controller, 5 folds, train + validation pooled)

| run_id | model | CV mean ± fold std | mean − 0.5·std | per fold |
|---|---|---|---|---|
| `b0_type_prior` | type prior + nearest boundary (= `sample_submission`, 0.352 public) | 0.390 ± 0.033 | 0.374 | .444 .371 .362 .399 .375 |
| **`b1_lgbm_raw`** | **LightGBM lambdarank, 87 raw features** | **0.659 ± 0.021** | **0.648** | .694 .663 .651 .642 .644 |
| `b1_lgbm_raw_epool` | b1 + per-episode probability pooling | 0.666 ± 0.026 | 0.653 | .710 .665 .659 .644 .652 |

Pooled episode-level SE is ≈ 0.85 pt and fold spread ≈ 2 pt, so b1 vs b1+epool (+0.7 pt) is noise. Per the plan's
tie rule (simpler wins) **`b1_lgbm_raw` is the baseline submission**. All runs are logged in
[`experiments.csv`](experiments.csv); fill in `public_lb` by hand after each Kaggle upload.

## Reproduce

```bash
# 1. data: unzip engage-ktn-2026-competition.zip into data/ (train.csv, validation.csv, test.csv,
#    sample_submission.csv, scorer.py). data/ is git-ignored.
make setup        # pinned deps, Python 3.13
make submission   # metric check -> folds check -> b1 CV + full refit -> submissions/submission.csv (~2 min, 4 cores)
make all          # also b0 and the episode-pooling variant
```

Runs are deterministic (`seed=42`, `deterministic=True`, fixed threads): a rerun gives bit-identical files.

## Layout

```
data/                          raw competition files (read-only, not in git)
harness/config.py              paths, ID columns, forbidden feature columns, K, seeds
harness/data.py                loaders (sorted by sample_id, row_index), run/episode segmentation
harness/folds.py + folds.csv   frozen GroupKFold-by-atco assignment - never re-randomise
harness/metric.py              top-3 hit; `python -m harness.metric` checks it equals data/scorer.py
harness/cv.py                  run_cv / report (per-fold + episode SE) / experiments.csv / artifacts
harness/submission.py          top-3 CSV writer + validator; validate_sub.py is the CLI
features/                      feature builders: build(df) -> DataFrame; FEATURE_SETS registry
models/heuristics.py           b0 - reproduces sample_submission exactly (asserted)
models/lgbm_ranker.py          b1 - LightGBM lambdarank, group = sample_id, NDCG@3
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

## Next steps (from the plan, informed by the baseline)

1. **Tier 2 - within-sample relational features**: rank / pct / diff-to-max / z-score of every numeric column
   within `sample_id` and within `(sample_id, task_index)`; rank among the tasks of the same aircraft
   (callsign = `task_name` suffix); explicit assume/entry twin pairing (assume wins 73 % of twins). b1's
   top gains are `task_index`, `distance_to_sector`, `time_to_sector`, `x_task_age`, `a_since_flight` - all
   raw values that need the relational view.
2. **Tier 3 - temporal**: per `(session, task_name)` deltas vs previous snapshots, time since task appeared,
   and short-window smoothing. Whole-run pooling only bought +0.7 pt because 13 % of runs hold more than one
   label - see FINDINGS.md.
3. Objective variants (`rank_xendcg`, binary + per-sample softmax), then CatBoost / set-model diversity and a
   CV-weighted blend.
4. Before the final: pick an open licence and add method description (rule 2.8.a).
