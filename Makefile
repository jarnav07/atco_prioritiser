PY ?= python
# the selected submission (5-fold leave-controller-out CV 0.923 vs 0.659 for b1_lgbm_raw)
BASELINE := b8_lgbm_f345

.PHONY: all submission setup check folds b0 b1 epool temporal final select holdout tune

# every baseline + the temporal models + the chosen submission
all: check folds b0 b1 epool temporal final select

# the one-command path from data/ to submissions/submission.csv
submission: check folds final select

setup:
	$(PY) -m pip install -r requirements.txt

# our metric == data/scorer.py
check:
	$(PY) -m harness.metric

# writes harness/folds.csv once; afterwards only verifies it is unchanged
folds:
	$(PY) -m harness.folds

b0:
	$(PY) -m models.heuristics

# Tier 1: LightGBM lambdarank on the 87 raw features
b1:
	$(PY) -m models.lgbm_ranker --features raw --run-id b1_lgbm_raw

epool:
	$(PY) -m models.postprocess --from b1_lgbm_raw

# Tier 3: session-temporal context (f3) alone, and its past-only subset (no look-ahead)
temporal:
	$(PY) -m models.lgbm_ranker --features raw+f3 --run-id b4_lgbm_f3
	$(PY) -m models.lgbm_ranker --features raw+f3past --run-id b11_lgbm_f3past

# the selected model: raw + temporal (f3) + aircraft state changes (f4) + task context (f5), 600 rounds (~6 min)
final:
	$(PY) -m models.lgbm_ranker --features raw+f3+f4+f5 --rounds 600 --run-id $(BASELINE)

select:
	cp submissions/$(BASELINE).csv submissions/submission.csv
	$(PY) -m harness.validate_sub submissions/submission.csv

# train / stop / eval controller split: lambdarank vs the cost function (models/cost.py)
holdout:
	$(PY) -m harness.holdout
	$(PY) -m models.holdout_eval

# choose regularisation by lowest stop-set cost, then score the winner once on the eval controllers
tune:
	$(PY) -m harness.holdout
	$(PY) -m models.holdout_eval --tune
