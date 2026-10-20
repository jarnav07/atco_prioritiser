PY ?= python
BASELINE := b1_lgbm_raw

.PHONY: all submission setup check folds b0 b1 epool select

# every baseline + the chosen submission
all: check folds b0 b1 epool select

# the one-command path from data/ to submissions/submission.csv
submission: check folds b1 select

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

b1:
	$(PY) -m models.lgbm_ranker --features raw --run-id $(BASELINE)

epool:
	$(PY) -m models.postprocess --from $(BASELINE)

select:
	cp submissions/$(BASELINE).csv submissions/submission.csv
	$(PY) -m harness.validate_sub submissions/submission.csv
