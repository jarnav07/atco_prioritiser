PY ?= python
# the selected submission: best look-ahead model (reads later snapshots of the session), 5-fold CV 0.923
BASELINE := b8_lgbm_f345
LOOKAHEAD := b8_lgbm_f345
# causal alternative (current + earlier snapshots only), CV 0.673
CAUSAL := c3_blend_causal

.PHONY: all submission causal-submission setup check folds b0 b1 epool causal causal-check temporal lookahead select holdout tune

# every baseline, the causal blend, the look-ahead models, and the chosen submission
all: check folds b0 b1 epool causal temporal lookahead select

# the one-command path from data/ to submissions/submission.csv (look-ahead b8, ~6 min on 4 cores)
submission: check folds lookahead select

# causal submission instead: make causal-submission
causal-submission: check folds causal-check causal
	cp submissions/$(CAUSAL).csv submissions/submission.csv
	$(PY) -m harness.validate_sub submissions/submission.csv

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

# truncating sessions must leave every earlier row's features unchanged
causal-check:
	$(PY) -m harness.causal raw+f6

# causal blend: raw (b1) + causal session context (f6) + cross-controller consensus (f7), equal-weight softmax
causal: b1
	$(PY) -m models.lgbm_ranker --features raw+f6 --run-id c1_lgbm_raw_f6
	$(PY) -m models.consensus_ranker --base raw --run-id c2_consensus_raw
	$(PY) -m models.blend --runs b1_lgbm_raw c1_lgbm_raw_f6 c2_consensus_raw --run-id $(CAUSAL)

# Tier 3: session-temporal context (f3) alone, and its past-only subset (no look-ahead)
temporal:
	$(PY) -m models.lgbm_ranker --features raw+f3 --run-id b4_lgbm_f3
	$(PY) -m models.lgbm_ranker --features raw+f3past --run-id b11_lgbm_f3past

# look-ahead: raw + temporal (f3) + aircraft state changes (f4) + task context (f5), 600 rounds (~6 min)
lookahead:
	$(PY) -m models.lgbm_ranker --features raw+f3+f4+f5 --rounds 600 --run-id $(LOOKAHEAD)

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
