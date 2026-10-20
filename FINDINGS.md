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

## Noise floor

5 folds of 2,469–2,757 samples carry 575–723 runs each ⇒ per-fold SE ≈ 1.8–2.1 pt; pooled SE ≈ 0.85 pt.
Fold-to-fold spread of b1 is ±2.1 pt. **Treat any change under ~1 pt as noise.**
