"""Paths and constants shared by every stream. Nothing here is tuned."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                      # raw competition files, read-only, not in git
ARTIFACTS = ROOT / "artifacts"            # <run_id>_{oof,test}.parquet: sample_id, row_index, score
SUBMISSIONS = ROOT / "submissions"
FOLDS_FILE = ROOT / "harness" / "folds.csv"
EXPERIMENTS_FILE = ROOT / "experiments.csv"

LABEL = "target_index"
# identifiers / keys: used for grouping and CV only, never as model inputs
ID_COLS = ["sample_id", "seq_len", "row_index", "session", "atco", "scenario", "timestamp", "task_name"]
# columns that would let a model memorise a controller or a moment of a scenario
FORBIDDEN_FEATURES = {"atco", "session", "scenario", "timestamp", "sample_id", "task_name", LABEL}

K = 3          # top-3 hit rate
N_FOLDS = 5    # GroupKFold by atco
SEED = 42

# consecutive labelled snapshots of one session that are at most this far apart belong to one label episode
# (label is unchanged 94.6 % of the time at <= 4 s, and ~0 % after any longer gap)
EPISODE_GAP_S = 4.5
