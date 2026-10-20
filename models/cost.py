"""The cost function: listwise softmax cross-entropy over the tasks of one snapshot.

    p_i  = exp(s_i) / sum_j exp(s_j)          (j = the open tasks of the same sample)
    cost = mean over samples of -log p_true

Exactly one row per sample is the answer and the three picks are mutually exclusive outcomes, so top-3 by
calibrated within-sample probability is the best possible selection for the top-3 hit rate; this cost trains
for that calibration directly. Used as a LightGBM custom objective (gradient p - y, hessian p(1 - p)) and as
the early-stopping metric on held-out controllers.
"""
import numpy as np

from harness.config import K


def _starts(sizes: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(sizes)[:-1]]


def softmax_by_group(scores: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    starts = _starts(sizes)
    z = scores - np.repeat(np.maximum.reduceat(scores, starts), sizes)
    e = np.exp(z)
    return e / np.repeat(np.add.reduceat(e, starts), sizes)


def objective(preds: np.ndarray, data) -> tuple[np.ndarray, np.ndarray]:
    p = softmax_by_group(preds, data.get_group())
    y = data.get_label()
    return p - y, np.maximum(p * (1.0 - p), 1e-12)


def cost_value(scores: np.ndarray, label: np.ndarray, sizes: np.ndarray) -> float:
    p = softmax_by_group(scores, sizes)
    return float(-np.log(np.maximum(p[label == 1], 1e-15)).mean())


def top3_value(scores: np.ndarray, label: np.ndarray, sizes: np.ndarray) -> float:
    """Share of samples whose true row has fewer than K rows scored strictly above it."""
    starts = _starts(sizes)
    true_score = np.add.reduceat(np.where(label == 1, scores, 0.0), starts)
    above = np.add.reduceat((scores > np.repeat(true_score, sizes)).astype(np.int64), starts)
    return float((above < K).mean())


def feval_cost(preds, data):
    return "softmax_ce", cost_value(preds, data.get_label(), data.get_group()), False


def feval_top3(preds, data):
    return "top3_hit", top3_value(preds, data.get_label(), data.get_group()), True
