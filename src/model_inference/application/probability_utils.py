"""Probability and post-processing helpers for model inference."""

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def ensure_probability_distribution(raw_probs):
    """Normalize model outputs into a probability distribution."""
    probs = np.asarray(raw_probs)
    if not np.isclose(probs.sum(), 1.0, atol=0.01):
        probs = _softmax(probs)
    return probs


def scalar_from_prediction(raw_prediction) -> float:
    """Extract scalar from numpy/tensor outputs safely."""
    prediction = np.asarray(raw_prediction)
    if prediction.shape == ():
        return float(prediction)
    return float(prediction.flat[0])
