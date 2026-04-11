import numpy as np

from src.model_inference.application.probability_utils import (
    ensure_probability_distribution,
    scalar_from_prediction,
)
from src.model_inference.application.prediction_mappers import (
    build_language_prediction,
    build_spoofing_prediction,
)


def test_ensure_probability_distribution_applies_softmax_when_needed():
    probs = ensure_probability_distribution(np.array([2.0, 1.0]))
    assert np.isclose(probs.sum(), 1.0)


def test_scalar_from_prediction_accepts_scalar_and_vector():
    assert scalar_from_prediction(np.array(0.7)) == 0.7
    assert scalar_from_prediction(np.array([0.3])) == 0.3


def test_prediction_mappers_build_expected_domain_objects():
    language = build_language_prediction(["Español", "Inglés"], np.array([0.8, 0.2]))
    spoofing = build_spoofing_prediction(["Humano", "Artificial"], 0.15)

    assert language.detected_language == "Español"
    assert language.prediction_index == 0
    assert spoofing.is_genuine is True
    assert spoofing.label == "Humano"
