"""Mapping helpers from model outputs to domain prediction entities."""

import numpy as np

from ...domain.entities.prediction import LanguagePrediction, SpoofingPrediction


def build_language_prediction(labels, probs) -> LanguagePrediction:
    """Map labels and probabilities into a LanguagePrediction entity."""
    prediction_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    probabilities_dict = {
        label: float(prob) for label, prob in zip(labels, probs)
    }

    return LanguagePrediction(
        detected_language=labels[prediction_idx],
        confidence=confidence,
        probabilities=probabilities_dict,
        prediction_index=prediction_idx,
    )


def build_spoofing_prediction(labels, spoof_prob: float) -> SpoofingPrediction:
    """Map spoofing probability into SpoofingPrediction entity."""
    genuine_prob = 1.0 - spoof_prob
    is_genuine = genuine_prob > 0.5
    confidence = genuine_prob if is_genuine else spoof_prob
    label = labels[0] if is_genuine else labels[1]

    return SpoofingPrediction(
        is_genuine=is_genuine,
        label=label,
        confidence=confidence,
        genuine_probability=genuine_prob,
        spoof_probability=spoof_prob,
    )
