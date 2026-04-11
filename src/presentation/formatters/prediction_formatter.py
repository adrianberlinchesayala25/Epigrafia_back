"""Formatter for prediction results to API responses."""
from typing import Dict, Any

from ...domain.entities.prediction import CompletePrediction


class PredictionFormatter:
    """
    Formatter for converting domain predictions to API response format.

    Single Responsibility: Only handles response formatting.
    Open/Closed: Easy to extend for new response formats without modifying existing code.
    """

    def format_prediction(self, prediction: CompletePrediction) -> Dict[str, Any]:
        """
        Format complete prediction as API response dictionary.

        This maintains backward compatibility with the existing API
        while using the new clean architecture.

        Args:
            prediction: Domain prediction object

        Returns:
            Dictionary formatted for API response
        """
        # Format language prediction
        response = {
            "success": True,
            "language": {
                "detected": prediction.language.detected_language,
                "confidence": prediction.language.confidence,
                "probabilities": prediction.language.probabilities
            },
            # Legacy fields for backward compatibility
            "language_prediction": prediction.language.prediction_index,
            "language_confidence": prediction.language.confidence
        }

        # Format spoofing prediction
        if prediction.spoofing:
            response["spoofing"] = {
                "is_genuine": prediction.spoofing.is_genuine,
                "label": prediction.spoofing.label,
                "confidence": prediction.spoofing.confidence,
                "genuine_probability": prediction.spoofing.genuine_probability,
                "spoof_probability": prediction.spoofing.spoof_probability
            }
        else:
            response["spoofing"] = None

        return response
