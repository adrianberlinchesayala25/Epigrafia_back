"""Domain entities for prediction results."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LanguagePrediction:
    """
    Domain model representing language detection prediction result.

    Contains the detected language, confidence score, and probability
    distribution across all possible languages.
    """
    detected_language: str
    confidence: float
    probabilities: Dict[str, float]
    prediction_index: int

    def __post_init__(self):
        """Validate prediction data."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if not self.detected_language:
            raise ValueError("Detected language cannot be empty")
        if self.prediction_index < 0:
            raise ValueError("Prediction index must be non-negative")


@dataclass
class SpoofingPrediction:
    """
    Domain model representing spoofing detection prediction result.

    Determines if the audio is genuine (human) or artificially generated (AI).
    """
    is_genuine: bool
    label: str  # "Humano" or "Artificial"
    confidence: float
    genuine_probability: float
    spoof_probability: float

    def __post_init__(self):
        """Validate prediction data."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if self.genuine_probability < 0.0 or self.genuine_probability > 1.0:
            raise ValueError("Genuine probability must be between 0 and 1")
        if self.spoof_probability < 0.0 or self.spoof_probability > 1.0:
            raise ValueError("Spoof probability must be between 0 and 1")
        if not self.label:
            raise ValueError("Label cannot be empty")


@dataclass
class CompletePrediction:
    """
    Aggregate domain model containing all prediction results.

    This represents the complete analysis result including language
    and spoofing detection (optional).
    """
    language: LanguagePrediction
    spoofing: Optional[SpoofingPrediction] = None

    def __post_init__(self):
        """Validate complete prediction."""
        if self.language is None:
            raise ValueError("Language prediction is required")
