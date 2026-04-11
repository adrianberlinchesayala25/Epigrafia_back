"""Application layer DTOs for responses."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LanguagePredictionDTO:
    """DTO for language prediction in API responses."""
    detected: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class SpoofingPredictionDTO:
    """DTO for spoofing prediction in API responses."""
    is_genuine: bool
    label: str
    confidence: float
    genuine_probability: float
    spoof_probability: float


@dataclass
class AnalyzeAudioResponse:
    """
    Data Transfer Object for analyze audio response.

    This represents the output of the full audio analysis pipeline.
    """
    success: bool
    language: LanguagePredictionDTO
    spoofing: Optional[SpoofingPredictionDTO] = None

    # Legacy fields for backward compatibility
    language_prediction: Optional[int] = None
    language_confidence: Optional[float] = None
