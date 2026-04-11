"""Domain entities for audio data representations."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class AudioFormat(Enum):
    """Audio format types supported by the system."""
    WAV = "wav"
    MP3 = "mp3"
    WEBM = "webm"
    OGG = "ogg"
    UNKNOWN = "unknown"


@dataclass
class AudioData:
    """
    Value object representing raw audio data.

    This is the domain representation of audio data that flows through
    the application. It contains the raw bytes and metadata about the audio.
    """
    raw_bytes: bytes
    format: AudioFormat
    filename: str
    duration_seconds: Optional[float] = None

    def __post_init__(self):
        """Validate audio data."""
        if not self.raw_bytes:
            raise ValueError("Audio data cannot be empty")
        if len(self.raw_bytes) == 0:
            raise ValueError("Audio bytes cannot be empty")


class FeatureType(Enum):
    """Types of audio features that can be extracted."""
    LANGUAGE = "language"
    ACCENT = "accent"
    SPOOFING = "spoofing"


@dataclass
class AudioFeatures:
    """
    Value object representing extracted audio features.

    This contains the processed features ready for ML model inference.
    Uses numpy arrays to represent MFCC and other audio features.
    """
    features: np.ndarray
    feature_type: FeatureType
    shape: tuple
    sample_rate: int

    def __post_init__(self):
        """Validate features."""
        if self.features is None or self.features.size == 0:
            raise ValueError("Features cannot be empty")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
