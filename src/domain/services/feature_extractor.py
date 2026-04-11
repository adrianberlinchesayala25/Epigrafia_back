"""Service interface for audio feature extraction."""
from abc import ABC, abstractmethod

from ..entities.audio import AudioData, AudioFeatures, FeatureType


class IFeatureExtractor(ABC):
    """
    Abstract interface for audio feature extraction.

    This defines the contract for extracting ML-ready features from audio.
    Implementations can use librosa, torchaudio, or any other library.

    This follows the Dependency Inversion Principle (DIP) - application
    logic depends on this interface, not on librosa directly.
    """

    @abstractmethod
    def extract_features(
        self,
        audio_data: AudioData,
        feature_type: FeatureType
    ) -> AudioFeatures:
        """
        Extract features from audio data.

        Different feature types may require different extraction parameters:
        - LANGUAGE/ACCENT: Standard MFCC with deltas (120 features)
        - SPOOFING: Specialized MFCC configuration (40 features)

        Args:
            audio_data: Domain audio data object
            feature_type: Type of features to extract

        Returns:
            AudioFeatures containing extracted feature arrays

        Raises:
            ValueError: If audio data is invalid or empty
        """
        pass
