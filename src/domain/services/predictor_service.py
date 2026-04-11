"""Service interface for ML prediction operations."""
from abc import ABC, abstractmethod
from typing import Optional

from ..entities.audio import AudioFeatures
from ..entities.prediction import (
    LanguagePrediction,
    SpoofingPrediction
)


class IPredictorService(ABC):
    """
    Abstract interface for ML prediction service.

    This defines the contract for running inference on ML models.
    Implementations can use TensorFlow, PyTorch, ONNX, or any framework.

    This follows the Dependency Inversion Principle (DIP) and
    Interface Segregation Principle (ISP).
    """

    @abstractmethod
    def predict_language(self, features: AudioFeatures) -> LanguagePrediction:
        """
        Predict language from audio features.

        Args:
            features: Extracted audio features

        Returns:
            LanguagePrediction with detected language and confidence

        Raises:
            ValueError: If features are invalid
            RuntimeError: If model is not loaded
        """
        pass

    @abstractmethod
    def predict_spoofing(self, features: AudioFeatures) -> Optional[SpoofingPrediction]:
        """
        Detect if audio is genuine (human) or spoofed (AI-generated).

        Args:
            features: Extracted audio features (spoofing-specific)

        Returns:
            SpoofingPrediction if model available, None otherwise
        """
        pass
