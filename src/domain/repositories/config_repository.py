"""Repository interface for configuration management."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from ..entities.model_metadata import ModelType


class IConfigRepository(ABC):
    """
    Abstract interface for configuration repository.

    This defines the contract for accessing application configuration.
    Implementations can load from YAML, JSON, database, or any source.

    This follows the Dependency Inversion Principle (DIP).
    """

    @abstractmethod
    def get_model_labels(self, model_type: ModelType) -> List[str]:
        """
        Get labels for a specific model type.

        Args:
            model_type: Type of model (language, accent, spoofing)

        Returns:
            List of label strings for predictions

        Example:
            For language model: ["Español", "Inglés", "Francés", "Alemán"]
        """
        pass

    @abstractmethod
    def get_model_path(self, model_type: ModelType) -> Path:
        """
        Get the file path for a specific model.

        Args:
            model_type: Type of model

        Returns:
            Path to model file
        """
        pass

    @abstractmethod
    def get_audio_config(self) -> Dict[str, Any]:
        """
        Get audio processing configuration.

        Returns:
            Dictionary with audio processing parameters:
            - sample_rate
            - duration_seconds
            - n_mfcc, n_mels, hop_length, n_fft
            - normalization settings
        """
        pass

    @abstractmethod
    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API configuration.

        Returns:
            Dictionary with API settings:
            - version
            - title
            - cors_origins
        """
        pass
