"""Service interface for audio processing operations."""
from abc import ABC, abstractmethod

from ..entities.audio import AudioData


class IAudioProcessor(ABC):
    """
    Abstract interface for audio processing operations.

    This defines the contract for audio manipulation operations like
    normalization, format conversion, and validation.

    This follows the Interface Segregation Principle (ISP) - clients
    only depend on the methods they actually use.
    """

    @abstractmethod
    def normalize_audio(self, audio_data: bytes, sample_rate: int) -> bytes:
        """
        Normalize audio volume to consistent levels.

        Applies peak normalization and RMS normalization to ensure
        consistent audio levels across different recordings.

        Args:
            audio_data: Raw audio samples
            sample_rate: Sample rate of audio

        Returns:
            Normalized audio samples
        """
        pass

    @abstractmethod
    def validate_audio(self, audio_data: bytes, sample_rate: int) -> bool:
        """
        Validate that audio is usable for prediction.

        Checks for silence, corruption, or other issues that would
        make prediction unreliable.

        Args:
            audio_data: Raw audio samples
            sample_rate: Sample rate of audio

        Returns:
            True if audio is valid, False otherwise
        """
        pass
