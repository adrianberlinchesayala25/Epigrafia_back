"""Application layer DTOs for requests."""
from dataclasses import dataclass


@dataclass
class AnalyzeAudioRequest:
    """
    Data Transfer Object for analyze audio request.

    This represents the input data needed for the full audio analysis pipeline.
    """
    audio_data: bytes
    filename: str

    def __post_init__(self):
        """Validate request data."""
        if not self.audio_data:
            raise ValueError("Audio data cannot be empty")
        if len(self.audio_data) == 0:
            raise ValueError("Audio data cannot be empty")
        if not self.filename:
            raise ValueError("Filename cannot be empty")
