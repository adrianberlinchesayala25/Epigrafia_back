"""Application settings loaded from environment variables."""
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from .env file and environment variables.

    Environment variables override defaults defined here.
    Uses pydantic-settings for validation and type safety.
    """

    # Environment
    environment: str = Field(default="production", description="Deployment environment")
    log_level: str = Field(default="INFO", description="Logging level")
    port: int = Field(default=8000, description="Server port")

    # Model paths
    models_dir: Path = Field(default=Path("models"), description="Base directory for models")
    language_model_path: Optional[Path] = Field(
        default=Path("models/language/language_model_best.keras"),
        description="Path to language detection model"
    )
    spoofing_model_path: Optional[Path] = Field(
        default=Path("models/spoofing/spoofing_best.keras"),
        description="Path to spoofing detection model"
    )

    # Audio processing configuration
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    duration_seconds: int = Field(default=3, description="Audio duration for processing")
    n_mfcc: int = Field(default=40, description="Number of MFCC coefficients")
    n_mels: int = Field(default=128, description="Number of mel bands")
    hop_length: int = Field(default=512, description="Hop length for STFT")
    n_fft: int = Field(default=2048, description="FFT window size")

    # Spoofing-specific configuration
    spoofing_max_length: int = Field(default=400, description="Max time steps for spoofing features")
    spoofing_n_mfcc: int = Field(default=40, description="MFCC coefficients for spoofing")

    # CORS configuration
    cors_origins: str = Field(default="*", description="Allowed CORS origins (comma-separated)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get model path by name.

        Args:
            model_name: One of 'language', 'spoofing'

        Returns:
            Path to model file if configured, None otherwise
        """
        model_paths = {
            "language": self.language_model_path,
            "spoofing": self.spoofing_model_path
        }
        return model_paths.get(model_name)
