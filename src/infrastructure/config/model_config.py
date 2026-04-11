"""Model-specific configuration loaded from YAML."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    """
    Configuration for a specific ML model.

    Loaded from config.yaml and provides model-specific settings
    like labels, paths, and parameters.
    """
    model_file: str
    labels: List[str]


@dataclass
class AudioProcessingConfig:
    """Configuration for audio processing pipeline."""
    sample_rate: int
    duration_seconds: int
    normalization: Dict[str, float]
    feature_extraction: Dict[str, Any]


class AppConfig:
    """
    Application configuration loaded from YAML file.

    This centralizes all configuration that doesn't come from environment
    variables, particularly model labels and audio processing parameters.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.

        Args:
            config_dict: Parsed YAML configuration
        """
        self._config = config_dict

        # Parse model configurations
        models_config = config_dict.get("models", {})
        self.base_directory = Path(models_config.get("base_directory", "models"))

        self.language_config = ModelConfig(
            model_file=models_config.get("language", {}).get("model_file", ""),
            labels=models_config.get("language", {}).get("labels", [])
        )

        self.spoofing_config = ModelConfig(
            model_file=models_config.get("spoofing", {}).get("model_file", ""),
            labels=models_config.get("spoofing", {}).get("labels", [])
        )

        # Parse audio processing config
        audio_config = config_dict.get("audio_processing", {})
        self.audio_processing = AudioProcessingConfig(
            sample_rate=audio_config.get("sample_rate", 16000),
            duration_seconds=audio_config.get("duration_seconds", 3),
            normalization=audio_config.get("normalization", {}),
            feature_extraction=audio_config.get("feature_extraction", {})
        )

        # Parse API config
        api_config = config_dict.get("api", {})
        self.api_version = api_config.get("version", "2.0.0")
        self.api_title = api_config.get("title", "EpigrafIA API")
        self.api_cors_origins = api_config.get("cors_origins", ["*"])

    def get_model_path(self, model_name: str) -> Path:
        """
        Get full path to model file.

        Args:
            model_name: One of 'language', 'spoofing'

        Returns:
            Full path to model file
        """
        config_map = {
            "language": self.language_config,
            "spoofing": self.spoofing_config
        }

        if model_name not in config_map:
            raise ValueError(f"Unknown model name: {model_name}")

        model_config = config_map[model_name]
        return self.base_directory / model_config.model_file

    def get_model_labels(self, model_name: str) -> List[str]:
        """
        Get labels for a specific model.

        Args:
            model_name: One of 'language', 'spoofing'

        Returns:
            List of label strings
        """
        config_map = {
            "language": self.language_config,
            "spoofing": self.spoofing_config
        }

        if model_name not in config_map:
            raise ValueError(f"Unknown model name: {model_name}")

        return config_map[model_name].labels
