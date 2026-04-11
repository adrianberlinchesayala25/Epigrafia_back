"""Implementation of IConfigRepository using YAML files."""
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ...domain.repositories.config_repository import IConfigRepository
from ...domain.entities.model_metadata import ModelType
from ..config.model_config import AppConfig


class YamlConfigRepository(IConfigRepository):
    """
    Configuration repository implementation using YAML files.

    This loads configuration from config.yaml and provides it through
    the IConfigRepository interface. This follows the Dependency Inversion
    Principle - high-level code depends on IConfigRepository, not on YAML.
    """

    def __init__(self, config_path: Path):
        """
        Initialize YAML config repository.

        Args:
            config_path: Path to config.yaml file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid YAML
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in config file: {e}")

        if not config_dict:
            raise ValueError("Config file is empty")

        self._app_config = AppConfig(config_dict)

    def get_model_labels(self, model_type: ModelType) -> List[str]:
        """
        Get labels for a specific model type.

        Args:
            model_type: Type of model (language, accent, spoofing)

        Returns:
            List of label strings for predictions
        """
        model_name_map = {
            ModelType.LANGUAGE: "language",
            ModelType.SPOOFING: "spoofing"
        }

        model_name = model_name_map.get(model_type)
        if not model_name:
            raise ValueError(f"Unknown model type: {model_type}")

        return self._app_config.get_model_labels(model_name)

    def get_model_path(self, model_type: ModelType) -> Path:
        """
        Get the file path for a specific model.

        Args:
            model_type: Type of model

        Returns:
            Path to model file
        """
        model_name_map = {
            ModelType.LANGUAGE: "language",
            ModelType.SPOOFING: "spoofing"
        }

        model_name = model_name_map.get(model_type)
        if not model_name:
            raise ValueError(f"Unknown model type: {model_type}")

        return self._app_config.get_model_path(model_name)

    def get_audio_config(self) -> Dict[str, Any]:
        """
        Get audio processing configuration.

        Returns:
            Dictionary with audio processing parameters
        """
        audio_config = self._app_config.audio_processing

        return {
            "sample_rate": audio_config.sample_rate,
            "duration_seconds": audio_config.duration_seconds,
            "normalization": audio_config.normalization,
            "feature_extraction": audio_config.feature_extraction
        }

    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API configuration.

        Returns:
            Dictionary with API settings
        """
        return {
            "version": self._app_config.api_version,
            "title": self._app_config.api_title,
            "cors_origins": self._app_config.api_cors_origins
        }
