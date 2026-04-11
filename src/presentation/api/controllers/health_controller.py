"""Controller for health check endpoints."""
import logging
from typing import Dict, Any

from ....domain.repositories.model_repository import IModelRepository
from ....domain.repositories.config_repository import IConfigRepository
from ....domain.entities.model_metadata import ModelType

logger = logging.getLogger(__name__)


class HealthController:
    """
    Controller for health check and system status endpoints.

    Single Responsibility: Handle health check logic.
    Dependency Inversion: Depends on repository interfaces.
    """

    def __init__(
        self,
        model_repository: IModelRepository,
        config_repository: IConfigRepository
    ):
        """
        Initialize health controller.

        Args:
            model_repository: Repository for accessing model information
            config_repository: Repository for accessing configuration
        """
        self._model_repo = model_repository
        self._config_repo = config_repository

    async def check_health(self) -> Dict[str, Any]:
        """
        Check if the API is healthy and ready to serve requests.

        Returns:
            Dictionary with health status
        """
        models_loaded = self._check_models_loaded()

        return {
            "status": "healthy",
            "models_loaded": models_loaded
        }

    async def get_models_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all models.

        Returns:
            Dictionary with model loading status and labels
        """
        # Check which models are loaded
        language_loaded = self._model_repo.is_model_loaded(ModelType.LANGUAGE)
        spoofing_loaded = self._model_repo.is_model_loaded(ModelType.SPOOFING)

        overall_loaded = language_loaded  # At least language model must be loaded

        # Get labels from config
        try:
            language_labels = self._config_repo.get_model_labels(ModelType.LANGUAGE)
        except Exception as e:
            logger.warning(f"Failed to get language labels: {e}")
            language_labels = []

        return {
            "loaded": overall_loaded,
            "language_model": language_loaded,
            "spoofing_model": spoofing_loaded,
            "language_labels": language_labels if language_loaded else []
        }

    def _check_models_loaded(self) -> bool:
        """
        Check if required models are loaded.

        Returns:
            True if at least language model is loaded
        """
        return self._model_repo.is_model_loaded(ModelType.LANGUAGE)
