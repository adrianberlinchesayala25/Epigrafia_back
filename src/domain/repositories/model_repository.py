"""Repository interface for ML model loading and management."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..entities.model_metadata import ModelMetadata, ModelType


class IModelRepository(ABC):
    """
    Abstract interface for model repository.

    This defines the contract for loading and managing ML models.
    Implementations can use TensorFlow, PyTorch, ONNX, or any other framework.

    This follows the Dependency Inversion Principle (DIP) - high-level code
    depends on this abstraction, not on concrete implementations.
    """

    @abstractmethod
    def load_model(self, model_type: ModelType, model_path: Path) -> Any:
        """
        Load an ML model from storage.

        Args:
            model_type: Type of model to load (language, accent, spoofing)
            model_path: Path to the model file

        Returns:
            Loaded model instance (framework-specific)

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted or invalid
        """
        pass

    @abstractmethod
    def get_model(self, model_type: ModelType) -> Optional[Any]:
        """
        Get a previously loaded model.

        Args:
            model_type: Type of model to retrieve

        Returns:
            Model instance if loaded, None otherwise
        """
        pass

    @abstractmethod
    def get_model_metadata(self, model_type: ModelType) -> ModelMetadata:
        """
        Get metadata about a model.

        Args:
            model_type: Type of model

        Returns:
            ModelMetadata containing model information
        """
        pass

    @abstractmethod
    def is_model_loaded(self, model_type: ModelType) -> bool:
        """
        Check if a model is loaded.

        Args:
            model_type: Type of model to check

        Returns:
            True if model is loaded, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources and release memory.

        Should be called on application shutdown.
        """
        pass
