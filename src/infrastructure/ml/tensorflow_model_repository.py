"""TensorFlow-based implementation of model repository."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf
import keras

from ...domain.repositories.model_repository import IModelRepository
from ...domain.entities.model_metadata import ModelMetadata, ModelType
from .custom_layers import SpecAugment, SEBlock

logger = logging.getLogger(__name__)


class TensorFlowModelRepository(IModelRepository):
    """
    Model repository implementation using TensorFlow/Keras.

    This class handles loading, storing, and managing Keras models.
    It follows the Repository pattern and implements IModelRepository interface.

    Dependency Inversion Principle: Application code depends on IModelRepository,
    not on this concrete TensorFlow implementation.
    """

    def __init__(self):
        """Initialize TensorFlow model repository."""
        self._models: Dict[ModelType, Any] = {}
        self._model_paths: Dict[ModelType, Path] = {}

        # Configure TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def load_model(self, model_type: ModelType, model_path: Path) -> Any:
        """
        Load a Keras model from file.

        Args:
            model_type: Type of model to load
            model_path: Path to .keras model file

        Returns:
            Loaded Keras model

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"📥 Loading {model_type.value} model from {model_path}")

        try:
            if model_type == ModelType.SPOOFING:
                # Spoofing model requires custom layers
                model = self._load_spoofing_model(model_path)
            else:
                # Standard model loading
                model = tf.keras.models.load_model(str(model_path))

            self._models[model_type] = model
            self._model_paths[model_type] = model_path

            logger.info(f"   ✅ Input shape: {model.input_shape}")
            logger.info(f"   ✅ Output shape: {model.output_shape}")

            return model

        except Exception as e:
            logger.error(f"   ❌ Failed to load {model_type.value} model: {e}")
            raise ValueError(f"Failed to load model: {e}") from e

    def _load_spoofing_model(self, model_path: Path) -> Any:
        """
        Load spoofing model with custom layers.

        Args:
            model_path: Path to spoofing model file

        Returns:
            Loaded Keras model with custom layers
        """
        custom_objects = {
            'SpecAugment': SpecAugment,
            'SEBlock': SEBlock
        }

        model = keras.models.load_model(
            str(model_path),
            compile=False,
            custom_objects=custom_objects
        )

        return model

    def get_model(self, model_type: ModelType) -> Optional[Any]:
        """
        Get a previously loaded model.

        Args:
            model_type: Type of model to retrieve

        Returns:
            Model instance if loaded, None otherwise
        """
        return self._models.get(model_type)

    def get_model_metadata(self, model_type: ModelType) -> ModelMetadata:
        """
        Get metadata about a model.

        Args:
            model_type: Type of model

        Returns:
            ModelMetadata with model information
        """
        model = self._models.get(model_type)
        model_path = self._model_paths.get(model_type)

        return ModelMetadata(
            model_type=model_type,
            model_path=model_path,
            labels=[],  # Labels come from config, not model itself
            input_shape=model.input_shape if model else None,
            is_loaded=model is not None
        )

    def is_model_loaded(self, model_type: ModelType) -> bool:
        """
        Check if a model is loaded.

        Args:
            model_type: Type of model to check

        Returns:
            True if model is loaded and ready
        """
        return model_type in self._models and self._models[model_type] is not None

    def cleanup(self) -> None:
        """
        Clean up resources and release memory.

        Deletes all loaded models to free GPU/CPU memory.
        Should be called on application shutdown.
        """
        logger.info("🧹 Cleaning up model resources...")

        for model_type in list(self._models.keys()):
            if self._models[model_type] is not None:
                del self._models[model_type]
                logger.info(f"   ✅ Released {model_type.value} model")

        self._models.clear()
        self._model_paths.clear()

        logger.info("✅ Model cleanup complete")
