"""TensorFlow-based implementation of predictor service."""
import logging
from typing import Optional

import numpy as np

from ...domain.services.predictor_service import IPredictorService
from ...domain.repositories.model_repository import IModelRepository
from ...domain.repositories.config_repository import IConfigRepository
from ...domain.entities.audio import AudioFeatures
from ...domain.entities.prediction import (
    LanguagePrediction,
    SpoofingPrediction,
)
from ...domain.entities.model_metadata import ModelType
from ...model_inference.application.probability_utils import (
    ensure_probability_distribution,
    scalar_from_prediction,
)
from ...model_inference.application.prediction_mappers import (
    build_language_prediction,
    build_spoofing_prediction,
)

logger = logging.getLogger(__name__)


class TensorFlowPredictorService(IPredictorService):
    """
    Predictor service implementation using TensorFlow/Keras.

    This class runs inference on loaded models and returns domain prediction objects.
    It follows the Dependency Inversion Principle by depending on repository interfaces,
    not concrete implementations.

    Single Responsibility: Only handles running inference, not loading models or extracting features.
    """

    def __init__(
        self,
        model_repository: IModelRepository,
        config_repository: IConfigRepository
    ):
        """
        Initialize TensorFlow predictor service.

        Args:
            model_repository: Repository for accessing loaded models
            config_repository: Repository for accessing configuration (labels)
        """
        self._model_repo = model_repository
        self._config_repo = config_repository

    def predict_language(self, features: AudioFeatures) -> LanguagePrediction:
        """
        Predict language from audio features.

        Args:
            features: Extracted audio features

        Returns:
            LanguagePrediction with detected language and confidence

        Raises:
            RuntimeError: If language model is not loaded
            ValueError: If features are invalid
        """
        if not self._model_repo.is_model_loaded(ModelType.LANGUAGE):
            raise RuntimeError("Language model not loaded")

        # Get model and labels
        model = self._model_repo.get_model(ModelType.LANGUAGE)
        labels = self._config_repo.get_model_labels(ModelType.LANGUAGE)

        # Run inference
        probs = model.predict(features.features, verbose=0)[0]
        probs = ensure_probability_distribution(probs)

        # Keep language priors balanced: all classes get the same weight.
        class_weights = np.ones(len(labels), dtype=np.float32)
        probs = probs * class_weights
        probs = probs / (np.sum(probs) + 1e-8)

        # Log detailed probabilities
        logger.info(f"🎯 Language prediction probabilities:")
        for i, (label, prob) in enumerate(zip(labels, probs)):
            marker = "👈" if i == np.argmax(probs) else ""
            logger.info(f"   {label}: {prob*100:.1f}% {marker}")

        return build_language_prediction(labels, probs)

    def predict_spoofing(self, features: AudioFeatures) -> Optional[SpoofingPrediction]:
        """
        Detect if audio is genuine (human) or spoofed (AI-generated).

        Args:
            features: Extracted audio features (spoofing-specific)

        Returns:
            SpoofingPrediction if model available, None otherwise
        """
        if not self._model_repo.is_model_loaded(ModelType.SPOOFING):
            logger.info("Spoofing model not loaded, skipping detection")
            return None

        try:
            # Get model and labels
            model = self._model_repo.get_model(ModelType.SPOOFING)
            labels = self._config_repo.get_model_labels(ModelType.SPOOFING)

            # Run inference
            prediction = model.predict(features.features, verbose=0)[0]
            spoof_prob = scalar_from_prediction(prediction)
            spoofing_prediction = build_spoofing_prediction(labels, spoof_prob)

            logger.info(f"🔍 Spoofing detection:")
            logger.info(f"   {labels[0]}: {spoofing_prediction.genuine_probability*100:.1f}%")
            logger.info(f"   {labels[1]}: {spoofing_prediction.spoof_probability*100:.1f}%")
            logger.info(
                f"   Resultado: {spoofing_prediction.label} "
                f"{'✓' if spoofing_prediction.is_genuine else '⚠'}"
            )

            return spoofing_prediction

        except Exception as e:
            logger.error(f"Spoofing prediction failed: {e}")
            return None
