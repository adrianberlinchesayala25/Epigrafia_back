"""Use case for complete audio analysis pipeline."""
import logging

from ...domain.entities.audio import AudioData, FeatureType
from ...domain.entities.prediction import CompletePrediction
from ...domain.services.feature_extractor import IFeatureExtractor
from ...domain.services.predictor_service import IPredictorService
from ...audio_processing.application.audio_format_detector import detect_audio_format

logger = logging.getLogger(__name__)


class AnalyzeAudioUseCase:
    """
    Use case for complete audio analysis.

    This orchestrates the entire analysis pipeline:
    1. Create domain audio entity
    2. Extract features (language + spoofing)
    3. Run predictions (language, accent, spoofing)
    4. Return complete prediction

    Single Responsibility: Orchestrate the analysis workflow.
    Follows Dependency Inversion: Depends on interfaces, not concrete classes.
    """

    def __init__(
        self,
        feature_extractor: IFeatureExtractor,
        predictor_service: IPredictorService
    ):
        """
        Initialize analyze audio use case.

        Args:
            feature_extractor: Service for extracting audio features
            predictor_service: Service for running predictions
        """
        self._feature_extractor = feature_extractor
        self._predictor_service = predictor_service

    def execute(
        self,
        audio_data: bytes,
        filename: str
    ) -> CompletePrediction:
        """
        Execute the complete audio analysis pipeline.

        Args:
            audio_data: Raw audio bytes
            filename: Name of audio file (used for format detection)

        Returns:
            CompletePrediction containing all analysis results

        Raises:
            ValueError: If audio data is invalid
            RuntimeError: If required models are not loaded
        """
        logger.info(f"🎵 Starting audio analysis for: {filename}")

        # 1. Create domain audio entity
        audio_format = detect_audio_format(filename)
        audio = AudioData(
            raw_bytes=audio_data,
            format=audio_format,
            filename=filename
        )

        # 2. Extract features for language detection
        logger.info("📊 Extracting features for language detection...")
        language_features = self._feature_extractor.extract_features(
            audio, FeatureType.LANGUAGE
        )

        # 3. Run language prediction
        logger.info("🎯 Predicting language...")
        language_prediction = self._predictor_service.predict_language(language_features)

        # 4. Extract features for spoofing detection (different config)
        logger.info("📊 Extracting features for spoofing detection...")
        spoofing_features = self._feature_extractor.extract_features(
            audio, FeatureType.SPOOFING
        )

        # 5. Run spoofing detection (optional)
        logger.info("🔍 Detecting spoofing...")
        spoofing_prediction = self._predictor_service.predict_spoofing(spoofing_features)

        # 6. Build complete prediction
        complete_prediction = CompletePrediction(
            language=language_prediction,
            spoofing=spoofing_prediction
        )

        logger.info(f"✅ Analysis complete: {language_prediction.detected_language} "
                   f"({language_prediction.confidence*100:.1f}%)")

        return complete_prediction
