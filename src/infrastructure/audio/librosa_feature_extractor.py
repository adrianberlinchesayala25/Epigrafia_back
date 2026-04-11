"""Librosa-based implementation of feature extractor."""
import logging

import numpy as np
import librosa

from ...domain.services.feature_extractor import IFeatureExtractor
from ...domain.entities.audio import AudioData, AudioFeatures, FeatureType
from ...infrastructure.config.settings import Settings
from ...audio_processing.infrastructure.audio_io import write_temp_audio, cleanup_temp_file
from ...audio_processing.infrastructure.audio_normalization import (
    normalize_standard_audio,
    normalize_peak_audio,
)
from ...audio_processing.infrastructure.mfcc_feature_builders import (
    fit_audio_to_duration,
    build_language_features,
    build_spoofing_features,
)

logger = logging.getLogger(__name__)


class LibrosaFeatureExtractor(IFeatureExtractor):
    """
    Feature extractor implementation using librosa.

    This extracts MFCC (Mel-Frequency Cepstral Coefficients) and delta features
    from audio data. Different feature configurations are used for different models.

    Dependency Inversion: Application depends on IFeatureExtractor interface,
    not on this librosa-specific implementation.
    """

    def __init__(self, settings: Settings):
        """
        Initialize librosa feature extractor.

        Args:
            settings: Application settings with audio processing configuration
        """
        self._settings = settings

    def extract_features(
        self,
        audio_data: AudioData,
        feature_type: FeatureType
    ) -> AudioFeatures:
        """
        Extract features from audio data.

        Args:
            audio_data: Domain audio data object
            feature_type: Type of features to extract

        Returns:
            AudioFeatures containing extracted feature arrays

        Raises:
            ValueError: If audio data is invalid
        """
        if feature_type == FeatureType.SPOOFING:
            return self._extract_spoofing_features(audio_data)
        else:
            # LANGUAGE and ACCENT use same feature configuration
            return self._extract_standard_features(audio_data)

    def _extract_standard_features(self, audio_data: AudioData) -> AudioFeatures:
        """
        Extract standard MFCC features for language/accent detection.

        Extracts 40 MFCC coefficients + delta + delta-delta = 120 features total.

        Args:
            audio_data: Audio data to process

        Returns:
            AudioFeatures with shape (1, time_frames, 120)
        """
        # Write audio to temp file for librosa
        audio_path = write_temp_audio(audio_data.raw_bytes, self._settings.sample_rate, logger)

        try:
            # Load audio with librosa
            logger.info(f"📂 Loading audio for standard feature extraction")
            y, sr = librosa.load(audio_path, sr=self._settings.sample_rate, mono=True)
            logger.info(f"   ✅ Audio loaded: {len(y)} samples, duration: {len(y)/sr:.2f}s")

            # Check audio quality
            audio_rms = np.sqrt(np.mean(y**2))
            audio_max = np.abs(y).max()
            logger.info(f"   📊 Audio stats (before norm): RMS={audio_rms:.6f}, Max={audio_max:.6f}")

            if audio_rms < 0.001:
                logger.warning(f"   ⚠️ Audio appears to be very quiet (RMS={audio_rms:.6f})")

            # Normalize audio (two-stage normalization)
            y = normalize_standard_audio(y, logger)

            # Ensure minimum duration
            y = fit_audio_to_duration(
                y,
                sample_rate=self._settings.sample_rate,
                duration_seconds=self._settings.duration_seconds,
            )

            features = build_language_features(y, self._settings)

            logger.info(f"📊 Features extracted: shape={features.shape}")

            return AudioFeatures(
                features=features,
                feature_type=FeatureType.LANGUAGE,  # Works for accent too
                shape=features.shape,
                sample_rate=self._settings.sample_rate
            )

        finally:
            cleanup_temp_file(audio_path, logger)

    def _extract_spoofing_features(self, audio_data: AudioData) -> AudioFeatures:
        """
        Extract MFCC features optimized for spoofing detection.

        Uses simpler configuration: 40 MFCC coefficients, fixed length.

        Args:
            audio_data: Audio data to process

        Returns:
            AudioFeatures with shape (1, MAX_LENGTH, 40)
        """
        audio_path = write_temp_audio(audio_data.raw_bytes, self._settings.sample_rate, logger)

        try:
            # Load audio with librosa
            logger.info(f"📂 Loading audio for spoofing feature extraction")
            y, sr = librosa.load(audio_path, sr=self._settings.sample_rate, mono=True)

            y = normalize_peak_audio(y)

            features = build_spoofing_features(
                y,
                sample_rate=sr,
                n_mfcc=self._settings.spoofing_n_mfcc,
                max_length=self._settings.spoofing_max_length,
            )

            logger.info(f"📊 Spoofing features extracted: shape={features.shape}")

            return AudioFeatures(
                features=features,
                feature_type=FeatureType.SPOOFING,
                shape=features.shape,
                sample_rate=self._settings.sample_rate
            )

        finally:
            cleanup_temp_file(audio_path, logger)
