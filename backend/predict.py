"""
🧠 EpigrafIA Predictor
======================
Audio prediction module for language and accent detection

Uses trained Keras models to predict:
- Language (Español, Inglés, Francés, Alemán)
- Accent (regional variants for each language)
"""

import io
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import tempfile
import os

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy libraries
tf = None
librosa = None


def _load_tensorflow():
    """Lazy load TensorFlow"""
    global tf
    if tf is None:
        import tensorflow as tensorflow_module
        tf = tensorflow_module
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    return tf


def _load_librosa():
    """Lazy load librosa"""
    global librosa
    if librosa is None:
        import librosa as librosa_module
        librosa = librosa_module
    return librosa


class AudioPredictor:
    """
    Audio prediction class for language and accent detection
    
    Attributes:
        language_model: Keras model for language detection
        accent_model: Keras model for accent detection
        models_loaded: Whether models are successfully loaded
    """
    
    # Audio processing configuration (must match training)
    SAMPLE_RATE = 16000
    DURATION = 3  # seconds
    N_MFCC = 40
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    
    def __init__(
        self,
        language_model_path: Optional[Union[str, Path]] = None,
        accent_model_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize predictor with model paths
        
        Args:
            language_model_path: Path to language detection model (.keras or .h5)
            accent_model_path: Path to accent detection model (.keras or .h5)
        """
        self.language_model = None
        self.accent_model = None
        self.models_loaded = False
        
        # Load TensorFlow
        _load_tensorflow()
        
        # Load models if paths provided
        if language_model_path:
            self.load_language_model(language_model_path)
        if accent_model_path:
            self.load_accent_model(accent_model_path)
        
        # Models are loaded if at least language model exists
        self.models_loaded = self.language_model is not None
    
    def load_language_model(self, model_path: Union[str, Path]) -> None:
        """Load language detection model"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Language model not found: {path}")
        
        logger.info(f"📥 Loading language model from {path}")
        self.language_model = tf.keras.models.load_model(str(path))
        logger.info(f"   Input shape: {self.language_model.input_shape}")
        logger.info(f"   Output shape: {self.language_model.output_shape}")
    
    def load_accent_model(self, model_path: Union[str, Path]) -> None:
        """Load accent detection model"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Accent model not found: {path}")
        
        logger.info(f"📥 Loading accent model from {path}")
        self.accent_model = tf.keras.models.load_model(str(path))
        logger.info(f"   Input shape: {self.accent_model.input_shape}")
        logger.info(f"   Output shape: {self.accent_model.output_shape}")
    
    def extract_features(self, audio_data: bytes) -> np.ndarray:
        """
        Extract MFCC features from audio data
        
        Args:
            audio_data: Raw audio bytes (WAV, MP3, WebM, etc.)
            
        Returns:
            numpy array of MFCC features shaped for model input
        """
        _load_librosa()
        
        # Detect if WebM by checking magic bytes
is_webm = False
        suffix = '.webm' if is_webm else '.wav'
        
        # Write original audio to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        wav_path = None
        try:
            # If WebM, convert to WAV using pydub (requires ffmpeg)
            if is_webm:
                logger.info("🔄 Converting WebM to WAV...")
                try:
                    from pydub import AudioSegment
                    audio_segment = AudioSegment.from_file(tmp_path, format="webm")
                    # Convert to mono and set sample rate
                    audio_segment = audio_segment.set_channels(1).set_frame_rate(self.SAMPLE_RATE)
                    
                    # Export as WAV
                    wav_path = tmp_path.replace('.webm', '.wav')
                    audio_segment.export(wav_path, format="wav")
                    load_path = wav_path
                    logger.info(f"   ✅ Converted to WAV: {len(audio_segment)}ms")
                except ImportError:
                    logger.warning("⚠️ pydub not installed, trying direct librosa load...")
                    load_path = tmp_path
                except Exception as e:
                    logger.error(f"⚠️ pydub conversion failed: {type(e).__name__}: {e}")
                    load_path = tmp_path
            else:
                load_path = tmp_path
            
            # NOTE: We no longer convert to MP3 as it can introduce different artifacts
            # The model with augmentation should handle the difference between WAV and MP3
            
            # Load audio with librosa
            logger.info(f"📂 Loading audio from: {load_path}")
            y, sr = librosa.load(load_path, sr=self.SAMPLE_RATE, mono=True)
            logger.info(f"   ✅ Audio loaded: {len(y)} samples, {sr}Hz, duration: {len(y)/sr:.2f}s")
            
            # Check for silence/empty audio
            audio_rms = np.sqrt(np.mean(y**2))
            audio_max = np.abs(y).max()
            logger.info(f"   📊 Audio stats (before norm): RMS={audio_rms:.6f}, Max={audio_max:.6f}")
            
            if audio_rms < 0.001:
                logger.warning(f"   ⚠️ Audio appears to be very quiet (RMS={audio_rms:.6f})")
            
            # ========== NORMALIZE AUDIO VOLUME ==========
            # Two-stage normalization for robustness:
            # 1. Peak normalization - ensure we use full dynamic range
            # 2. RMS normalization - match training data volume levels
            
            # Stage 1: Peak normalization to 0.8 (leave headroom)
            TARGET_PEAK = 0.8
            if audio_max > 0.01:  # Only if there's actual audio
                peak_factor = TARGET_PEAK / audio_max
                y = y * peak_factor
                logger.info(f"   🔊 Peak normalized: factor={peak_factor:.2f}")
            
            # Stage 2: RMS normalization (typical speech in Common Voice is ~0.05-0.15)
            # We use a slightly lower target to avoid over-amplification artifacts
            audio_rms_after_peak = np.sqrt(np.mean(y**2))
            TARGET_RMS = 0.08  # Slightly lower than before for more natural sound
            
            if audio_rms_after_peak > 0.001 and audio_rms_after_peak < TARGET_RMS * 0.5:
                # Only boost if really quiet after peak normalization
                rms_factor = min(TARGET_RMS / audio_rms_after_peak, 3.0)  # Limit to 3x
                y = y * rms_factor
                y = np.clip(y, -1.0, 1.0)
                new_rms = np.sqrt(np.mean(y**2))
                logger.info(f"   🔊 RMS boosted: factor={rms_factor:.2f}, final RMS={new_rms:.6f}")
            else:
                new_rms = audio_rms_after_peak
                logger.info(f"   ✅ Audio level OK: RMS={new_rms:.6f}")
            
            # Ensure minimum duration
            min_samples = self.SAMPLE_RATE * self.DURATION
            if len(y) < min_samples:
                # Pad with zeros if too short
                y = np.pad(y, (0, min_samples - len(y)), mode='constant')
            else:
                # Trim to exact duration
                y = y[:min_samples]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=y,
                sr=self.SAMPLE_RATE,
                n_mfcc=self.N_MFCC,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH
            )
            
            # Compute delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Stack features: (n_mfcc * 3, time_frames)
            features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
            
            # Normalize
            features = (features - features.mean()) / (features.std() + 1e-8)
            
            # Transpose to (time_frames, features) and add batch dimension
            features = features.T  # (time_frames, 120)
            features = np.expand_dims(features, axis=0)  # (1, time_frames, 120)
            
            logger.info(f"📊 Features extracted: shape={features.shape}")
            
            return features
            
        finally:
            # Cleanup temp files
            try:
                os.unlink(tmp_path)
            except:
                pass
            if wav_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
    
    def predict(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Run full prediction pipeline on audio data
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Dictionary with language and accent predictions
        """
        if self.language_model is None:
            raise RuntimeError("Language model not loaded")
        
        # Extract features
        features = self.extract_features(audio_data)
        
        # Run language prediction
        language_probs = self.language_model.predict(features, verbose=0)[0]
        
        # Apply softmax if needed (ensure probabilities sum to 1)
        if not np.isclose(language_probs.sum(), 1.0, atol=0.01):
            language_probs = tf.nn.softmax(language_probs).numpy()
        
        # Log detailed probabilities for debugging

        logger.info(f"🎯 Prediction probabilities:")
        for i, (name, prob) in enumerate(zip(lang_names, language_probs)):
            marker = "👈" if i == np.argmax(language_probs) else ""
            logger.info(f"   {name}: {prob*100:.1f}% {marker}")
        
        result = {
            'language_probabilities': language_probs.tolist(),
            'language_prediction': int(np.argmax(language_probs)),
            'language_confidence': float(np.max(language_probs))
        }
        
        # Run accent prediction if model available
        if self.accent_model is not None:
            accent_probs = self.accent_model.predict(features, verbose=0)[0]
            if not np.isclose(accent_probs.sum(), 1.0, atol=0.01):
                accent_probs = tf.nn.softmax(accent_probs).numpy()
            
            result.update({
                'accent_probabilities': accent_probs.tolist(),
                'accent_prediction': int(np.argmax(accent_probs)),
                'accent_confidence': float(np.max(accent_probs))
            })
        else:
            result.update({
                'accent_probabilities': None,
                'accent_prediction': None,
                'accent_confidence': None
            })
        
        return result
    
    def predict_language(self, audio_data: bytes) -> Dict[str, Any]:
        """Predict only language"""
        if self.language_model is None:
            raise RuntimeError("Language model not loaded")
        
        features = self.extract_features(audio_data)
        probs = self.language_model.predict(features, verbose=0)[0]
        
        if not np.isclose(probs.sum(), 1.0, atol=0.01):
            probs = tf.nn.softmax(probs).numpy()
        
        return {
            'probabilities': probs,
            'prediction': int(np.argmax(probs)),
            'confidence': float(np.max(probs))
        }
    
    def predict_accent(self, audio_data: bytes) -> Dict[str, Any]:
        """Predict only accent"""
        if self.accent_model is None:
            raise RuntimeError("Accent model not loaded")
        
        features = self.extract_features(audio_data)
        probs = self.accent_model.predict(features, verbose=0)[0]
        
        if not np.isclose(probs.sum(), 1.0, atol=0.01):
            probs = tf.nn.softmax(probs).numpy()
        
        return {
            'probabilities': probs,
            'prediction': int(np.argmax(probs)),
            'confidence': float(np.max(probs))
        }
    
    def cleanup(self):
        """Release model resources"""
        if self.language_model:
            del self.language_model
        if self.accent_model:
            del self.accent_model
        self.models_loaded = False
        
        # Clear TF session
        if tf:
            tf.keras.backend.clear_session()
        
        logger.info("🗑️ Predictor resources released")


# ============================================
# Standalone Test
# ============================================

