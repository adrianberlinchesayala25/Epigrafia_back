"""
🧠 EpigrafIA Predictor
======================
Audio prediction module for language, accent and spoofing detection

Uses trained Keras models to predict:
- Language (Español, Inglés, Francés, Alemán)
- Accent (regional variants for each language)
- Spoofing (human vs AI-generated voice)
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
    Audio prediction class for language, accent and spoofing detection
    
    Attributes:
        language_model: Keras model for language detection
        accent_model: Keras model for accent detection
        spoofing_model: Keras model for spoofing detection
        models_loaded: Whether models are successfully loaded
    """
    
    # Audio processing configuration (must match training)
    SAMPLE_RATE = 16000
    DURATION = 3  # seconds
    N_MFCC = 40
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    
    # Spoofing model configuration (must match training)
    SPOOFING_MAX_LENGTH = 400  # Time steps (frames)
    SPOOFING_N_MFCC = 40       # MFCC coefficients
    
    def __init__(
        self,
        language_model_path: Optional[Union[str, Path]] = None,
        accent_model_path: Optional[Union[str, Path]] = None,
        spoofing_model_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize predictor with model paths
        
        Args:
            language_model_path: Path to language detection model (.keras or .h5)
            accent_model_path: Path to accent detection model (.keras or .h5)
            spoofing_model_path: Path to spoofing detection model (.keras or .h5)
        """
        self.language_model = None
        self.accent_model = None
        self.spoofing_model = None
        self.models_loaded = False
        
        # Load TensorFlow
        _load_tensorflow()
        
        # Load models if paths provided
        if language_model_path:
            self.load_language_model(language_model_path)
        if accent_model_path:
            self.load_accent_model(accent_model_path)
        if spoofing_model_path:
            self.load_spoofing_model(spoofing_model_path)
        
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
    
    def load_spoofing_model(self, model_path: Union[str, Path]) -> None:
        """Load spoofing detection model"""
        import keras
        from keras import layers
        
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Spoofing model not found: {path}")
        
        logger.info(f"Loading spoofing model from {path}")
        
        # Define custom layers that were used during training
        @keras.saving.register_keras_serializable()
        class SpecAugment(keras.layers.Layer):
            """SpecAugment layer for inference (no augmentation)"""
            def __init__(self, time_mask_param=40, freq_mask_param=8, **kwargs):
                super().__init__(**kwargs)
                self.time_mask_param = time_mask_param
                self.freq_mask_param = freq_mask_param
            
            def call(self, x, training=False):
                # No augmentation during inference
                return x
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    'time_mask_param': self.time_mask_param,
                    'freq_mask_param': self.freq_mask_param
                })
                return config
        
        @keras.saving.register_keras_serializable()
        class SEBlock(keras.layers.Layer):
            """Squeeze-and-Excitation block for inference"""
            def __init__(self, ratio=8, **kwargs):
                super().__init__(**kwargs)
                self.ratio = ratio
                self.channels = None
                self.fc1 = None
                self.fc2 = None
            
            def build(self, input_shape):
                self.channels = input_shape[-1]
                self.fc1 = layers.Dense(self.channels // self.ratio, activation='relu')
                self.fc2 = layers.Dense(self.channels, activation='sigmoid')
                super().build(input_shape)
            
            def call(self, x):
                # Use static shape to determine input dimensionality
                input_rank = len(x.shape)
                
                if input_rank == 2:
                    # 2D input: x shape is (batch, features)
                    # No squeeze needed, apply FC layers directly
                    excitation = self.fc1(x)
                    excitation = self.fc2(excitation)
                    return x * excitation
                else:
                    # 3D input: x shape is (batch, time, features)
                    # Squeeze: global average pooling across time dimension
                    squeeze = tf.reduce_mean(x, axis=1, keepdims=True)
                    # Excitation
                    excitation = self.fc1(squeeze)
                    excitation = self.fc2(excitation)
                    return x * excitation
            
            def get_config(self):
                config = super().get_config()
                config.update({'ratio': self.ratio})
                return config
        
        # Custom objects for loading
        custom_objects = {
            'SpecAugment': SpecAugment,
            'SEBlock': SEBlock
        }
        
        try:
            self.spoofing_model = keras.models.load_model(
                str(path), 
                compile=False,
                custom_objects=custom_objects
            )
            logger.info(f"   Input shape: {self.spoofing_model.input_shape}")
            logger.info(f"   Output shape: {self.spoofing_model.output_shape}")
        except Exception as e:
            logger.error(f"   Failed to load spoofing model: {e}")
            self.spoofing_model = None
    
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
        is_webm = audio_data[:4] == b'\x1a\x45\xdf\xa3'
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
            Dictionary with language, accent and spoofing predictions
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
        lang_names = ['Español', 'Inglés', 'Francés', 'Alemán']
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
        
        # Run spoofing detection if model available
        if self.spoofing_model is not None:
            try:
                spoofing_result = self.predict_spoofing(audio_data)
                result.update({
                    'spoofing': spoofing_result
                })
            except Exception as e:
                logger.error(f"Spoofing prediction failed: {e}")
                result.update({
                    'spoofing': None
                })
        else:
            result.update({
                'spoofing': None
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
    
    def extract_spoofing_features(self, audio_data: bytes) -> np.ndarray:
        """
        Extract MFCC features for spoofing detection
        Uses different configuration optimized for spoofing model
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            numpy array of MFCC features shaped (1, MAX_LENGTH, N_MFCC)
        """
        _load_librosa()
        
        # Detect if WebM by checking magic bytes
        is_webm = audio_data[:4] == b'\x1a\x45\xdf\xa3'
        suffix = '.webm' if is_webm else '.wav'
        
        # Write original audio to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        wav_path = None
        try:
            # If WebM, convert to WAV using pydub
            if is_webm:
                try:
                    from pydub import AudioSegment
                    audio_segment = AudioSegment.from_file(tmp_path, format="webm")
                    audio_segment = audio_segment.set_channels(1).set_frame_rate(self.SAMPLE_RATE)
                    wav_path = tmp_path.replace('.webm', '.wav')
                    audio_segment.export(wav_path, format="wav")
                    audio_path = wav_path
                except Exception as e:
                    logger.warning(f"WebM conversion failed: {e}")
                    audio_path = tmp_path
            else:
                audio_path = tmp_path
            
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE)
            
            # Normalize audio
            y = y / (np.max(np.abs(y)) + 1e-8)
            
            # Extract MFCCs for spoofing
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.SPOOFING_N_MFCC,
                hop_length=512,
                n_fft=2048
            )
            
            # Transpose to (time, features)
            mfcc = mfcc.T
            
            # Pad or truncate to fixed length
            if mfcc.shape[0] < self.SPOOFING_MAX_LENGTH:
                pad_width = self.SPOOFING_MAX_LENGTH - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:self.SPOOFING_MAX_LENGTH, :]
            
            # Normalize features
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            
            # Add batch dimension
            return mfcc[np.newaxis, ...]
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
            if wav_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
    
    def predict_spoofing(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Predict if audio is genuine human voice or AI-generated (spoofed)
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Dictionary with spoofing prediction:
                - is_genuine: True if human, False if spoofed
                - genuine_probability: Probability of being genuine human voice
                - spoof_probability: Probability of being AI-generated
                - confidence: Confidence of prediction (0-1)
                - label: "Humano" or "Artificial"
        """
        if self.spoofing_model is None:
            raise RuntimeError("Spoofing model not loaded")
        
        # Extract features for spoofing
        features = self.extract_spoofing_features(audio_data)
        
        # Run prediction
        prediction = self.spoofing_model.predict(features, verbose=0)[0]
        
        # Model outputs single sigmoid (probability of being SPOOF)
        if len(prediction.shape) == 0:
            spoof_prob = float(prediction)
        else:
            spoof_prob = float(prediction[0])
        
        genuine_prob = 1.0 - spoof_prob
        is_genuine = genuine_prob > 0.5
        confidence = genuine_prob if is_genuine else spoof_prob
        
        logger.info(f"🔍 Spoofing detection:")
        logger.info(f"   Humano: {genuine_prob*100:.1f}%")
        logger.info(f"   Artificial: {spoof_prob*100:.1f}%")
        logger.info(f"   Resultado: {'Humano ✓' if is_genuine else 'Artificial ⚠'}")
        
        return {
            'is_genuine': is_genuine,
            'genuine_probability': genuine_prob,
            'spoof_probability': spoof_prob,
            'confidence': confidence,
            'label': "Humano" if is_genuine else "Artificial"
        }
    
    def cleanup(self):
        """Release model resources"""
        if self.language_model:
            del self.language_model
        if self.accent_model:
            del self.accent_model
        if self.spoofing_model:
            del self.spoofing_model
        self.models_loaded = False
        
        # Clear TF session
        if tf:
            tf.keras.backend.clear_session()
        
        logger.info("🗑️ Predictor resources released")


# ============================================
# Standalone Test
# ============================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        predictor = AudioPredictor(
            language_model_path="outputs/models_trained/language_model.keras",
            accent_model_path="outputs/models_trained/accent_model.keras"
        )
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        result = predictor.predict(audio_data)
        
        print("\n📊 Prediction Results:")
        print(f"   Language: {result['language_prediction']} ({result['language_confidence']:.2%})")
        print(f"   Accent: {result['accent_prediction']} ({result['accent_confidence']:.2%})")
        
    else:
        print("Usage: python predict.py <audio_file>")
