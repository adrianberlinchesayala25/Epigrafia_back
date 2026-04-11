"""Feature tensor builders for language and spoofing models."""

import numpy as np
import librosa


def fit_audio_to_duration(y: np.ndarray, sample_rate: int, duration_seconds: int) -> np.ndarray:
    """Pad or trim audio to a fixed duration."""
    min_samples = sample_rate * duration_seconds
    if len(y) < min_samples:
        return np.pad(y, (0, min_samples - len(y)), mode="constant")
    return y[:min_samples]


def build_language_features(y: np.ndarray, settings) -> np.ndarray:
    """Build language/accent features with MFCC + deltas."""
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=settings.sample_rate,
        n_mfcc=settings.n_mfcc,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
    )

    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    features = (features - features.mean()) / (features.std() + 1e-8)
    features = np.expand_dims(features.T, axis=0)
    return features


def build_spoofing_features(y: np.ndarray, sample_rate: int, n_mfcc: int, max_length: int) -> np.ndarray:
    """Build spoofing MFCC tensor with fixed temporal length."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=512,
        n_fft=2048,
    ).T

    if mfcc.shape[0] < max_length:
        pad_width = max_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_length, :]

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc[np.newaxis, ...]
