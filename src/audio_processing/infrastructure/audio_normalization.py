"""Reusable audio normalization strategies."""

import numpy as np


def normalize_standard_audio(y: np.ndarray, logger) -> np.ndarray:
    """Two-stage normalization used by language feature extraction."""
    target_peak = 0.8
    audio_max = np.abs(y).max()

    if audio_max > 0.01:
        peak_factor = target_peak / audio_max
        y = y * peak_factor
        logger.info(f"   Peak normalized: factor={peak_factor:.2f}")

    audio_rms = np.sqrt(np.mean(y**2))
    target_rms = 0.08

    if audio_rms > 0.001 and audio_rms < target_rms * 0.5:
        rms_factor = min(target_rms / audio_rms, 3.0)
        y = y * rms_factor
        y = np.clip(y, -1.0, 1.0)
        new_rms = np.sqrt(np.mean(y**2))
        logger.info(f"   RMS boosted: factor={rms_factor:.2f}, final RMS={new_rms:.6f}")
        return y

    logger.info(f"   Audio level OK: RMS={audio_rms:.6f}")
    return y


def normalize_peak_audio(y: np.ndarray) -> np.ndarray:
    """Simple peak normalization used by spoofing feature extraction."""
    return y / (np.max(np.abs(y)) + 1e-8)
