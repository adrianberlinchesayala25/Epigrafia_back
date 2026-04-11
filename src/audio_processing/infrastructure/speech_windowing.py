"""Helpers to focus language features on the speech segment."""

import numpy as np


def select_speech_window(
    y: np.ndarray,
    sample_rate: int,
    logger,
    frame_ms: int = 30,
    threshold_ratio: float = 0.12,
    min_rms: float = 1e-4,
) -> np.ndarray:
    """Trim leading silence by finding the first speech-like frame."""
    if y.size == 0:
        return y

    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    total_frames = y.size // frame_size
    if total_frames <= 1:
        return y

    frames = y[: total_frames * frame_size].reshape(total_frames, frame_size)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    max_rms = float(np.max(rms))
    if max_rms < min_rms:
        return y

    speech_threshold = max(min_rms, max_rms * threshold_ratio)
    speech_frames = np.where(rms >= speech_threshold)[0]

    if speech_frames.size == 0:
        return y

    first_speech_sample = int(speech_frames[0] * frame_size)
    pre_roll = int(sample_rate * 0.15)
    start_sample = max(0, first_speech_sample - pre_roll)

    if start_sample > 0:
        logger.info(
            f"   Speech aligned: removed {start_sample / sample_rate:.2f}s leading silence"
        )

    return y[start_sample:]
