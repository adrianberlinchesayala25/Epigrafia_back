"""Temporary audio file handling and format conversion."""

import os
import tempfile
from pathlib import Path


def write_temp_audio(audio_bytes: bytes, sample_rate: int, logger) -> Path:
    """Write raw audio bytes to temp file and convert webm to wav when needed."""
    is_webm = audio_bytes[:4] == b"\x1a\x45\xdf\xa3"
    suffix = ".webm" if is_webm else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    if not is_webm:
        return tmp_path

    logger.info("Converting WebM to WAV...")
    try:
        from pydub import AudioSegment

        audio_segment = AudioSegment.from_file(str(tmp_path), format="webm")
        audio_segment = audio_segment.set_channels(1).set_frame_rate(sample_rate)

        wav_path = tmp_path.with_suffix(".wav")
        audio_segment.export(str(wav_path), format="wav")
        os.unlink(tmp_path)
        return wav_path
    except Exception as exc:
        logger.warning(f"WebM conversion failed: {exc}, using original file")
        return tmp_path


def cleanup_temp_file(file_path: Path, logger) -> None:
    """Delete a temporary file if it exists."""
    try:
        if file_path.exists():
            os.unlink(file_path)
    except Exception as exc:
        logger.warning(f"Failed to cleanup temp file {file_path}: {exc}")
