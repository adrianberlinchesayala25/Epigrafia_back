"""Audio format detection helpers."""

from ...domain.entities.audio import AudioFormat


def detect_audio_format(filename: str) -> AudioFormat:
    """Detect supported audio format from filename extension."""
    filename_lower = filename.lower()

    format_by_extension = {
        ".wav": AudioFormat.WAV,
        ".mp3": AudioFormat.MP3,
        ".webm": AudioFormat.WEBM,
        ".ogg": AudioFormat.OGG,
    }

    for extension, detected_format in format_by_extension.items():
        if filename_lower.endswith(extension):
            return detected_format

    return AudioFormat.UNKNOWN
