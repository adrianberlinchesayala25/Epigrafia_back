from src.audio_processing.application.audio_format_detector import detect_audio_format
from src.domain.entities.audio import AudioFormat


def test_detect_audio_format_supported_extensions():
    assert detect_audio_format("sample.wav") == AudioFormat.WAV
    assert detect_audio_format("sample.mp3") == AudioFormat.MP3
    assert detect_audio_format("sample.webm") == AudioFormat.WEBM
    assert detect_audio_format("sample.ogg") == AudioFormat.OGG


def test_detect_audio_format_fallback_unknown():
    assert detect_audio_format("sample.bin") == AudioFormat.UNKNOWN
