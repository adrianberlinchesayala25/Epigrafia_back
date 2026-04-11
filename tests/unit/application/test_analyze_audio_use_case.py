from src.application.use_cases.analyze_audio_use_case import AnalyzeAudioUseCase
from src.domain.entities.audio import AudioFeatures, FeatureType
from src.domain.entities.prediction import (
    CompletePrediction,
    LanguagePrediction,
    SpoofingPrediction,
)


class FakeFeatureExtractor:
    def __init__(self):
        self.calls = []

    def extract_features(self, audio_data, feature_type):
        self.calls.append((audio_data.filename, feature_type))
        return AudioFeatures(
            features=__import__("numpy").zeros((1, 10, 40)),
            feature_type=feature_type,
            shape=(1, 10, 40),
            sample_rate=16000,
        )


class FakePredictorService:
    def predict_language(self, features):
        return LanguagePrediction(
            detected_language="Español",
            confidence=0.95,
            probabilities={"Español": 0.95, "Inglés": 0.05},
            prediction_index=0,
        )

    def predict_spoofing(self, features):
        return SpoofingPrediction(
            is_genuine=True,
            label="Humano",
            confidence=0.9,
            genuine_probability=0.9,
            spoof_probability=0.1,
        )


def test_execute_runs_full_pipeline_and_preserves_contract():
    feature_extractor = FakeFeatureExtractor()
    predictor_service = FakePredictorService()
    use_case = AnalyzeAudioUseCase(feature_extractor, predictor_service)

    result = use_case.execute(b"audio-bytes", "voice.wav")

    assert isinstance(result, CompletePrediction)
    assert result.language.detected_language == "Español"
    assert result.spoofing is not None
    assert feature_extractor.calls == [
        ("voice.wav", FeatureType.LANGUAGE),
        ("voice.wav", FeatureType.SPOOFING),
    ]
