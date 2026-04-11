from src.domain.entities.prediction import (
    CompletePrediction,
    LanguagePrediction,
    SpoofingPrediction,
)
from src.presentation.formatters.prediction_formatter import PredictionFormatter


def test_format_prediction_keeps_legacy_fields_and_spoofing_block():
    formatter = PredictionFormatter()
    prediction = CompletePrediction(
        language=LanguagePrediction(
            detected_language="Español",
            confidence=0.92,
            probabilities={"Español": 0.92, "Inglés": 0.08},
            prediction_index=0,
        ),
        spoofing=SpoofingPrediction(
            is_genuine=True,
            label="Humano",
            confidence=0.88,
            genuine_probability=0.88,
            spoof_probability=0.12,
        ),
    )

    response = formatter.format_prediction(prediction)

    assert response["success"] is True
    assert response["language"]["detected"] == "Español"
    assert response["language_prediction"] == 0
    assert response["language_confidence"] == 0.92
    assert response["spoofing"]["label"] == "Humano"


def test_format_prediction_returns_none_when_spoofing_missing():
    formatter = PredictionFormatter()
    prediction = CompletePrediction(
        language=LanguagePrediction(
            detected_language="Español",
            confidence=0.92,
            probabilities={"Español": 0.92, "Inglés": 0.08},
            prediction_index=0,
        ),
        spoofing=None,
    )

    response = formatter.format_prediction(prediction)
    assert response["spoofing"] is None
