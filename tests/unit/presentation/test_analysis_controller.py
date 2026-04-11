import pytest
from fastapi import HTTPException

from src.domain.entities.prediction import CompletePrediction, LanguagePrediction
from src.presentation.api.controllers.analysis_controller import AnalysisController


class FakeUseCase:
    def execute(self, audio_data, filename):
        return CompletePrediction(
            language=LanguagePrediction(
                detected_language="Español",
                confidence=0.77,
                probabilities={"Español": 0.77, "Inglés": 0.23},
                prediction_index=0,
            ),
            spoofing=None,
        )


class FakeFormatter:
    def format_prediction(self, prediction):
        return {"success": True, "language": {"detected": prediction.language.detected_language}}


@pytest.mark.asyncio
async def test_analyze_returns_json_response():
    controller = AnalysisController(FakeUseCase(), FakeFormatter())

    response = await controller.analyze(b"raw-audio", "clip.wav")

    assert response.status_code == 200
    assert b'"success":true' in response.body


@pytest.mark.asyncio
async def test_analyze_rejects_empty_audio():
    controller = AnalysisController(FakeUseCase(), FakeFormatter())

    with pytest.raises(HTTPException) as exc:
        await controller.analyze(b"", "clip.wav")

    assert exc.value.status_code == 400
