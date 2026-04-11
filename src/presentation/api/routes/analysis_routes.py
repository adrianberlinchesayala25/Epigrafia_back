"""FastAPI routes for audio analysis endpoints."""
from fastapi import APIRouter, File, UploadFile, Depends
from dependency_injector.wiring import inject, Provide

from ..controllers.analysis_controller import AnalysisController

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze")
@inject
async def analyze_audio(
    audio: UploadFile = File(...),
    controller: AnalysisController = Depends(Provide["controllers.analysis"])
):
    """
    Analyze audio file for language and spoofing detection.

    Accepts audio files in WAV, MP3, WebM, or OGG format.

    Returns:
        JSON response with:
        - Language detection (detected language + confidence + probabilities)
        - Spoofing detection (human vs AI-generated, if model available)
    """
    audio_data = await audio.read()
    return await controller.analyze(audio_data, audio.filename or "unknown")
