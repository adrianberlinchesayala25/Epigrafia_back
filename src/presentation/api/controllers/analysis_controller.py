"""Controller for audio analysis endpoints."""
import logging
from typing import Dict, Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ....application.use_cases.analyze_audio_use_case import AnalyzeAudioUseCase
from ...formatters.prediction_formatter import PredictionFormatter

logger = logging.getLogger(__name__)


class AnalysisController:
    """
    Controller for audio analysis endpoints.

    Single Responsibility: Handle analysis request/response logic.
    Dependency Inversion: Depends on use case interface.
    """

    def __init__(
        self,
        analyze_use_case: AnalyzeAudioUseCase,
        formatter: PredictionFormatter
    ):
        """
        Initialize analysis controller.

        Args:
            analyze_use_case: Use case for audio analysis
            formatter: Formatter for prediction responses
        """
        self._use_case = analyze_use_case
        self._formatter = formatter

    async def analyze(self, audio_data: bytes, filename: str) -> JSONResponse:
        """
        Analyze audio file and return prediction results.

        Args:
            audio_data: Raw audio bytes
            filename: Original filename

        Returns:
            JSONResponse with prediction results

        Raises:
            HTTPException: If validation fails or prediction errors occur
        """
        # Validation
        if len(audio_data) == 0:
            logger.warning("Received empty audio file")
            raise HTTPException(status_code=400, detail="Empty audio file")

        logger.info(f"📥 Received audio: {filename} ({len(audio_data)} bytes)")

        try:
            # Execute use case
            prediction = self._use_case.execute(audio_data, filename)

            # Format response
            response = self._formatter.format_prediction(prediction)

            logger.info(f"✅ Prediction: {prediction.language.detected_language} "
                       f"({prediction.language.confidence*100:.1f}%)")

            return JSONResponse(content=response)

        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        except RuntimeError as e:
            logger.error(f"❌ Runtime error: {e}")
            raise HTTPException(status_code=500, detail=f"Model error: {e}")

        except Exception as e:
            logger.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")
