"""
EpigrafIA Backend - FastAPI Server
=====================================
API para deteccion de idioma y acento usando Deep Learning

Endpoints:
- POST /api/analyze - Analiza audio y devuelve predicciones
- GET /api/health - Estado del servidor
- GET /api/models/status - Estado de los modelos cargados
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import from same package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import AudioPredictor

# ============================================
# Configuration
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# Labels for predictions
LANGUAGE_LABELS = ['EspaÃ±ol', 'InglÃ©s', 'FrancÃ©s', 'AlemÃ¡n']
ACCENT_LABELS = [
    'EspaÃ±a', 'MÃ©xico', 'UK', 'USA',
    'Francia', 'Quebec', 'Alemania', 'Austria'
]

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="EpigrafIA API",
    description="API de detecciÃ³n de idioma y acento con Deep Learning",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producciÃ³n, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[AudioPredictor] = None


# ============================================
# Startup / Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global predictor
    
    logger.info("ðŸš€ Starting EpigrafIA API...")
    
    # Try to find the best model first, then fall back to regular
  language_model_path = MODELS_DIR / "language_model.keras"
    
accent_model_path = MODELS_DIR / "accent_model.keras"
if not accent_model_path.exists():
    accent_model_path = None
    predictor = AudioPredictor(
    language_model_path=language_model_path,
    accent_model_path=accent_model_path
)

    try:
        predictor = AudioPredictor(
            language_model_path=language_model_path,
            accent_model_path=accent_model_path if accent_model_path.exists() else None
        )
        logger.info("âœ… Models loaded successfully!")
        
    except FileNotFoundError as e:
        logger.warning(f"âš ï¸ Models not found: {e}")
        logger.warning("   The API will start but predictions won't work.")
        logger.warning("   Train the models first using the notebooks.")
        
    except Exception as e:
        logger.error(f"âŒ Error loading models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global predictor
    if predictor:
        predictor.cleanup()
    logger.info("ðŸ‘‹ EpigrafIA API shutting down...")


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "EpigrafIA API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /api/analyze",
            "health": "GET /api/health",
            "models": "GET /api/models/status",
            "docs": "GET /api/docs"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": predictor is not None and predictor.models_loaded
    }


@app.get("/api/models/status")
async def models_status():
    """Get status of loaded models"""
    if predictor is None:
        return {
            "loaded": False,
            "error": "Predictor not initialized"
        }
    
    return {
        "loaded": predictor.models_loaded,
        "language_model": predictor.language_model is not None,
        "accent_model": predictor.accent_model is not None,
        "language_labels": LANGUAGE_LABELS,
        "accent_labels": ACCENT_LABELS
    }


@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze audio file and return language/accent predictions
    
    Accepts: WAV, MP3, WebM, OGG audio files
    Returns: JSON with predictions and probabilities
    """
    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check content type
    allowed_types = ['audio/wav', 'audio/webm', 'audio/mp3', 'audio/mpeg', 
                     'audio/ogg', 'audio/x-wav', 'audio/wave']
    
    content_type = audio.content_type or ''
    if not any(t in content_type for t in ['audio', 'webm', 'wav', 'mp3', 'ogg']):
        logger.warning(f"Unexpected content type: {content_type}")
    
    # Check if models are loaded
    if predictor is None or not predictor.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train the models first."
        )
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"ðŸ“¥ Received audio: {audio.filename} ({len(audio_data)} bytes)")
        
        # Run prediction
        result = predictor.predict(audio_data)
        
        # Format response - convert lists to numpy arrays for argmax/max
        import numpy as np
        language_probs = np.array(result['language_probabilities'])
        
        # Build language response
        lang_idx = int(language_probs.argmax())
        response = {
            "success": True,
            "language": {
                "detected": LANGUAGE_LABELS[lang_idx],
                "confidence": float(language_probs.max()),
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(LANGUAGE_LABELS, language_probs)
                }
            },
            "language_prediction": lang_idx,
            "language_confidence": float(language_probs.max()),
            "accent": {
                "detected": "No disponible",
                "confidence": 0.0,
                "probabilities": {}
            },
            "accent_prediction": 0,
            "accent_confidence": 0.0
        }
        
        # Add accent if available
        accent_probs_raw = result.get('accent_probabilities')
        if accent_probs_raw is not None:
            accent_probs = np.array(accent_probs_raw)
            accent_idx = int(accent_probs.argmax())
            response["accent"] = {
                "detected": ACCENT_LABELS[accent_idx] if accent_idx < len(ACCENT_LABELS) else "Desconocido",
                "confidence": float(accent_probs.max()),
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(ACCENT_LABELS, accent_probs)
                }
            }
            response["accent_prediction"] = accent_idx
            response["accent_confidence"] = float(accent_probs.max())
        
        # Log all probabilities for debugging
        probs_str = " | ".join([f"{LANGUAGE_LABELS[i]}: {language_probs[i]*100:.1f}%" for i in range(len(LANGUAGE_LABELS))])
        logger.info(f"ðŸ“Š Probabilities: {probs_str}")
        logger.info(f"âœ… Prediction: {response['language']['detected']} ({response['language']['confidence']*100:.1f}%)")
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        import traceback
        error_msg = str(e) or "Unknown error"
        logger.error(f"Prediction error: {type(e).__name__}: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {type(e).__name__}: {error_msg}")
