"""
 EpigrafIA Backend - FastAPI Server
=====================================
API para detección de idioma, acento y spoofing usando Deep Learning

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
sys.path.insert(0, str(Path(__file__).parent))
from backend.predict import AudioPredictor

# ============================================
# Configuration
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths - Ajustados para estructura de Render
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Labels for predictions
LANGUAGE_LABELS = ["Español", "Inglés", "Francés", "Alemán"]
ACCENT_LABELS = [
    "España", "México", "UK", "USA",
    "Francia", "Quebec", "Alemania", "Austria"
]

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="EpigrafIA API",
    description="API de detección de idioma, acento y spoofing con Deep Learning",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    
    logger.info(" Starting EpigrafIA API v2.0...")
    
    # Language model path
    language_model_path = MODELS_DIR / "language" / "language_model_best.keras"
    if not language_model_path.exists():
        language_model_path = MODELS_DIR / "language" / "language_model.keras"
    
    # Accent model path (opcional)
    accent_model_path = MODELS_DIR / "accent" / "accent_model.keras"
    
    # Spoofing model path
    spoofing_model_path = MODELS_DIR / "spoofing" / "spoofing_best.keras"
    
    logger.info(f" Models directory: {MODELS_DIR}")
    logger.info(f"   Language model: {language_model_path} (exists: {language_model_path.exists()})")
    logger.info(f"   Spoofing model: {spoofing_model_path} (exists: {spoofing_model_path.exists()})")
    
    try:
        predictor = AudioPredictor(
            language_model_path=language_model_path if language_model_path.exists() else None,
            accent_model_path=accent_model_path if accent_model_path.exists() else None,
            spoofing_model_path=spoofing_model_path if spoofing_model_path.exists() else None
        )
        logger.info(" Models loaded successfully!")
        
    except FileNotFoundError as e:
        logger.warning(f" Models not found: {e}")
        logger.warning("   The API will start but predictions won't work.")
        
    except Exception as e:
        logger.error(f" Error loading models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global predictor
    if predictor:
        predictor.cleanup()
    logger.info(" EpigrafIA API shutting down...")


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "EpigrafIA API",
        "version": "2.0.0",
        "status": "running",
        "features": ["language_detection", "accent_detection", "spoofing_detection"],
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
        "spoofing_model": predictor.spoofing_model is not None,
        "language_labels": LANGUAGE_LABELS,
        "accent_labels": ACCENT_LABELS
    }


@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze audio file and return language/accent/spoofing predictions
    
    Accepts: WAV, MP3, WebM, OGG audio files
    Returns: JSON with predictions and probabilities
    """
    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check if models are loaded
    if predictor is None or not predictor.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f" Received audio: {audio.filename} ({len(audio_data)} bytes)")
        
        # Run prediction
        result = predictor.predict(audio_data)
        
        # Format response
        import numpy as np
        language_probs = np.array(result["language_probabilities"])
        
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
        accent_probs_raw = result.get("accent_probabilities")
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
        
        # Add spoofing detection if available
        spoofing_result = result.get("spoofing")
        if spoofing_result is not None:
            response["spoofing"] = {
                "is_genuine": spoofing_result["is_genuine"],
                "label": spoofing_result["label"],
                "confidence": spoofing_result["confidence"],
                "genuine_probability": spoofing_result["genuine_probability"],
                "spoof_probability": spoofing_result["spoof_probability"]
            }
            logger.info(f" Spoofing: {spoofing_result['label']} ({spoofing_result['confidence']*100:.1f}%)")
        else:
            response["spoofing"] = None
        
        # Log prediction
        logger.info(f" Prediction: {response['language']['detected']} ({response['language']['confidence']*100:.1f}%)")
        
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


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
