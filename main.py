"""
🧠 EpigrafIA Backend - FastAPI Server (Clean Architecture)
===========================================================
API para detección de idioma, acento y spoofing usando Deep Learning

This version uses Clean Architecture with SOLID principles:
- Domain Layer: Entities and interfaces (no dependencies)
- Application Layer: Use cases (business logic)
- Infrastructure Layer: TensorFlow, Librosa implementations
- Presentation Layer: FastAPI routes and controllers
- DI Container: Dependency injection

Endpoints:
- POST /api/analyze - Analiza audio y devuelve predicciones
- GET /api/health - Estado del servidor
- GET /api/models/status - Estado de los modelos cargados
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.di.containers import Container
from src.infrastructure.config.settings import Settings
from src.presentation.api.routes import health_routes, analysis_routes
from src.domain.entities.model_metadata import ModelType

# ============================================
# Configuration
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# Dependency Injection Container
# ============================================

# Initialize DI container
container = Container()

# Load settings
settings = container.settings()

# Wire modules (connect Depends() in routes to container)
container.wire(modules=[
    health_routes,
    analysis_routes
])

logger.info("✅ DI Container initialized and wired")

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="EpigrafIA API",
    description="API de detección de idioma, acento y spoofing con Deep Learning (Clean Architecture)",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_routes.router)
app.include_router(analysis_routes.router)

logger.info("✅ FastAPI app configured with routes")

# ============================================
# Startup / Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup using dependency injection."""
    logger.info("🚀 Starting EpigrafIA API v2.0 (Clean Architecture)...")

    # Get repositories from DI container
    model_repo = container.model_repository()
    config_repo = container.config_repository()

    # Load models
    models_to_load = [
        (ModelType.LANGUAGE, settings.language_model_path),
        (ModelType.SPOOFING, settings.spoofing_model_path)
    ]

    for model_type, model_path in models_to_load:
        if model_path and model_path.exists():
            try:
                model_repo.load_model(model_type, model_path)
                logger.info(f"✅ Loaded {model_type.value} model from {model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_type.value} model: {e}")
        else:
            logger.info(f"⏭️ Skipping {model_type.value} model (not found or not configured)")

    # Check if at least language model is loaded
    if model_repo.is_model_loaded(ModelType.LANGUAGE):
        logger.info("✅ Models loaded successfully! API is ready.")
    else:
        logger.error("❌ Language model not loaded. API will start but predictions won't work.")

    logger.info("=" * 60)
    logger.info("🎉 EpigrafIA API is running!")
    logger.info(f"📍 Port: {settings.port}")
    logger.info(f"📍 Docs: http://localhost:{settings.port}/api/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 EpigrafIA API shutting down...")

    # Get model repository and cleanup
    model_repo = container.model_repository()
    model_repo.cleanup()

    logger.info("✅ Cleanup complete. Goodbye!")


# ============================================
# Root Endpoint
# ============================================

@app.get("/")
async def root():
    """
    Root endpoint - API info.

    Returns basic information about the API and available endpoints.
    """
    return {
        "name": "EpigrafIA API",
        "version": "2.0.0",
        "architecture": "Clean Architecture with SOLID principles",
        "status": "running",
        "features": [
            "language_detection",
            "spoofing_detection"
        ],
        "endpoints": {
            "analyze": "POST /api/analyze",
            "health": "GET /api/health",
            "models": "GET /api/models/status",
            "docs": "GET /api/docs"
        },
        "principles": [
            "Single Responsibility Principle",
            "Open/Closed Principle",
            "Liskov Substitution Principle",
            "Interface Segregation Principle",
            "Dependency Inversion Principle"
        ]
    }


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Disable reload in production
    )
