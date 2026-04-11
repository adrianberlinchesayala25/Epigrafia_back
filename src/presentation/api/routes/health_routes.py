"""FastAPI routes for health check endpoints."""
from fastapi import APIRouter, Depends
from dependency_injector.wiring import inject, Provide

from ..controllers.health_controller import HealthController

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
@inject
async def health_check(
    controller: HealthController = Depends(Provide["controllers.health"])
):
    """
    Health check endpoint.

    Returns API status and whether models are loaded.
    """
    return await controller.check_health()


@router.get("/models/status")
@inject
async def models_status(
    controller: HealthController = Depends(Provide["controllers.health"])
):
    """
    Model status endpoint.

    Returns detailed information about which models are loaded
    and available labels for each model.
    """
    return await controller.get_models_status()
