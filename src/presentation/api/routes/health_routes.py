"""FastAPI routes for health check endpoints."""
from fastapi import APIRouter, Request

from ..controllers.health_controller import HealthController

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health_check(
    request: Request,
):
    """
    Health check endpoint.

    Returns API status and whether models are loaded.
    """
    controller: HealthController = request.app.state.container.health_controller()
    return await controller.check_health()


@router.get("/models/status")
async def models_status(
    request: Request,
):
    """
    Model status endpoint.

    Returns detailed information about which models are loaded
    and available labels for each model.
    """
    controller: HealthController = request.app.state.container.health_controller()
    return await controller.get_models_status()
