"""Dependency Injection container for the application."""
from pathlib import Path

from dependency_injector import containers, providers

# Infrastructure
from ..infrastructure.config.settings import Settings
from ..infrastructure.persistence.yaml_config_repository import YamlConfigRepository
from ..infrastructure.ml.tensorflow_model_repository import TensorFlowModelRepository
from ..infrastructure.ml.tensorflow_predictor import TensorFlowPredictorService
from ..infrastructure.audio.librosa_feature_extractor import LibrosaFeatureExtractor

# Application
from ..application.use_cases.analyze_audio_use_case import AnalyzeAudioUseCase

# Presentation
from ..presentation.formatters.prediction_formatter import PredictionFormatter
from ..presentation.api.controllers.health_controller import HealthController
from ..presentation.api.controllers.analysis_controller import AnalysisController


class Container(containers.DeclarativeContainer):
    """
    Dependency Injection container.

    This is the heart of the Clean Architecture implementation.
    It wires all dependencies together, ensuring proper dependency flow:

    Presentation → Application → Domain ← Infrastructure

    The container follows these principles:
    - Dependency Inversion: High-level modules depend on abstractions (interfaces)
    - Single Responsibility: Each provider creates one thing
    - Open/Closed: Easy to add new services without modifying existing code
    """

    # ========== CONFIGURATION ==========

    config = providers.Configuration()

    settings = providers.Singleton(
        Settings
    )

    config_repository = providers.Singleton(
        YamlConfigRepository,
        config_path=Path("config.yaml")
    )

    # ========== INFRASTRUCTURE LAYER - ML ==========

    model_repository = providers.Singleton(
        TensorFlowModelRepository
    )

    predictor_service = providers.Singleton(
        TensorFlowPredictorService,
        model_repository=model_repository,
        config_repository=config_repository
    )

    # ========== INFRASTRUCTURE LAYER - AUDIO ==========

    feature_extractor = providers.Singleton(
        LibrosaFeatureExtractor,
        settings=settings
    )

    # ========== APPLICATION LAYER - USE CASES ==========

    analyze_audio_use_case = providers.Factory(
        AnalyzeAudioUseCase,
        feature_extractor=feature_extractor,
        predictor_service=predictor_service
    )

    # ========== PRESENTATION LAYER - FORMATTERS ==========

    prediction_formatter = providers.Singleton(
        PredictionFormatter
    )

    # ========== PRESENTATION LAYER - CONTROLLERS ==========

    health_controller = providers.Factory(
        HealthController,
        model_repository=model_repository,
        config_repository=config_repository
    )

    analysis_controller = providers.Factory(
        AnalysisController,
        analyze_use_case=analyze_audio_use_case,
        formatter=prediction_formatter
    )

    # Controllers container for dependency injection in routes
    controllers = providers.Dict(
        health=health_controller,
        analysis=analysis_controller
    )
