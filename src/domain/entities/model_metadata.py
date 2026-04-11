"""Domain entities for model metadata and configuration."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ModelType(Enum):
    """Types of ML models supported by the system."""
    LANGUAGE = "language"
    SPOOFING = "spoofing"


@dataclass
class ModelMetadata:
    """
    Domain model representing metadata about a loaded ML model.

    Contains information about the model type, loading status,
    configuration, and input/output specifications.
    """
    model_type: ModelType
    model_path: Optional[Path]
    labels: List[str]
    input_shape: Optional[tuple]
    is_loaded: bool

    def __post_init__(self):
        """Validate model metadata."""
        if not isinstance(self.model_type, ModelType):
            raise ValueError("model_type must be a ModelType enum")
        if self.is_loaded and self.model_path is None:
            raise ValueError("Loaded model must have a path")
        if self.is_loaded and not self.labels:
            raise ValueError("Loaded model must have labels")
