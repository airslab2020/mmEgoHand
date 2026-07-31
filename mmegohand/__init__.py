"""mmEgoHand research implementation."""

from .configuration import ExperimentConfig, load_config
from .gesture import build_gesture_model
from .model import MMEgoHandPose, ModelConfig, build_model

__all__ = [
    "ExperimentConfig",
    "MMEgoHandPose",
    "ModelConfig",
    "build_gesture_model",
    "build_model",
    "load_config",
]

__version__ = "1.0.0"
