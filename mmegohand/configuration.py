"""Configuration loading shared by training and evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

from .model import ModelConfig


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    train_split: str = "splits/train.txt"
    test_split: str = "splits/test.txt"
    radar_dir: str = "mmwave"
    imu_dir: str = "imu"
    pseudo_label_dir: str = "pseudo_labels"
    pose_output_dir: str = "pose_outputs"
    raw_frames: int = 40
    model_frames: int = 30


@dataclass(frozen=True)
class LossConfig:
    matching_class_cost: float = 5.0
    matching_keypoint_cost: float = 25.0
    class_loss_weight: float = 1.0
    keypoint_loss_weight: float = 5.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    num_workers: int = 2
    drop_last_train: bool = True
    drop_last_eval: bool = True
    output_dir: str = "outputs/pose"


@dataclass(frozen=True)
class GestureConfig:
    architecture: str = "resnet50"
    num_classes: int = 8
    batch_size: int = 32
    epochs: int = 500
    learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-2
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    drop_last_train: bool = True
    drop_last_eval: bool = True
    legacy_split_denominator: bool = True
    output_dir: str = "outputs/gesture"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    training: TrainingConfig
    gesture: GestureConfig


ConfigType = TypeVar("ConfigType")


def _construct(config_type: Type[ConfigType], values: Dict[str, Any]) -> ConfigType:
    known = {field.name for field in fields(config_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"unknown {config_type.__name__} keys: {', '.join(sorted(unknown))}"
        )
    return config_type(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    expected = {"model", "data", "loss", "training", "gesture"}
    unknown = set(raw) - expected
    if unknown:
        raise ValueError(f"unknown top-level keys: {', '.join(sorted(unknown))}")
    return ExperimentConfig(
        model=_construct(ModelConfig, raw.get("model", {})),
        data=_construct(DataConfig, raw.get("data", {})),
        loss=_construct(LossConfig, raw.get("loss", {})),
        training=_construct(TrainingConfig, raw.get("training", {})),
        gesture=_construct(GestureConfig, raw.get("gesture", {})),
    )
