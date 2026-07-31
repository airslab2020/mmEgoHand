"""Dataset readers for synchronized radar, IMU, and pose sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .configuration import DataConfig


@dataclass(frozen=True)
class SampleRecord:
    name: str
    hand_setting: str
    scene: str
    gesture: int
    repetition: int


def parse_split(path: str | Path) -> List[SampleRecord]:
    records = []
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                raise ValueError(
                    f"{split_path}:{line_number}: expected 5 comma-separated fields"
                )
            name, hand_setting, scene, gesture, repetition = parts
            expected_name = "_".join(
                (hand_setting, scene, gesture, repetition)
            )
            if name != expected_name:
                raise ValueError(
                    f"{split_path}:{line_number}: {name!r} does not match "
                    f"metadata {expected_name!r}"
                )
            records.append(
                SampleRecord(
                    name=name,
                    hand_setting=hand_setting,
                    scene=scene,
                    gesture=int(gesture),
                    repetition=int(repetition),
                )
            )
    if not records:
        raise ValueError(f"split is empty: {split_path}")
    return records


def uniform_indices(source_frames: int, target_frames: int) -> np.ndarray:
    if source_frames <= 0 or target_frames <= 0:
        raise ValueError("frame counts must be positive")
    return np.linspace(0, source_frames - 1, target_frames, dtype=int)


def _resolve_existing(root: Path, directories: Sequence[str], name: str) -> Path:
    checked = []
    for directory in directories:
        path = root / directory / name
        checked.append(path)
        if path.exists():
            return path
    paths = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(f"sample file not found; checked: {paths}")


def _normalize_pseudo_labels(array: np.ndarray, frames: int) -> np.ndarray:
    if array.shape[0] != frames:
        array = array[uniform_indices(array.shape[0], frames)]
    if array.ndim == 3 and array.shape[1:] == (21, 3):
        return array.reshape(frames, 1, 63)
    if array.ndim == 3 and array.shape[1:] == (42, 3):
        return array.reshape(frames, 2, 63)
    if array.ndim == 4 and array.shape[2:] == (21, 3):
        return array.reshape(frames, array.shape[1], 63)
    if array.ndim == 3 and array.shape[-1] == 63:
        return array
    raise ValueError(
        "pseudo labels must have shape [T,21,3], [T,42,3], "
        f"[T,H,21,3], or [T,H,63], got {array.shape}"
    )


class PoseDataset(Dataset):
    def __init__(self, config: DataConfig, split_path: str | Path) -> None:
        self.config = config
        self.root = Path(config.root)
        self.records = parse_split(split_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        record = self.records[index]
        radar = self._load_radar(record.name)
        imu = self._load_imu(record.name)
        pseudo_labels = self._load_pseudo_labels(record.name)
        hand_count = pseudo_labels.shape[1]
        target = {
            "kpt": torch.from_numpy(pseudo_labels).float(),
            "kpt_cls": torch.ones(
                self.config.model_frames, hand_count, 1, dtype=torch.long
            ),
            "label": torch.tensor(record.gesture - 1, dtype=torch.long),
            "filename": torch.tensor(
                [
                    int(record.hand_setting),
                    int(record.scene),
                    record.gesture,
                    record.repetition,
                ],
                dtype=torch.long,
            ),
        }
        return radar, imu, target

    def _load_radar(self, name: str) -> Tensor:
        path = self.root / self.config.radar_dir / f"{name}.mat"
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"{path} does not contain an HDF5 dataset named 'data'")
            radar = np.asarray(handle["data"]).transpose((2, 1, 0))
        if radar.shape[0] != self.config.raw_frames:
            raise ValueError(
                f"{path}: expected {self.config.raw_frames} frames, got {radar.shape[0]}"
            )
        radar = radar[
            uniform_indices(self.config.raw_frames, self.config.model_frames)
        ]
        return torch.from_numpy(np.ascontiguousarray(radar)).float()

    def _load_imu(self, name: str) -> Tensor:
        path = self.root / self.config.imu_dir / f"{name}.npy"
        imu = np.load(path)
        if imu.shape[0] != self.config.raw_frames:
            raise ValueError(
                f"{path}: expected {self.config.raw_frames} frames, got {imu.shape[0]}"
            )
        imu = imu[
            uniform_indices(self.config.raw_frames, self.config.model_frames)
        ].reshape(self.config.model_frames, -1)
        if imu.shape[1] != 6:
            raise ValueError(f"{path}: expected six IMU channels, got {imu.shape}")
        return torch.from_numpy(np.ascontiguousarray(imu)).float()

    def _load_pseudo_labels(self, name: str) -> np.ndarray:
        path = _resolve_existing(
            self.root,
            (
                self.config.pseudo_label_dir,
                "kpt_gt",
                "keypoint_gt",
            ),
            f"{name}.npy",
        )
        return _normalize_pseudo_labels(
            np.load(path), self.config.model_frames
        ).astype(np.float32, copy=False)


class GesturePoseDataset(Dataset):
    """Read exported pose sequences for the paper's ResNet-50 classifier."""

    left_hand_settings = frozenset({"02", "04", "05", "12"})

    def __init__(self, config: DataConfig, split_path: str | Path) -> None:
        self.config = config
        self.root = Path(config.root)
        self.records = parse_split(split_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        path = _resolve_existing(
            self.root,
            (
                self.config.pose_output_dir,
                "keypoint_output",
                "kpt_output",
                "kpt",
            ),
            f"{record.name}.npy",
        )
        pose = np.load(path)
        if pose.shape[0] != self.config.model_frames:
            pose = pose[
                uniform_indices(pose.shape[0], self.config.model_frames)
            ]
        pose = pose.reshape(self.config.model_frames, -1, 3)
        if pose.shape[1] == 21:
            empty_hand = np.zeros_like(pose)
            if record.hand_setting in self.left_hand_settings:
                pose = np.concatenate((pose, empty_hand), axis=1)
            else:
                pose = np.concatenate((empty_hand, pose), axis=1)
        elif pose.shape[1] != 42:
            raise ValueError(f"{path}: expected 21 or 42 joints, got {pose.shape}")
        return (
            torch.from_numpy(np.ascontiguousarray(pose)).float(),
            torch.tensor(record.gesture - 1, dtype=torch.long),
        )


def collate_pose_batch(
    batch: Iterable[tuple[Tensor, Tensor, Dict[str, Tensor]]],
) -> tuple[Tensor, Tensor, List[Dict[str, Tensor]]]:
    radar, imu, targets = zip(*batch)
    return torch.stack(radar), torch.stack(imu), list(targets)
