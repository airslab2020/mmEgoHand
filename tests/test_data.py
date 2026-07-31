from pathlib import Path

import h5py
import numpy as np

from mmegohand.configuration import DataConfig
from mmegohand.data import GesturePoseDataset, PoseDataset


def write_split(path: Path, name: str) -> None:
    path.write_text(f"{name},{name.replace('_', ',')}\n", encoding="utf-8")


def test_pose_dataset_shapes(tmp_path: Path) -> None:
    name = "01_01_01_01"
    for directory in ("mmwave", "imu", "pseudo_labels"):
        (tmp_path / directory).mkdir()
    split = tmp_path / "split.txt"
    write_split(split, name)
    with h5py.File(tmp_path / "mmwave" / f"{name}.mat", "w") as handle:
        handle.create_dataset(
            "data", data=np.zeros((128, 256, 40), dtype=np.float32)
        )
    np.save(tmp_path / "imu" / f"{name}.npy", np.zeros((40, 2, 3)))
    np.save(
        tmp_path / "pseudo_labels" / f"{name}.npy",
        np.zeros((30, 21, 3)),
    )

    dataset = PoseDataset(DataConfig(root=str(tmp_path)), split)
    radar, imu, target = dataset[0]
    assert radar.shape == (30, 256, 128)
    assert imu.shape == (30, 6)
    assert target["kpt"].shape == (30, 1, 63)
    assert target["kpt_cls"].shape == (30, 1, 1)


def test_single_left_hand_uses_first_slot(tmp_path: Path) -> None:
    name = "02_01_01_01"
    (tmp_path / "pose_outputs").mkdir()
    split = tmp_path / "split.txt"
    write_split(split, name)
    pose = np.ones((30, 21, 3), dtype=np.float32)
    np.save(tmp_path / "pose_outputs" / f"{name}.npy", pose)

    dataset = GesturePoseDataset(DataConfig(root=str(tmp_path)), split)
    padded, label = dataset[0]
    assert padded.shape == (30, 42, 3)
    assert padded[:, :21].sum().item() == 30 * 21 * 3
    assert padded[:, 21:].sum().item() == 0
    assert label.item() == 0
