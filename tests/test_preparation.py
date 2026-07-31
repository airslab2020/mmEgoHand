from pathlib import Path

import numpy as np

from scripts.prepare_sensor_data import normalize_heatmap, read_imu_csv


def test_heatmap_axis_normalization(tmp_path: Path) -> None:
    values = np.arange(24).reshape(3, 4, 2)
    normalized = normalize_heatmap(values, (2, 4, 3), tmp_path / "radar.mat")
    assert normalized.shape == (2, 4, 3)
    assert normalized[1, 3, 2] == values[2, 3, 1]


def test_imu_csv_interpolation(tmp_path: Path) -> None:
    path = tmp_path / "imu.csv"
    path.write_text(
        "ax,ay,az,gx,gy,gz\n"
        "0,1,2,3,4,5\n"
        "10,11,12,13,14,15\n",
        encoding="utf-8",
    )
    values = read_imu_csv(
        path, ["ax", "ay", "az", "gx", "gy", "gz"], frames=3
    )
    assert values.shape == (3, 6)
    np.testing.assert_allclose(values[1], np.arange(5, 11))
