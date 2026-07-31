#!/usr/bin/env python3
"""Convert radar heatmaps and IMU CSV files to the training data layout."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-angle-dir", type=Path, required=True)
    parser.add_argument("--range-doppler-dir", type=Path, required=True)
    parser.add_argument("--imu-csv-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--range-angle-key", default="RangeAngle")
    parser.add_argument("--range-doppler-key", default="RangeDoppler")
    parser.add_argument(
        "--imu-columns",
        nargs=6,
        default=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    )
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--range-bins", type=int, default=256)
    parser.add_argument("--feature-bins", type=int, default=64)
    return parser.parse_args()


def load_mat_array(path: Path, key: str) -> np.ndarray:
    try:
        values = loadmat(path)[key]
    except (NotImplementedError, ValueError):
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"{path} does not contain {key!r}")
            values = np.asarray(handle[key])
    except KeyError as error:
        raise KeyError(f"{path} does not contain {key!r}") from error
    return np.asarray(values)


def normalize_heatmap(
    values: np.ndarray, expected_shape: tuple[int, int, int], path: Path
) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"{path}: expected a 3D heatmap, got {values.shape}")
    matches = [
        permutation
        for permutation in itertools.permutations(range(3))
        if tuple(values.shape[index] for index in permutation) == expected_shape
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: cannot uniquely map {values.shape} to {expected_shape}"
        )
    return np.transpose(values, matches[0]).astype(np.float32, copy=False)


def read_imu_csv(path: Path, columns: list[str], frames: int) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing IMU columns {', '.join(sorted(missing))}"
            )
        for row in reader:
            rows.append([float(row[column]) for column in columns])
    values = np.asarray(rows, dtype=np.float32)
    if values.shape[0] < 2:
        raise ValueError(f"{path}: at least two IMU samples are required")
    source_time = np.linspace(0.0, 1.0, values.shape[0])
    target_time = np.linspace(0.0, 1.0, frames)
    return np.stack(
        [
            np.interp(target_time, source_time, values[:, channel])
            for channel in range(values.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    radar_output = args.output_root / "mmwave"
    imu_output = args.output_root / "imu"
    radar_output.mkdir(parents=True, exist_ok=True)
    imu_output.mkdir(parents=True, exist_ok=True)
    expected_shape = (args.frames, args.range_bins, args.feature_bins)

    angle_files = {path.stem: path for path in args.range_angle_dir.glob("*.mat")}
    doppler_files = {
        path.stem: path for path in args.range_doppler_dir.glob("*.mat")
    }
    radar_names = sorted(set(angle_files) & set(doppler_files))
    if not radar_names:
        raise FileNotFoundError("no matching range-angle/range-Doppler files")
    for name in radar_names:
        angle = normalize_heatmap(
            load_mat_array(angle_files[name], args.range_angle_key),
            expected_shape,
            angle_files[name],
        )
        doppler = normalize_heatmap(
            load_mat_array(doppler_files[name], args.range_doppler_key),
            expected_shape,
            doppler_files[name],
        )
        combined = np.concatenate((angle, doppler), axis=2)
        with h5py.File(radar_output / f"{name}.mat", "w") as handle:
            handle.create_dataset("data", data=combined.transpose((2, 1, 0)))

    imu_files = sorted(args.imu_csv_dir.glob("*.csv"))
    if not imu_files:
        raise FileNotFoundError("no IMU CSV files found")
    for path in imu_files:
        imu = read_imu_csv(path, list(args.imu_columns), args.frames)
        np.save(imu_output / f"{path.stem}.npy", imu)

    print(f"prepared {len(radar_names)} radar clips and {len(imu_files)} IMU clips")


if __name__ == "__main__":
    main()
