# Data Format

## Sample identifier

For the three paper scenes, a sample identifier is:

```text
SS_CC_GG_RR
```

- `SS`: one of 12 subject-hand settings
- `CC`: scene identifier (`01`, `02`, or `03`)
- `GG`: gesture identifier (`01` through `08`)
- `RR`: repetition identifier (`01` through `20`)

## Radar

`mmwave/<sample>.mat` is an HDF5 file containing `data`. After the legacy
transpose, its shape is `40 x 256 x 128`. The loader uniformly selects 30
frames.

## IMU

`imu/<sample>.npy` contains 40 synchronized six-channel samples. Arrays stored
as `40 x 2 x 3` or `40 x 6` are both reshaped to `30 x 6` after frame
selection.

## Pseudo labels

`pseudo_labels/<sample>.npy` accepts:

- `T x 21 x 3` for one hand
- `T x 42 x 3` for two hands
- `T x H x 21 x 3`
- `T x H x 63`

The coordinate values are MediaPipe world landmarks expressed in meters and are
used directly as the camera-view target representation. No camera intrinsic
calibration, camera-to-radar extrinsic calibration, scale fitting, or rigid
coordinate transformation is applied. The model learns a direct supervised
mapping from synchronized radar/IMU inputs to this pseudo-label convention; its
outputs are not radar-centered geometric coordinates. The loader resamples and
reshapes the training target to `30 x H x 63`.

## Pose outputs

`scripts/export_poses.py` writes `30 x 21 x 3` for one hand or
`30 x 42 x 3` for two hands. `GesturePoseDataset` inserts a zero-valued absent
hand slot for one-hand samples, yielding `30 x 42 x 3` for ResNet-50.

## Preparation boundary

`scripts/prepare_sensor_data.py` consumes range-angle and range-Doppler
heatmaps already produced by the radar DSP pipeline. It normalizes each map to
`40 x 256 x 64`, concatenates them to `40 x 256 x 128`, and writes the HDF5
layout expected by `PoseDataset`. It also interpolates named accelerometer and
gyroscope CSV columns to 40 synchronized samples.

`scripts/generate_pseudo_labels.py` uses MediaPipe world landmarks. It applies
the paper's 40% missing-frame rule, fills retained temporal gaps by linear
interpolation, and writes 30-frame pseudo-label arrays.
