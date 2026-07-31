# Paper and Code Alignment

This note records the settings used by the paper-facing release and their code
locations.

| Paper item | Value | Source |
| --- | --- | --- |
| Frames per clip | 30 from 40 raw samples | `mmegohand/data.py` |
| Radar input | `30 x 256 x 128` | `ModelConfig` |
| IMU input | `30 x 6` | `PoseDataset._load_imu` |
| Radar CNN channels | `8,16,32,64,64` | `MMEgoHandPose.mmwave_backbone` |
| IMU LSTM | 2 layers, hidden 512 | `MMEgoHandPose.imu_backbone` |
| Radar / IMU encoders | 6 / 6 | `ModelConfig.encoder_layers` |
| Pose decoder | 6 blocks | `ModelConfig.pose_decoder_layers` |
| Context decoder | 30 blocks | `ModelConfig.context_decoder_layers` |
| Hidden size / heads | 512 / 8 | `ModelConfig` |
| Active FFN | 1024 | `ModelConfig.active_ffn_dim` |
| Learned position embedding | 15 row + 15 column features | `position_encoding.py` |
| Radar / IMU fusion | 0.9 / 0.1 | `ModelConfig` |
| Matching costs | class 5, keypoint L1 25 | `LossConfig` |
| Loss weights | class 1, keypoint MSE 5 | `LossConfig` |
| Pose training | AdamW, `1e-4`, batch 32, 200 epochs | `TrainingConfig` |
| Scheduler | StepLR 100, gamma 0.5 | `TrainingConfig` |
| Gesture training | ResNet-50, `1e-2`, batch 32, 500 epochs | `GestureConfig` |

The paper configuration also preserves `drop_last=True` in evaluation. The
legacy gesture script divided the correct predictions from complete batches by
the full split size. Evaluation output therefore includes both the legacy
`accuracy` and conventional `processed_accuracy` values.

## Checkpoint Compatibility

The paper model contains 293,110,429 registered parameters. This count includes
legacy tensors that are present in the original state dict but are not called
by the forward method:

- `linear1` and `linear2` in each Transformer block use a hidden size of 2048.
  The active `ffn` path uses a hidden size of 1024.
- `input_proj`, `feature_linear`, and `feature_linear_imu` are retained at the
  model level.

`keep_legacy_parameters: true` is therefore required when loading the original
checkpoint. Setting it to `false` creates a smaller state dict, but that model
is not the paper-profiled 293.11M-parameter model.

## Split Provenance

The manifests in `splits/` are merged from the latest fixed lists in
WhisperYi/mmVR. They contain 5,206 retained samples:

| Scene | Train | Test |
| --- | ---: | ---: |
| 01 | 1,505 | 379 |
| 02 | 1,520 | 382 |
| 03 | 1,126 | 294 |
| Total | 4,151 | 1,055 |

The training set excludes repetitions 5, 10, 15, and 20; the test set contains
only those repetitions. There is no sample-name overlap.

## Label Terminology

Camera/MediaPipe outputs are called **pseudo labels**, not ground truth. The
loader uses `pseudo_labels/` as the canonical directory while accepting the
legacy `kpt_gt/` and `keypoint_gt/` directory names.
