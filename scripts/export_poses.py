#!/usr/bin/env python3
"""Export matched pose predictions for downstream gesture recognition."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import REPO_ROOT
from mmegohand.configuration import load_config
from mmegohand.data import PoseDataset, collate_pose_batch
from mmegohand.losses import move_targets
from mmegohand.matcher import HungarianMatcher
from mmegohand.model import build_model
from mmegohand.runtime import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument(
        "--split", choices=("train", "test", "all"), default="all"
    )
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def export_split(
    model: torch.nn.Module,
    matcher: HungarianMatcher,
    dataset: PoseDataset,
    output_dir: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        collate_fn=collate_pose_batch,
        pin_memory=device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    record_offset = 0
    with torch.no_grad():
        for radar, imu, targets in loader:
            radar = radar.to(device, non_blocking=True)
            imu = imu.to(device, non_blocking=True)
            targets = move_targets(targets, device)
            outputs = model(radar, imu)
            frame_matches = [
                matcher(outputs, targets, frame)
                for frame in range(model.config.frames)
            ]
            for batch_index, target in enumerate(targets):
                hand_count = target["kpt"].shape[1]
                sequence = []
                for frame, matches in enumerate(frame_matches):
                    source, destination = matches[batch_index]
                    ordered_source = source[torch.argsort(destination)]
                    sequence.append(
                        outputs["pred_kpt"][
                            frame, batch_index, ordered_source[:hand_count]
                        ]
                    )
                pose = torch.stack(sequence).reshape(
                    model.config.frames, hand_count * 21, 3
                )
                name = dataset.records[record_offset + batch_index].name
                np.save(output_dir / f"{name}.npy", pose.cpu().numpy())
            record_offset += radar.shape[0]


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    config = load_config(resolve(repo_root, args.config))
    data = replace(
        config.data,
        root=str(resolve(repo_root, args.data_root or config.data.root)),
    )
    device = resolve_device(args.device or config.training.device)
    model = build_model(config.model).to(device)
    load_checkpoint(args.checkpoint, model, device)
    matcher = HungarianMatcher(
        config.loss.matching_class_cost,
        config.loss.matching_keypoint_cost,
    )
    output_dir = Path(data.root) / data.pose_output_dir
    split_names = (
        ("train", "test") if args.split == "all" else (args.split,)
    )
    for split_name in split_names:
        split_path = (
            data.train_split if split_name == "train" else data.test_split
        )
        dataset = PoseDataset(data, resolve(repo_root, split_path))
        export_split(
            model,
            matcher,
            dataset,
            output_dir,
            config.training.batch_size,
            config.training.num_workers,
            device,
        )


if __name__ == "__main__":
    main()
