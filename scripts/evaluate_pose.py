#!/usr/bin/env python3
"""Evaluate pose MPJPE against MediaPipe-derived pseudo labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from torch.utils.data import DataLoader

from _bootstrap import REPO_ROOT
from mmegohand.configuration import load_config
from mmegohand.data import PoseDataset, collate_pose_batch
from mmegohand.engine import run_pose_epoch
from mmegohand.losses import build_criterion
from mmegohand.model import build_model
from mmegohand.runtime import load_checkpoint, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--include-partial-batch", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    config = load_config(resolve(repo_root, args.config))
    data = replace(
        config.data,
        root=str(resolve(repo_root, args.data_root or config.data.root)),
    )
    device = resolve_device(args.device or config.training.device)
    seed_everything(config.training.seed)
    split = data.train_split if args.split == "train" else data.test_split
    dataset = PoseDataset(data, resolve(repo_root, split))
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=(
            False if args.include_partial_batch else config.training.drop_last_eval
        ),
        num_workers=config.training.num_workers,
        collate_fn=collate_pose_batch,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config.model).to(device)
    load_checkpoint(args.checkpoint, model, device)
    criterion = build_criterion(
        config.loss.matching_class_cost,
        config.loss.matching_keypoint_cost,
        config.loss.class_loss_weight,
        config.loss.keypoint_loss_weight,
        config.model.frames,
    ).to(device)
    metrics = run_pose_epoch(model, criterion, loader, device)
    metrics["split_items"] = len(dataset)
    metrics["evaluated_items"] = len(loader) * config.training.batch_size
    if not loader.drop_last and len(dataset) % config.training.batch_size:
        metrics["evaluated_items"] = len(dataset)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
