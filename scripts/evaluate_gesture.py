#!/usr/bin/env python3
"""Evaluate the downstream gesture classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from torch.utils.data import DataLoader

from _bootstrap import REPO_ROOT
from mmegohand.configuration import load_config
from mmegohand.data import GesturePoseDataset
from mmegohand.engine import run_gesture_epoch
from mmegohand.gesture import build_gesture_model
from mmegohand.runtime import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
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
    dataset = GesturePoseDataset(data, resolve(repo_root, data.test_split))
    loader = DataLoader(
        dataset,
        batch_size=config.gesture.batch_size,
        shuffle=False,
        drop_last=(
            False
            if args.include_partial_batch
            else config.gesture.drop_last_eval
        ),
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_gesture_model(
        config.gesture.architecture, config.gesture.num_classes
    ).to(device)
    load_checkpoint(args.checkpoint, model, device)
    metrics = run_gesture_epoch(
        model, loader, device, num_classes=config.gesture.num_classes
    )
    metrics["split_items"] = len(dataset)
    if config.gesture.legacy_split_denominator:
        metrics["processed_accuracy"] = metrics["accuracy"]
        metrics["accuracy"] = metrics["correct"] / len(dataset)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
