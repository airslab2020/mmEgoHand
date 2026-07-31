#!/usr/bin/env python3
"""Train the paper's ResNet-50 pose-sequence gesture classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _bootstrap import REPO_ROOT
from mmegohand.configuration import load_config
from mmegohand.data import GesturePoseDataset
from mmegohand.engine import run_gesture_epoch
from mmegohand.gesture import build_gesture_model
from mmegohand.runtime import (
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--include-partial-eval-batch", action="store_true")
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
    output_dir = resolve(
        repo_root, args.output_dir or config.gesture.output_dir
    )
    seed_everything(config.training.seed)

    train_dataset = GesturePoseDataset(
        data, resolve(repo_root, data.train_split)
    )
    test_dataset = GesturePoseDataset(data, resolve(repo_root, data.test_split))
    loader_options = {
        "batch_size": config.gesture.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=config.gesture.drop_last_train,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=(
            False
            if args.include_partial_eval_batch
            else config.gesture.drop_last_eval
        ),
        **loader_options,
    )

    model = build_gesture_model(
        config.gesture.architecture, config.gesture.num_classes
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.gesture.learning_rate,
        weight_decay=config.gesture.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.gesture.scheduler_step,
        gamma=config.gesture.scheduler_gamma,
    )
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        payload = load_checkpoint(
            args.resume, model, device, optimizer, scheduler
        )
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_accuracy = float(
            payload.get("metrics", {}).get("accuracy", best_accuracy)
        )

    history = []
    for epoch in range(start_epoch, config.gesture.epochs):
        train_metrics = run_gesture_epoch(
            model,
            train_loader,
            device,
            optimizer,
            config.gesture.num_classes,
        )
        test_metrics = run_gesture_epoch(
            model,
            test_loader,
            device,
            num_classes=config.gesture.num_classes,
        )
        if config.gesture.legacy_split_denominator:
            train_metrics["processed_accuracy"] = train_metrics["accuracy"]
            test_metrics["processed_accuracy"] = test_metrics["accuracy"]
            train_metrics["accuracy"] = (
                train_metrics["correct"] / len(train_dataset)
            )
            test_metrics["accuracy"] = (
                test_metrics["correct"] / len(test_dataset)
            )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            test_metrics,
        )
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(test_metrics["accuracy"])
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                test_metrics,
            )
        write_json(output_dir / "history.json", {"epochs": history})


if __name__ == "__main__":
    main()
