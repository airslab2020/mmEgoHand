#!/usr/bin/env python3
"""Train the mmEgoHand pose estimator."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _bootstrap import REPO_ROOT
from mmegohand.configuration import load_config
from mmegohand.data import PoseDataset, collate_pose_batch
from mmegohand.engine import run_pose_epoch
from mmegohand.losses import build_criterion
from mmegohand.model import build_model
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
    parser.add_argument(
        "--include-partial-eval-batch",
        action="store_true",
        help="evaluate every split item instead of matching the legacy loader",
    )
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
    training = replace(
        config.training,
        device=args.device or config.training.device,
        output_dir=str(
            resolve(repo_root, args.output_dir or config.training.output_dir)
        ),
        drop_last_eval=(
            False
            if args.include_partial_eval_batch
            else config.training.drop_last_eval
        ),
    )

    seed_everything(training.seed)
    device = resolve_device(training.device)
    train_dataset = PoseDataset(
        data, resolve(repo_root, data.train_split)
    )
    test_dataset = PoseDataset(data, resolve(repo_root, data.test_split))
    loader_options = {
        "batch_size": training.batch_size,
        "num_workers": training.num_workers,
        "collate_fn": collate_pose_batch,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=training.drop_last_train,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=training.drop_last_eval,
        **loader_options,
    )

    model = build_model(config.model).to(device)
    criterion = build_criterion(
        config.loss.matching_class_cost,
        config.loss.matching_keypoint_cost,
        config.loss.class_loss_weight,
        config.loss.keypoint_loss_weight,
        config.model.frames,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training.scheduler_step,
        gamma=training.scheduler_gamma,
    )

    output_dir = Path(training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_mpjpe = float("inf")
    if args.resume:
        payload = load_checkpoint(
            args.resume, model, device, optimizer, scheduler
        )
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_mpjpe = float(
            payload.get("metrics", {}).get("mpjpe_mm", best_mpjpe)
        )

    history = []
    for epoch in range(start_epoch, training.epochs):
        train_metrics = run_pose_epoch(
            model, criterion, train_loader, device, optimizer
        )
        test_metrics = run_pose_epoch(
            model, criterion, test_loader, device
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
        if test_metrics["mpjpe_mm"] < best_mpjpe:
            best_mpjpe = test_metrics["mpjpe_mm"]
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
