"""Training and evaluation loops shared by command-line entry points."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import torch
from torch import nn

from .losses import SetCriterion, move_targets


def run_pose_epoch(
    model: nn.Module,
    criterion: SetCriterion,
    loader: Iterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion.train(training)
    totals = defaultdict(float)
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for radar, imu, targets in loader:
            radar = radar.to(device, non_blocking=True)
            imu = imu.to(device, non_blocking=True)
            targets = move_targets(targets, device)
            outputs = model(radar, imu)
            losses = criterion(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
            batch_size = radar.shape[0]
            samples += batch_size
            for name, value in losses.items():
                totals[name] += float(value.detach()) * batch_size
    return {name: value / max(samples, 1) for name, value in totals.items()}


def run_gesture_epoch(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    num_classes: int = 8,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    samples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for pose, labels in loader:
            pose = pose.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(pose)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            batch_size = pose.shape[0]
            samples += batch_size
            loss_sum += float(loss.detach()) * batch_size
            correct += int((predictions == labels).sum())
            for expected, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[int(expected), int(predicted)] += 1
    return {
        "loss": loss_sum / max(samples, 1),
        "accuracy": correct / max(samples, 1),
        "correct": correct,
        "samples": samples,
        "confusion_matrix": confusion.tolist(),
    }
