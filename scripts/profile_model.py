#!/usr/bin/env python3
"""Profile parameter count, latency, throughput, and optional FLOPs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import torch

from _bootstrap import REPO_ROOT
from mmegohand.configuration import load_config
from mmegohand.model import build_model
from mmegohand.runtime import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-flops", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = load_config(config_path)
    device = resolve_device(args.device or config.training.device)
    model = build_model(config.model).to(device).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device)

    radar = torch.randn(
        1,
        config.model.frames,
        config.model.radar_height,
        config.model.radar_width,
        device=device,
    )
    imu = torch.randn(1, config.model.frames, 6, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(radar, imu)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(radar, imu)
        synchronize(device)
    latency_seconds = (time.perf_counter() - start) / args.iterations

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    report = {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "fp32_weight_mib": parameters * 4 / (1024**2),
        "latency_ms_per_sample": latency_seconds * 1000,
        "realtime_speedup": 2.0 / latency_seconds,
        "data_fps": config.model.frames / latency_seconds,
        "process_rss_mib": psutil.Process().memory_info().rss / (1024**2),
        "device": str(device),
    }
    if device.type == "cuda":
        report["peak_gpu_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
        report["peak_gpu_reserved_mib"] = (
            torch.cuda.max_memory_reserved(device) / (1024**2)
        )

    if args.profile_flops:
        activities = [
            torch.profiler.ProfilerActivity.CUDA
            if device.type == "cuda"
            else torch.profiler.ProfilerActivity.CPU
        ]
        with torch.no_grad(), torch.profiler.profile(
            activities=activities, with_flops=True
        ) as profiler:
            model(radar, imu)
            synchronize(device)
        flops = sum(event.flops for event in profiler.key_averages())
        report["profiled_gflops"] = flops / 1.0e9
        report["profiled_gmacs"] = flops / 2.0e9

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
