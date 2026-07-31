#!/usr/bin/env python3
"""Generate MediaPipe world-landmark pseudo labels from synchronized videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--max-missing-ratio", type=float, default=0.4)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--swap-handedness",
        action="store_true",
        help="swap MediaPipe left/right labels for the camera convention",
    )
    return parser.parse_args()


def sample_metadata(path: Path) -> tuple[int, int]:
    fields = path.stem.split("_")
    if len(fields) != 4:
        raise ValueError(f"{path}: expected sample name SS_CC_GG_RR")
    gesture = int(fields[2])
    return gesture, 2 if gesture in (7, 8) else 1


def interpolate_track(track: np.ndarray) -> np.ndarray:
    frames = track.shape[0]
    output = track.copy()
    timeline = np.arange(frames)
    for joint in range(track.shape[1]):
        for coordinate in range(3):
            values = track[:, joint, coordinate]
            valid = np.flatnonzero(~np.isnan(values))
            if valid.size == 0:
                raise ValueError("a complete hand track is missing")
            output[:, joint, coordinate] = np.interp(
                timeline, valid, values[valid]
            )
    return output


def resample(values: np.ndarray, frames: int) -> np.ndarray:
    source_time = np.linspace(0.0, 1.0, values.shape[0])
    target_time = np.linspace(0.0, 1.0, frames)
    flattened = values.reshape(values.shape[0], -1)
    sampled = np.stack(
        [
            np.interp(target_time, source_time, flattened[:, column])
            for column in range(flattened.shape[1])
        ],
        axis=1,
    )
    return sampled.reshape(frames, *values.shape[1:]).astype(np.float32)


def process_video(
    path: Path,
    hands: object,
    output_dir: Path,
    output_frames: int,
    max_missing_ratio: float,
    swap_handedness: bool,
) -> dict:
    _, expected_hands = sample_metadata(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    left_frames = []
    right_frames = []
    detected_frames = 0
    total_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        total_frames += 1
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        candidates = {}
        if results.multi_hand_world_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_world_landmarks,
                results.multi_handedness,
            ):
                classification = handedness.classification[0]
                label = classification.label
                if swap_handedness:
                    label = "Left" if label == "Right" else "Right"
                coordinates = np.asarray(
                    [(point.x, point.y, point.z) for point in landmarks.landmark],
                    dtype=np.float32,
                )
                if label not in candidates or classification.score > candidates[label][0]:
                    candidates[label] = (classification.score, coordinates)

        missing = np.full((21, 3), np.nan, dtype=np.float32)
        left = candidates.get("Left", (0.0, missing))[1]
        right = candidates.get("Right", (0.0, missing))[1]
        if expected_hands == 1:
            available = [value for value in candidates.values()]
            selected = max(available, key=lambda item: item[0])[1] if available else missing
            left_frames.append(selected)
            if not np.isnan(selected).all():
                detected_frames += 1
        else:
            left_frames.append(left)
            right_frames.append(right)
            if not np.isnan(left).all() and not np.isnan(right).all():
                detected_frames += 1
    capture.release()

    if total_frames == 0:
        raise ValueError(f"{path}: video contains no readable frames")
    missing_ratio = 1.0 - detected_frames / total_frames
    report = {
        "sample": path.stem,
        "source_frames": total_frames,
        "detected_frames": detected_frames,
        "missing_ratio": missing_ratio,
        "status": "discarded" if missing_ratio > max_missing_ratio else "kept",
    }
    if report["status"] == "discarded":
        return report

    left_track = interpolate_track(np.stack(left_frames))
    if expected_hands == 1:
        labels = resample(left_track, output_frames)
    else:
        right_track = interpolate_track(np.stack(right_frames))
        labels = resample(
            np.concatenate((left_track, right_track), axis=1), output_frames
        )
    np.save(output_dir / f"{path.stem}.npy", labels)
    return report


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.max_missing_ratio < 1.0:
        raise ValueError("--max-missing-ratio must be in [0, 1)")
    output_dir = args.output_root / "pseudo_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        path
        for path in args.video_dir.iterdir()
        if path.suffix.lower() in {".avi", ".mov", ".mp4", ".mkv"}
    )
    if not videos:
        raise FileNotFoundError("no supported video files found")

    reports = []
    hands_module = mp.solutions.hands
    with hands_module.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        for path in videos:
            reports.append(
                process_video(
                    path,
                    hands,
                    output_dir,
                    args.frames,
                    args.max_missing_ratio,
                    args.swap_handedness,
                )
            )

    report_path = args.output_root / "pseudo_label_report.json"
    report_path.write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    kept = sum(report["status"] == "kept" for report in reports)
    print(f"kept {kept}/{len(reports)} videos; report: {report_path}")


if __name__ == "__main__":
    main()
