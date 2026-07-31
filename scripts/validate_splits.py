#!/usr/bin/env python3
"""Validate the fixed-repetition train/test protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _bootstrap import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="splits/train.txt")
    parser.add_argument("--test", default="splits/test.txt")
    return parser.parse_args()


def read_split(path: Path) -> list[tuple[str, str, str, int, int]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.strip().split(",")
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_number}: expected five fields")
            name, hand_setting, scene, gesture, repetition = parts
            expected = "_".join(parts[1:])
            if name != expected:
                raise ValueError(
                    f"{path}:{line_number}: name {name} != metadata {expected}"
                )
            records.append(
                (name, hand_setting, scene, int(gesture), int(repetition))
            )
    return records


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    train_path = Path(args.train)
    test_path = Path(args.test)
    if not train_path.is_absolute():
        train_path = repo_root / train_path
    if not test_path.is_absolute():
        test_path = repo_root / test_path
    train = read_split(train_path)
    test = read_split(test_path)
    held_out_repetitions = {5, 10, 15, 20}
    bad_train = [
        record[0] for record in train if record[4] in held_out_repetitions
    ]
    bad_test = [
        record[0] for record in test if record[4] not in held_out_repetitions
    ]
    overlap = sorted(
        {record[0] for record in train} & {record[0] for record in test}
    )
    if bad_train or bad_test or overlap:
        raise ValueError(
            "invalid split protocol: "
            f"bad_train={len(bad_train)}, bad_test={len(bad_test)}, "
            f"overlap={len(overlap)}"
        )

    def counts(records: list[tuple[str, str, str, int, int]]) -> dict:
        return {
            "items": len(records),
            "scenes": dict(sorted(Counter(row[2] for row in records).items())),
            "gestures": dict(
                sorted(Counter(str(row[3]) for row in records).items())
            ),
            "hand_settings": dict(
                sorted(Counter(row[1] for row in records).items())
            ),
        }

    report = {
        "protocol": {
            "train_repetitions": [
                repetition
                for repetition in range(1, 21)
                if repetition not in held_out_repetitions
            ],
            "test_repetitions": sorted(held_out_repetitions),
        },
        "train": counts(train),
        "test": counts(test),
        "total_items": len(train) + len(test),
        "overlap": 0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
