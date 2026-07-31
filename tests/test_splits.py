from pathlib import Path

from mmegohand.data import parse_split


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_split_protocol() -> None:
    train = parse_split(ROOT / "splits/train.txt")
    test = parse_split(ROOT / "splits/test.txt")
    held_out = {5, 10, 15, 20}
    assert len(train) == 4151
    assert len(test) == 1055
    assert all(record.repetition not in held_out for record in train)
    assert all(record.repetition in held_out for record in test)
    assert {record.name for record in train}.isdisjoint(
        record.name for record in test
    )
