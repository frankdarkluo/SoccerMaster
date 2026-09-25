#!/usr/bin/env python3
"""Merge the four preannotation CSV files into data/codex_all.csv."""

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("id", "战术", "时间范围", "判断依据", "置信度")
SOURCES = (
    ("data/codex_preannotations.csv", 0),
    ("data/codex_preannotations_train_reverse.csv", 1),
    ("data/codex_preannotations_valid_forward.csv", 0),
    ("data/codex_preannotations_valid_reverse.csv", 1),
)
OUTPUT = ROOT / "data/codex_all.csv"
ID_PATTERN = re.compile(r"(train|valid)/AA-(\d+)")
TIME_PATTERN = re.compile(r"(\d{2}):(\d{2})–\d{2}:\d{2}")


def sort_key(row: dict[str, str]) -> tuple[int, int, int, str]:
    split, number = ID_PATTERN.fullmatch(row["id"]).groups()
    minutes, seconds = TIME_PATTERN.fullmatch(row["时间范围"]).groups()
    return (split != "train", int(number), int(minutes) * 60 + int(seconds), row["战术"])


def main() -> None:
    selected: dict[tuple[str, str, str], tuple[tuple[bool, int], dict[str, str]]] = {}

    for relative_path, reverse_priority in SOURCES:
        with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                key = tuple(row[field] for field in FIELDS[:3])
                rank = (bool(row["置信度"].strip()), reverse_priority)
                if key not in selected or rank > selected[key][0]:
                    selected[key] = (rank, row)

    rows = [row for _, row in selected.values()]
    assert len(rows) == 1395
    assert Counter(row["id"].split("/", 1)[0] for row in rows) == {"train": 499, "valid": 896}
    assert all(ID_PATTERN.fullmatch(row["id"]) for row in rows)
    assert all(TIME_PATTERN.fullmatch(row["时间范围"]) for row in rows)
    rows.sort(key=sort_key)

    with OUTPUT.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
