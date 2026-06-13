"""Small logging helpers for CSV and JSONL files."""

import csv
import json
from pathlib import Path
from typing import Iterable


def append_csv_row(path: str | Path, row: dict, fieldnames: Iterable[str]) -> None:
    """Append one CSV row, writing the header if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl_record(path: str | Path, record: dict) -> None:
    """Append one JSON object to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def write_json(path: str | Path, data: dict) -> None:
    """Write a dictionary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
