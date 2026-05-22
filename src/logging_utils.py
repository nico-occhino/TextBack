"""Logging helpers for optimization and inference results."""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


def append_jsonl(path: str | Path, row: Dict) -> None:
    """Append one dictionary as a JSONL row.

    Args:
        path: Output JSONL file path.
        row: Serializable dictionary to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


def append_csv(path: str | Path, row: Dict, fieldnames: Iterable[str]) -> None:
    """Append one dictionary to a CSV file.

    Args:
        path: Output CSV file path.
        row: Dictionary containing scalar values.
        fieldnames: Fixed CSV column order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: str | Path, data: Dict) -> None:
    """Write a dictionary as pretty JSON.

    Args:
        path: Output JSON path.
        data: Serializable dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
