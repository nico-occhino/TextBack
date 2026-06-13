"""Evaluate the RobustBench classifier on a manually prepared ImageNet subset.

Expected folder layout:

data/imagenet_subset/
  school bus/
    img001.JPEG
  golden retriever/
    img001.JPEG

Folder names are used as target class labels, so they should match torchvision
ImageNet labels exactly when possible.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import build_classifier
from src.config import create_output_dirs, load_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

PREDICTION_COLUMNS = [
    "target_class",
    "image_path",
    "top1_label",
    "top1_confidence",
    "target_confidence",
    "target_rank",
    "top1_correct",
    "top5_correct",
]

SUMMARY_COLUMNS = [
    "target_class",
    "n_images",
    "top1_accuracy",
    "top5_accuracy",
    "mean_target_confidence",
    "median_target_confidence",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a real ImageNet subset.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load config, classify subset images, and save CSV results."""
    args = parse_args()
    config = load_config(args.config)
    create_output_dirs(config)

    subset_root = Path(config["paths"]["imagenet_subset_root"])
    if not subset_root.exists():
        raise FileNotFoundError(
            f"ImageNet subset folder not found: {subset_root}. "
            "Create class folders under data/imagenet_subset first."
        )

    classifier = build_classifier(config)
    results_dir = Path(config["paths"]["results_dir"])
    predictions_path = results_dir / "real_subset_predictions.csv"
    summary_path = results_dir / "real_subset_summary.csv"

    _write_header(predictions_path, PREDICTION_COLUMNS)
    _write_header(summary_path, SUMMARY_COLUMNS)

    for class_dir in sorted(path for path in subset_root.iterdir() if path.is_dir()):
        target_class = class_dir.name
        image_paths = _find_images(class_dir)
        print(f"Evaluating {target_class}: {len(image_paths)} images")

        class_rows = []
        for image_path in image_paths:
            result = classifier.predict(
                image_path=image_path,
                target_class=target_class,
                top_k=int(config["experiment"].get("top_k", 5)),
            )

            row = _prediction_row(target_class, image_path, result)
            class_rows.append(row)
            _append_row(predictions_path, row, PREDICTION_COLUMNS)

        summary_row = _summary_row(target_class, class_rows)
        _append_row(summary_path, summary_row, SUMMARY_COLUMNS)

    print(f"Saved per-image predictions to {predictions_path}")
    print(f"Saved class summary to {summary_path}")


def _find_images(class_dir: Path) -> list[Path]:
    """Find image files inside one class folder."""
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _prediction_row(target_class: str, image_path: Path, result: dict) -> dict:
    """Convert one classifier result into a CSV row."""
    target_rank = result["target_rank"]
    return {
        "target_class": target_class,
        "image_path": str(image_path),
        "top1_label": result["top1_label"],
        "top1_confidence": result["top1_confidence"],
        "target_confidence": result["target_confidence"],
        "target_rank": target_rank,
        "top1_correct": target_rank == 1,
        "top5_correct": target_rank is not None and target_rank <= 5,
    }


def _summary_row(target_class: str, rows: list[dict]) -> dict:
    """Compute class-level metrics from per-image rows."""
    n_images = len(rows)
    if n_images == 0:
        return {
            "target_class": target_class,
            "n_images": 0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "mean_target_confidence": 0.0,
            "median_target_confidence": 0.0,
        }

    target_confidences = [float(row["target_confidence"]) for row in rows]
    return {
        "target_class": target_class,
        "n_images": n_images,
        "top1_accuracy": _mean_bool(row["top1_correct"] for row in rows),
        "top5_accuracy": _mean_bool(row["top5_correct"] for row in rows),
        "mean_target_confidence": statistics.mean(target_confidences),
        "median_target_confidence": statistics.median(target_confidences),
    }


def _mean_bool(values) -> float:
    """Compute the average of boolean values."""
    values = list(values)
    return sum(int(value) for value in values) / len(values) if values else 0.0


def _write_header(path: Path, fieldnames: list[str]) -> None:
    """Create a CSV file with only its header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def _append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row to an existing CSV file."""
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(row)


if __name__ == "__main__":
    main()
