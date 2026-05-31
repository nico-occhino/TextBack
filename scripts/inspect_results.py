"""Inspect TextBack result files from the command line.

This script is intentionally lightweight: it uses only the standard library
plus the project config loader, and it prints a compact oral-exam summary.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect TextBack results.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load available result files and print a readable summary."""
    args = parse_args()
    config = load_config(args.config)
    results_dir = Path(config["paths"]["results_dir"])

    final_prompts = _read_json(results_dir / "final_prompts.json")
    best_prompt_metadata = _read_json(results_dir / "best_prompt_metadata.json")
    activation_rates = _read_json(results_dir / "activation_rates.json")
    inference_summary = _read_json(results_dir / "inference_summary.json")
    real_subset_summary = _read_csv(results_dir / "real_subset_summary.csv")
    optimization_logs = _read_csv(results_dir / "optimization_logs.csv")

    _print_final_prompts(final_prompts)
    _print_activation_rates(activation_rates)
    _print_inference_summary(inference_summary)
    _print_real_subset_summary(real_subset_summary)
    _print_baseline_comparison(real_subset_summary, activation_rates)
    _print_optimization_trajectory(optimization_logs)
    _print_guardrail_counts(optimization_logs)
    _print_best_prompt_metadata(best_prompt_metadata)


def _read_json(path: Path):
    """Read JSON when present, otherwise return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict] | None:
    """Read a CSV file when present, otherwise return None."""
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _print_final_prompts(final_prompts) -> None:
    """Print final prompts per class."""
    print("\nA. Final Prompts")
    if not final_prompts:
        print("  results/final_prompts.json not found")
        return

    for target_class, prompt in final_prompts.items():
        print(f"  {target_class}: {prompt}")


def _print_activation_rates(activation_rates) -> None:
    """Print top-1 activation rates."""
    print("\nB. Activation Rates")
    if not activation_rates:
        print("  results/activation_rates.json not found")
        return

    for target_class, rate in activation_rates.items():
        print(f"  {target_class}: {_format_float(rate)}")


def _print_inference_summary(inference_summary) -> None:
    """Print inference diagnostic metrics."""
    print("\nC. Inference Summary")
    if not inference_summary:
        print("  results/inference_summary.json not found")
        return

    for target_class, metrics in inference_summary.items():
        top1 = _format_float(metrics.get("top1_activation_rate"))
        top5 = _format_float(metrics.get("top5_activation_rate"))
        confidence = _format_float(metrics.get("mean_target_confidence"))
        rank = _format_float(metrics.get("mean_target_rank"))
        print(
            f"  {target_class}: top1={top1}, top5={top5}, "
            f"mean_conf={confidence}, mean_rank={rank}"
        )


def _print_real_subset_summary(rows: list[dict] | None) -> None:
    """Print real ImageNet subset baseline metrics."""
    print("\nD. Real-Subset Baseline")
    if not rows:
        print("  results/real_subset_summary.csv not found")
        return

    for row in rows:
        print(
            f"  {row['target_class']}: n={row['n_images']}, "
            f"top1={_format_float(row['top1_accuracy'])}, "
            f"top5={_format_float(row['top5_accuracy'])}"
        )


def _print_baseline_comparison(rows: list[dict] | None, activation_rates) -> None:
    """Print generated-image activation next to real-subset accuracy."""
    print("\nE. Real Baseline vs Generated Activation")
    if not rows or not activation_rates:
        print("  Need both real_subset_summary.csv and activation_rates.json")
        return

    real_top1 = {row["target_class"]: row["top1_accuracy"] for row in rows}
    for target_class, generated_rate in activation_rates.items():
        real_rate = real_top1.get(target_class)
        print(
            f"  {target_class}: real_top1={_format_float(real_rate)}, "
            f"generated_top1={_format_float(generated_rate)}"
        )


def _print_optimization_trajectory(rows: list[dict] | None) -> None:
    """Print target confidence/rank over optimization iterations."""
    print("\nF. Optimization Trajectory")
    if not rows:
        print("  results/optimization_logs.csv not found")
        return

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["target_class"]].append(row)

    for target_class, class_rows in by_class.items():
        class_rows.sort(key=lambda row: int(row["iteration"]))
        pieces = []
        for row in class_rows:
            pieces.append(
                f"step {row['iteration']}: conf={_format_float(row['target_confidence'])}, "
                f"rank={row['target_rank']}"
            )
        print(f"  {target_class}: " + " | ".join(pieces))


def _print_guardrail_counts(rows: list[dict] | None) -> None:
    """Print how often TextGrad updates were rejected by lexical guardrails."""
    print("\nG. Guardrail Rejections")
    if not rows:
        print("  results/optimization_logs.csv not found")
        return

    counts = defaultdict(lambda: {"total": 0, "rejected": 0})
    for row in rows:
        target_class = row["target_class"]
        counts[target_class]["total"] += 1
        if str(row.get("was_rejected", "")).lower() == "true":
            counts[target_class]["rejected"] += 1

    for target_class, values in counts.items():
        print(f"  {target_class}: {values['rejected']}/{values['total']} rejected")


def _print_best_prompt_metadata(metadata) -> None:
    """Print which optimization step produced the saved final prompt."""
    print("\nH. Best Prompt Selection")
    if not metadata:
        print("  results/best_prompt_metadata.json not found")
        return

    for target_class, values in metadata.items():
        confidence = _format_float(values.get("best_target_confidence"))
        print(
            f"  {target_class}: best_iteration={values.get('best_iteration')}, "
            f"confidence={confidence}, rank={values.get('best_target_rank')}"
        )


def _format_float(value) -> str:
    """Format numeric strings and floats consistently for console output."""
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
