"""Inspect TextBack result files from the command line.

This script is intentionally lightweight: it uses only the standard library
and it prints a compact oral-exam summary from saved result files.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect TextBack results.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load available result files and print a readable summary."""
    _configure_stdout()
    args = parse_args()
    results_dir = _load_results_dir(args.config)

    final_prompts = _read_json(results_dir / "final_prompts.json")
    initial_prompts = _read_json(results_dir / "initial_prompts.json")
    initial_prompt_metadata = _read_json(results_dir / "initial_prompt_metadata.json")
    best_prompt_metadata = _read_json(results_dir / "best_prompt_metadata.json")
    activation_rates = _read_json(results_dir / "activation_rates.json")
    inference_summary = _read_json(results_dir / "inference_summary.json")
    inference_results = _read_csv(results_dir / "inference_results.csv")
    real_subset_summary = _read_csv(results_dir / "real_subset_summary.csv")
    optimization_logs = _read_csv(results_dir / "optimization_logs.csv")
    descriptor_memory = _read_json(results_dir / "descriptor_memory.json")

    _print_final_prompts(final_prompts)
    _print_initial_prompts(initial_prompts)
    _print_initial_prompt_metadata(initial_prompt_metadata)
    _print_activation_rates(activation_rates)
    _print_inference_summary(inference_summary)
    _print_real_subset_summary(real_subset_summary)
    _print_baseline_comparison(real_subset_summary, inference_summary)
    _print_confusion_distribution(inference_results)
    _print_optimization_trajectory(optimization_logs)
    _print_guardrail_counts(optimization_logs)
    _print_best_prompt_metadata(best_prompt_metadata)
    _print_descriptor_memory(descriptor_memory)


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


def _configure_stdout() -> None:
    """Prefer UTF-8 output when the host Python exposes stream reconfiguration."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_results_dir(config_path: str) -> Path:
    """Read paths.results_dir from the project YAML without extra dependencies."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    section = None
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
                section = line[:-1].strip()
                continue
            if section == "paths":
                key, separator, value = line.strip().partition(":")
                if separator and key == "results_dir":
                    return _resolve_project_path(_strip_yaml_scalar(value))

    raise ValueError(f"Could not find paths.results_dir in {path}")


def _strip_yaml_scalar(value: str) -> str:
    """Strip whitespace and simple YAML string quotes."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _resolve_project_path(value: str) -> Path:
    """Resolve project-relative config paths."""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _print_final_prompts(final_prompts) -> None:
    """Print final prompts per class."""
    print("\nA. Final Prompts")
    if not final_prompts:
        print("  results/final_prompts.json not found")
        return

    for target_class, prompt in final_prompts.items():
        print(f"  {target_class}: {prompt}")


def _print_initial_prompts(initial_prompts) -> None:
    """Print cached LLM initial prompts per class."""
    print("\nInitial LLM Prompts")
    if not initial_prompts:
        print("  results/initial_prompts.json not found")
        return

    for target_class, prompt in initial_prompts.items():
        print(f"  {target_class}: {prompt}")


def _print_initial_prompt_metadata(metadata) -> None:
    """Print where each initial prompt came from."""
    print("\nInitial Prompt Sources")
    if not metadata:
        print("  results/initial_prompt_metadata.json not found")
        return

    for target_class, values in metadata.items():
        source = values.get("source")
        forbidden_terms = values.get("forbidden_terms", [])
        attempts = values.get("attempts")
        error = values.get("error", "")
        print(
            f"  {target_class}: source={source}, attempts={attempts}, "
            f"forbidden_terms={forbidden_terms}, error={error}"
        )


def _print_activation_rates(activation_rates) -> None:
    """Print top-1 activation rates."""
    print("\nB. AMR@1")
    if not activation_rates:
        print("  results/activation_rates.json not found")
        return

    for target_class, rate in activation_rates.items():
        print(f"  {target_class}: {_format_float(rate)}")


def _print_inference_summary(inference_summary) -> None:
    """Print primary and secondary activation maximization metrics."""
    print("\nC. Inference Summary")
    if not inference_summary:
        print("  results/inference_summary.json not found")
        return

    for target_class, metrics in inference_summary.items():
        top1 = _format_float(metrics.get("top1_activation_rate"))
        top5 = _format_float(metrics.get("top5_activation_rate"))
        print(f"  {target_class}: AMR@1={top1}, AMR@5={top5}")


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


def _print_baseline_comparison(rows: list[dict] | None, inference_summary) -> None:
    """Print generated-image activation next to real-subset accuracy."""
    print("\nE. Real Baseline vs Generated Activation")
    if not rows or not inference_summary:
        print("  Need both real_subset_summary.csv and inference_summary.json")
        return

    real_metrics = {row["target_class"]: row for row in rows}
    for target_class, generated_metrics in inference_summary.items():
        real_row = real_metrics.get(target_class, {})
        real_top1 = _to_float(real_row.get("top1_accuracy"))
        real_top5 = _to_float(real_row.get("top5_accuracy"))
        generated_top1 = _to_float(generated_metrics.get("top1_activation_rate"))
        generated_top5 = _to_float(generated_metrics.get("top5_activation_rate"))
        print(
            f"  {target_class}: real_top1={_format_float(real_top1)}, "
            f"generated_AMR@1={_format_float(generated_top1)}, "
            f"real_top5={_format_float(real_top5)}, "
            f"generated_AMR@5={_format_float(generated_top5)}"
        )


def _print_confusion_distribution(rows: list[dict] | None) -> None:
    """Print the most frequent non-target top-1 predictions."""
    print("\nF. Confusion Distribution")
    if not rows:
        print("  results/inference_results.csv not found")
        return

    counts_by_class = defaultdict(Counter)
    for row in rows:
        if _is_true(row.get("top1_correct")):
            continue
        target_class = row.get("target_class")
        top1_label = row.get("top1_label")
        if target_class and top1_label:
            counts_by_class[target_class][top1_label] += 1

    if not counts_by_class:
        print("  No non-target top-1 predictions found")
        return

    for target_class, counts in counts_by_class.items():
        print(f"  {target_class}:")
        for label, count in counts.most_common(5):
            print(f"    {label}: {count}")


def _print_optimization_trajectory(rows: list[dict] | None) -> None:
    """Print target confidence/rank over optimization iterations."""
    print("\nG. Optimization Trajectory")
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
    print("\nH. Guardrail Rejections")
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
    print("\nI. Best Prompt Selection")
    if not metadata:
        print("  results/best_prompt_metadata.json not found")
        return

    for target_class, values in metadata.items():
        confidence = _format_float(values.get("best_target_confidence"))
        print(
            f"  {target_class}: best_iteration={values.get('best_iteration')}, "
            f"confidence={confidence}, rank={values.get('best_target_rank')}"
        )


def _print_descriptor_memory(descriptor_memory) -> None:
    """Print positive descriptor memory when present."""
    print("\nJ. Positive Descriptor Memory")
    if not descriptor_memory:
        print("  results/descriptor_memory.json not found")
        return

    for target_class, descriptors in descriptor_memory.items():
        print(f"  {target_class}:")
        if not descriptors:
            print("    (none)")
            continue
        for descriptor in descriptors:
            print(f"    - {descriptor}")


def _is_true(value) -> bool:
    """Return whether a CSV boolean-like value is true."""
    return str(value).strip().lower() == "true"


def _to_float(value):
    """Convert numeric strings to floats when possible."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
