from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


REPORT_CASES = [
    ("cowboy hat", "success"),
    ("volcano", "success"),
    ("sports car", "partial"),
    ("tabby", "failure"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report tables and figures from saved TextBack results.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output-dir", default="results/report_assets")
    parser.add_argument("--gradcam-metadata", default="results/gradcam/gradcam_metadata.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results_dir = project_path(config["paths"]["results_dir"])
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_results = read_required_csv(results_dir / "inference_results.csv")
    inference_results["target_rank"] = pd.to_numeric(inference_results["target_rank"])
    inference_results["target_confidence"] = pd.to_numeric(inference_results["target_confidence"])

    create_main_results_table(config, results_dir, inference_results, output_dir)
    create_optimization_trajectory_plot(config, results_dir, output_dir)
    create_confusion_table(config, inference_results, output_dir)
    create_gradcam_case_figure(config, inference_results, project_path(args.gradcam_metadata), output_dir)

    print(f"Report assets written to {output_dir}")


def create_main_results_table(config: dict, results_dir: Path, inference_results: pd.DataFrame, output_dir: Path) -> None:
    real_subset = read_optional_csv(results_dir / "real_subset_summary.csv")
    rows = []

    for target_class in config["experiment"]["target_classes"]:
        class_rows = inference_results[inference_results["target_class"] == target_class]
        real_row = lookup_real_row(real_subset, target_class)
        rows.append(
            {
                "Class": target_class,
                "AMR@1": mean_indicator(class_rows["target_rank"] == 1),
                "AMR@5": mean_indicator(class_rows["target_rank"] <= 5),
                "Mean confidence": class_rows["target_confidence"].mean(),
                "Mean rank": class_rows["target_rank"].mean(),
                "Real Top-1": real_row.get("top1_accuracy"),
                "Real Top-5": real_row.get("top5_accuracy"),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "main_results_table.csv", index=False)
    write_main_results_latex(table, output_dir / "main_results_table.tex")


def create_optimization_trajectory_plot(config: dict, results_dir: Path, output_dir: Path) -> None:
    optimization_path = results_dir / "optimization_logs.csv"
    if not optimization_path.exists():
        print(f"Optimization log not found: {optimization_path}")
        return

    logs = pd.read_csv(optimization_path)
    logs["iteration"] = pd.to_numeric(logs["iteration"])
    logs["target_confidence"] = pd.to_numeric(logs["target_confidence"])

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for target_class in config["experiment"]["target_classes"]:
        class_logs = logs[logs["target_class"] == target_class].sort_values("iteration")
        if class_logs.empty:
            continue
        axis.plot(
            class_logs["iteration"],
            class_logs["target_confidence"],
            marker="o",
            label=target_class,
        )

    axis.set_xlabel("TextGrad optimization step")
    axis.set_ylabel("Target confidence")
    axis.set_title("Optimization trajectory")
    axis.set_xticks(sorted(logs["iteration"].unique()))
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "optimization_trajectory.png", dpi=200)
    figure.savefig(output_dir / "optimization_trajectory.pdf")
    plt.close(figure)


def create_confusion_table(config: dict, inference_results: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for target_class in config["experiment"]["target_classes"]:
        class_rows = inference_results[inference_results["target_class"] == target_class]
        wrong_rows = class_rows[class_rows["top1_label"] != target_class]
        counts = wrong_rows["top1_label"].value_counts().head(5)
        confusions = "; ".join(f"{label} ({count})" for label, count in counts.items())
        rows.append({"Class": target_class, "Most frequent top-1 confusions": confusions})

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "confusion_distribution_table.csv", index=False)
    write_confusion_latex(table, output_dir / "confusion_distribution_table.tex")


def create_gradcam_case_figure(
    config: dict,
    inference_results: pd.DataFrame,
    metadata_path: Path,
    output_dir: Path,
) -> None:
    if not metadata_path.exists():
        print(f"Grad-CAM metadata not found: {metadata_path}")
        print("Run: python scripts/run_gradcam.py --config configs/final.yaml --max-generated-per-class 20 --max-real-per-class 1")
        return

    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        print(f"Grad-CAM metadata is empty: {metadata_path}")
        return

    metadata["target_rank"] = pd.to_numeric(metadata["target_rank"], errors="coerce")
    metadata["target_confidence"] = pd.to_numeric(metadata["target_confidence"], errors="coerce")
    selected_rows = []

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (target_class, case_type) in zip(axes.ravel(), REPORT_CASES):
        row = select_gradcam_row(metadata, target_class, case_type)
        if row is None:
            axis.axis("off")
            axis.set_title(f"{target_class}: missing Grad-CAM")
            continue

        output_path = project_path(row["output_path"])
        if not output_path.exists():
            axis.axis("off")
            axis.set_title(f"{target_class}: file not found")
            print(f"Missing Grad-CAM image: {output_path}")
            continue

        image = Image.open(output_path).convert("RGB")
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{target_class} ({case_type})\n"
            f"top-1: {row.get('top1_label', '')}, rank: {int(row['target_rank'])}, "
            f"conf: {float(row['target_confidence']):.3f}",
            fontsize=9,
        )
        selected_rows.append(row.to_dict())

    figure.tight_layout()
    figure.savefig(output_dir / "gradcam_four_examples.png", dpi=200)
    figure.savefig(output_dir / "gradcam_four_examples.pdf")
    plt.close(figure)

    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(output_dir / "gradcam_four_examples_selection.csv", index=False)

    if not has_failure_gradcam(metadata, "tabby"):
        print("No generated tabby Grad-CAM with target_rank > 5 was found in metadata.")
        print("For a clearer failure figure, rerun Grad-CAM with --max-generated-per-class 20.")


def select_gradcam_row(metadata: pd.DataFrame, target_class: str, case_type: str):
    rows = metadata[
        (metadata["source_type"] == "generated")
        & (metadata["target_class"] == target_class)
    ].copy()
    if rows.empty:
        return None

    if case_type == "success":
        preferred = rows[rows["target_rank"] == 1]
        if preferred.empty:
            preferred = rows[rows["target_rank"] <= 5]
        if preferred.empty:
            preferred = rows
        return preferred.sort_values(["target_rank", "target_confidence"], ascending=[True, False]).iloc[0]

    if case_type == "partial":
        preferred = rows[(rows["target_rank"] > 1) & (rows["target_rank"] <= 5)]
        if preferred.empty:
            preferred = rows[rows["target_rank"] <= 5]
        if preferred.empty:
            preferred = rows
        return preferred.sort_values(["target_rank", "target_confidence"], ascending=[True, False]).iloc[0]

    preferred = rows[rows["target_rank"] > 5]
    if preferred.empty:
        preferred = rows
    return preferred.sort_values(["target_rank", "target_confidence"], ascending=[False, False]).iloc[0]


def has_failure_gradcam(metadata: pd.DataFrame, target_class: str) -> bool:
    rows = metadata[
        (metadata["source_type"] == "generated")
        & (metadata["target_class"] == target_class)
    ]
    return bool((rows["target_rank"] > 5).any())


def write_main_results_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Class & AMR@1 & AMR@5 & Real Top-1 & Real Top-5 \\",
        r"\hline",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"{latex_escape(row['Class'])} & "
            f"{format_float(row['AMR@1'])} & "
            f"{format_float(row['AMR@5'])} & "
            f"{format_float(row['Real Top-1'])} & "
            f"{format_float(row['Real Top-5'])} \\\\" 
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Generated activation metrics compared with the real ImageNet subset baseline.}",
            r"\label{tab:main_results}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_confusion_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lp{0.70\linewidth}}",
        r"\hline",
        r"Class & Most frequent top-1 confusions \\",
        r"\hline",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"{latex_escape(row['Class'])} & "
            f"{latex_escape(row['Most frequent top-1 confusions'])} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Most frequent wrong top-1 predictions on generated inference images.}",
            r"\label{tab:confusion_distribution}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def lookup_real_row(real_subset: pd.DataFrame | None, target_class: str) -> dict:
    if real_subset is None or real_subset.empty:
        return {}
    rows = real_subset[real_subset["target_class"] == target_class]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Optional result file not found: {path}")
        return None
    return pd.read_csv(path)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def mean_indicator(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def format_float(value) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.2f}"


def latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
