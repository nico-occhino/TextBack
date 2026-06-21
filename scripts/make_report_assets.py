from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report tables and figures from saved TextBack results.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output-dir", default="results/report_assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results_dir = project_path(config["paths"]["results_dir"])
    xai_dir = results_dir / "xai"
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_results = read_required_csv(results_dir / "inference_results.csv")
    inference_results["target_rank"] = pd.to_numeric(inference_results["target_rank"])
    inference_results["target_confidence"] = pd.to_numeric(inference_results["target_confidence"])

    create_main_results_table(config, results_dir, inference_results, output_dir)
    create_optimization_trajectory_plot(config, results_dir, output_dir)
    create_confusion_table(config, inference_results, output_dir)
    create_occlusion_summary_assets(xai_dir, output_dir)
    create_selected_xai_figures(xai_dir, output_dir)

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


def create_occlusion_summary_assets(xai_dir: Path, output_dir: Path) -> None:
    summary_path = xai_dir / "occlusion_summary.csv"
    results_path = xai_dir / "occlusion_results.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary.to_csv(output_dir / "xai_occlusion_summary_table.csv", index=False)
        write_occlusion_summary_latex(summary, output_dir / "xai_occlusion_summary_table.tex")
    else:
        print(f"Occlusion summary not found: {summary_path}")

    if results_path.exists():
        occlusion_results = pd.read_csv(results_path)
        create_relative_occlusion_boxplot(occlusion_results, output_dir / "xai_relative_occlusion_boxplot.png")
    else:
        print(f"Occlusion results not found: {results_path}")


def create_relative_occlusion_boxplot(occlusion_results: pd.DataFrame, output_path: Path) -> None:
    if occlusion_results.empty:
        print("Occlusion results are empty.")
        return

    grouped = []
    labels = []
    for target_class, rows in occlusion_results.groupby("target_class", sort=False):
        values = pd.to_numeric(rows["relative_max_occlusion_drop"], errors="coerce").dropna()
        if values.empty:
            continue
        grouped.append(values.to_numpy())
        labels.append(target_class)

    if not grouped:
        print("No relative occlusion values available for boxplot.")
        return

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.boxplot(grouped, labels=labels, showfliers=False)
    axis.set_ylabel("Relative max occlusion drop")
    axis.set_title("Occlusion sensitivity by target class")
    axis.grid(True, axis="y", alpha=0.3)
    figure.autofmt_xdate(rotation=20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def create_selected_xai_figures(xai_dir: Path, output_dir: Path) -> None:
    selected_path = xai_dir / "xai_selected_examples.csv"
    if not selected_path.exists():
        print(f"Selected XAI examples not found: {selected_path}")
        print("Run: python scripts/run_xai.py --config configs/final.yaml --max-images-per-class 100")
        return

    selected = pd.read_csv(selected_path)
    if selected.empty:
        print(f"Selected XAI examples are empty: {selected_path}")
        return

    create_heatmap_figure(
        selected,
        heatmap_column="gradcam_npy_path",
        title="Selected Target Grad-CAM Examples",
        output_path=output_dir / "gradcam_selected_examples.png",
        cmap="magma",
    )
    create_heatmap_figure(
        selected,
        heatmap_column="occlusion_npy_path",
        title="Selected Occlusion Sensitivity Examples",
        output_path=output_dir / "occlusion_selected_examples.png",
        cmap="hot",
    )


def create_heatmap_figure(
    selected: pd.DataFrame,
    heatmap_column: str,
    title: str,
    output_path: Path,
    cmap: str,
) -> None:
    if heatmap_column not in selected.columns:
        print(f"Selected XAI metadata does not contain {heatmap_column}.")
        return

    n_rows = len(selected)
    figure, axes = plt.subplots(n_rows, 3, figsize=(12, 3.2 * n_rows), squeeze=False)
    for row_index, row in enumerate(selected.to_dict("records")):
        image = Image.open(project_path(row["image_path"])).convert("RGB")
        heatmap = np.load(project_path(row[heatmap_column]))
        normalized = normalize_heatmap(heatmap)
        visual_image = image.resize((normalized.shape[1], normalized.shape[0]))

        axes[row_index, 0].imshow(visual_image)
        axes[row_index, 0].set_title("Original", fontsize=9)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(visual_image)
        axes[row_index, 1].imshow(normalized, cmap=cmap, alpha=0.45)
        axes[row_index, 1].set_title(
            f"{row['target_class']} ({row['case_type']})\n"
            f"top-1: {row.get('top1_label', '')}, rank: {int(row['target_rank'])}",
            fontsize=9,
        )
        axes[row_index, 1].axis("off")

        heat = axes[row_index, 2].imshow(normalized, cmap=cmap)
        axes[row_index, 2].set_title("Heatmap", fontsize=9)
        axes[row_index, 2].axis("off")
        figure.colorbar(heat, ax=axes[row_index, 2], fraction=0.046, pad=0.04)

    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.maximum(heatmap, 0.0)
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


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


def write_occlusion_summary_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Class & N & Mean max drop & Mean relative max drop & Mean relative positive drop \\",
        r"\hline",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"{latex_escape(row['target_class'])} & "
            f"{int(row['n_images'])} & "
            f"{format_float(row['mean_max_occlusion_drop'])} & "
            f"{format_float(row['mean_relative_max_occlusion_drop'])} & "
            f"{format_float(row['mean_relative_positive_occlusion_drop'])} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Occlusion sensitivity summary over generated inference images.}",
            r"\label{tab:xai_occlusion_summary}",
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
