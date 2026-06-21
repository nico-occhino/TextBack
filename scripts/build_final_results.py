import argparse
import json
import sys
from collections import Counter, defaultdict
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


DEFAULT_OUTPUT_DIR = "results/final_report_assets"
DEFAULT_XAI_DIR = "results/xai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one clean TextBack results folder: tables, PNG figures, and a CLI summary. "
            "This script consolidates inspect_results.py, make_report_assets.py, and saved XAI artifacts."
        )
    )
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xai-dir", default=DEFAULT_XAI_DIR)
    parser.add_argument("--top-confusions", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", help="Do not print the textual CLI summary.")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    config = load_config(args.config)

    results_dir = project_path(config["paths"]["results_dir"])
    output_dir = project_path(args.output_dir)
    xai_dir = project_path(args.xai_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_result_bundle(config, results_dir, xai_dir)

    main_table = build_main_results_table(config, bundle)
    real_vs_generated = build_real_vs_generated_table(config, bundle)
    confusion_table = build_confusion_table(config, bundle, args.top_confusions)
    optimization_table = build_optimization_table(config, bundle)
    guardrail_table = build_guardrail_table(config, bundle)
    prompt_table = build_prompt_table(config, bundle)
    xai_table = build_xai_summary_table(config, bundle)

    write_tables(
        output_dir=output_dir,
        main_table=main_table,
        real_vs_generated=real_vs_generated,
        confusion_table=confusion_table,
        optimization_table=optimization_table,
        guardrail_table=guardrail_table,
        prompt_table=prompt_table,
        xai_table=xai_table,
    )

    create_optimization_trajectory_plot(config, bundle, output_dir)
    create_real_vs_generated_plot(config, real_vs_generated, output_dir)
    create_relative_occlusion_boxplot(config, bundle, output_dir)
    create_xai_selected_examples_figure(config, bundle, output_dir)

    report_text = build_text_summary(
        config=config,
        output_dir=output_dir,
        main_table=main_table,
        real_vs_generated=real_vs_generated,
        confusion_table=confusion_table,
        optimization_table=optimization_table,
        guardrail_table=guardrail_table,
        prompt_table=prompt_table,
        xai_table=xai_table,
        bundle=bundle,
    )
    (output_dir / "summary.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "summary.md").write_text(report_text, encoding="utf-8")

    if not args.quiet:
        print(report_text)


def load_result_bundle(config: dict, results_dir: Path, xai_dir: Path) -> dict:
    inference_results = read_required_csv(results_dir / "inference_results.csv")
    inference_results["target_rank"] = pd.to_numeric(inference_results["target_rank"], errors="coerce")
    inference_results["target_confidence"] = pd.to_numeric(inference_results["target_confidence"], errors="coerce")
    inference_results["top1_confidence"] = pd.to_numeric(inference_results["top1_confidence"], errors="coerce")

    optimization_logs = read_optional_csv(results_dir / "optimization_logs.csv")
    if optimization_logs is not None:
        optimization_logs["iteration"] = pd.to_numeric(optimization_logs["iteration"], errors="coerce")
        optimization_logs["target_confidence"] = pd.to_numeric(optimization_logs["target_confidence"], errors="coerce")
        optimization_logs["target_rank"] = pd.to_numeric(optimization_logs["target_rank"], errors="coerce")

    return {
        "results_dir": results_dir,
        "xai_dir": xai_dir,
        "final_prompts": read_json(results_dir / "final_prompts.json"),
        "initial_prompt_metadata": read_json(results_dir / "initial_prompt_metadata.json"),
        "best_prompt_metadata": read_json(results_dir / "best_prompt_metadata.json"),
        "activation_rates": read_json(results_dir / "activation_rates.json"),
        "inference_summary": read_json(results_dir / "inference_summary.json"),
        "real_subset_summary": read_optional_csv(results_dir / "real_subset_summary.csv"),
        "inference_results": inference_results,
        "optimization_logs": optimization_logs,
        "descriptor_memory": read_json(results_dir / "descriptor_memory.json"),
        "cue_synthesis": read_json(results_dir / "cue_synthesis.json"),
        "occlusion_summary": read_optional_csv(first_existing([
            xai_dir / "occlusion_summary.csv",
            results_dir / "occlusion_summary.csv",
        ])),
        "occlusion_results": read_optional_csv(first_existing([
            xai_dir / "occlusion_results.csv",
            results_dir / "occlusion_results.csv",
        ])),
        "xai_selected_examples": read_optional_csv(first_existing([
            xai_dir / "xai_selected_examples.csv",
            xai_dir / "selected_examples.csv",
            results_dir / "xai_selected_examples.csv",
        ])),
        "xai_metadata": read_optional_csv(first_existing([
            xai_dir / "xai_metadata.csv",
            results_dir / "xai_metadata.csv",
        ])),
    }


def build_main_results_table(config: dict, bundle: dict) -> pd.DataFrame:
    inference_results = bundle["inference_results"]
    inference_summary = bundle.get("inference_summary") or {}
    rows = []

    for target_class in target_classes(config):
        class_rows = inference_results[inference_results["target_class"] == target_class]
        summary = inference_summary.get(target_class, {})
        rows.append(
            {
                "Class": target_class,
                "AMR@1": metric_or_compute(summary, "top1_activation_rate", class_rows["target_rank"] == 1),
                "AMR@5": metric_or_compute(summary, "top5_activation_rate", class_rows["target_rank"] <= 5),
                "Mean confidence": metric_or_compute(summary, "mean_target_confidence", class_rows["target_confidence"]),
                "Median confidence": metric_or_compute(summary, "median_target_confidence", class_rows["target_confidence"].median()),
                "Mean rank": metric_or_compute(summary, "mean_target_rank", class_rows["target_rank"]),
            }
        )
    return pd.DataFrame(rows)


def build_real_vs_generated_table(config: dict, bundle: dict) -> pd.DataFrame:
    main_table = build_main_results_table(config, bundle)
    real_subset = bundle.get("real_subset_summary")
    rows = []
    for row in main_table.to_dict("records"):
        target_class = row["Class"]
        real_row = lookup_real_row(real_subset, target_class)
        real_top1 = to_float(real_row.get("top1_accuracy"))
        real_top5 = to_float(real_row.get("top5_accuracy"))
        rows.append(
            {
                "Class": target_class,
                "Generated AMR@1": row["AMR@1"],
                "Real Top-1": real_top1,
                "Delta Top-1": safe_delta(row["AMR@1"], real_top1),
                "Generated AMR@5": row["AMR@5"],
                "Real Top-5": real_top5,
                "Delta Top-5": safe_delta(row["AMR@5"], real_top5),
            }
        )
    return pd.DataFrame(rows)


def build_confusion_table(config: dict, bundle: dict, top_n: int) -> pd.DataFrame:
    inference_results = bundle["inference_results"]
    rows = []
    for target_class in target_classes(config):
        class_rows = inference_results[inference_results["target_class"] == target_class]
        wrong_rows = class_rows[class_rows["target_rank"] != 1]
        counts = wrong_rows["top1_label"].value_counts().head(top_n)
        rows.append(
            {
                "Class": target_class,
                "Most frequent wrong top-1 predictions": "; ".join(
                    f"{label} ({count})" for label, count in counts.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_optimization_table(config: dict, bundle: dict) -> pd.DataFrame:
    metadata = bundle.get("best_prompt_metadata") or {}
    logs = bundle.get("optimization_logs")
    rows = []
    for target_class in target_classes(config):
        item = metadata.get(target_class, {})
        if item:
            best_iteration = item.get("best_iteration")
            best_confidence = item.get("best_target_confidence")
            best_rank = item.get("best_target_rank")
        elif logs is not None:
            class_logs = logs[logs["target_class"] == target_class]
            best_row = class_logs.sort_values("target_confidence", ascending=False).head(1)
            if best_row.empty:
                best_iteration, best_confidence, best_rank = None, None, None
            else:
                best_iteration = int(best_row.iloc[0]["iteration"])
                best_confidence = float(best_row.iloc[0]["target_confidence"])
                best_rank = int(best_row.iloc[0]["target_rank"])
        else:
            best_iteration, best_confidence, best_rank = None, None, None
        rows.append(
            {
                "Class": target_class,
                "Best iteration": best_iteration,
                "Best confidence": to_float(best_confidence),
                "Best rank": best_rank,
            }
        )
    return pd.DataFrame(rows)


def build_guardrail_table(config: dict, bundle: dict) -> pd.DataFrame:
    logs = bundle.get("optimization_logs")
    rows = []
    for target_class in target_classes(config):
        if logs is None:
            total = None
            rejected = None
        else:
            class_logs = logs[logs["target_class"] == target_class]
            total = len(class_logs)
            rejected = int(class_logs["was_rejected"].astype(str).str.lower().eq("true").sum())
        rows.append(
            {
                "Class": target_class,
                "Rejected updates": rejected,
                "Total updates": total,
                "Rejection rate": (rejected / total) if total else None,
            }
        )
    return pd.DataFrame(rows)


def build_prompt_table(config: dict, bundle: dict) -> pd.DataFrame:
    final_prompts = bundle.get("final_prompts") or {}
    descriptor_memory = bundle.get("descriptor_memory") or {}
    rows = []
    for target_class in target_classes(config):
        descriptors = descriptor_memory.get(target_class, [])
        rows.append(
            {
                "Class": target_class,
                "Final prompt": final_prompts.get(target_class, ""),
                "Positive descriptors": "; ".join(descriptors[:12]),
            }
        )
    return pd.DataFrame(rows)


def build_xai_summary_table(config: dict, bundle: dict) -> pd.DataFrame:
    occlusion_summary = bundle.get("occlusion_summary")
    selected = bundle.get("xai_selected_examples")
    metadata = bundle.get("xai_metadata")
    rows = []

    for target_class in target_classes(config):
        occ = lookup_row(occlusion_summary, "target_class", target_class)
        selected_row = lookup_row(selected, "target_class", target_class)
        metadata_row = lookup_row(metadata, "target_class", target_class)
        selected_source = selected_row or metadata_row or {}
        rows.append(
            {
                "Class": target_class,
                "Case": selected_source.get("case_type"),
                "Selected top-1": selected_source.get("top1_label"),
                "Selected rank": to_float(selected_source.get("target_rank")),
                "Selected confidence": to_float(selected_source.get("target_confidence")),
                "Mean max occlusion drop": to_float(occ.get("mean_max_occlusion_drop")),
                "Median max occlusion drop": to_float(occ.get("median_max_occlusion_drop")),
                "Mean relative max drop": to_float(occ.get("mean_relative_max_occlusion_drop")),
            }
        )
    return pd.DataFrame(rows)


def write_tables(
    output_dir: Path,
    main_table: pd.DataFrame,
    real_vs_generated: pd.DataFrame,
    confusion_table: pd.DataFrame,
    optimization_table: pd.DataFrame,
    guardrail_table: pd.DataFrame,
    prompt_table: pd.DataFrame,
    xai_table: pd.DataFrame,
) -> None:
    tables = {
        "main_results_table": main_table,
        "real_vs_generated_table": real_vs_generated,
        "confusion_distribution_table": confusion_table,
        "optimization_best_steps_table": optimization_table,
        "guardrail_rejections_table": guardrail_table,
        "final_prompts_table": prompt_table,
        "xai_occlusion_summary_table": xai_table,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    write_latex_table(
        main_table[["Class", "AMR@1", "AMR@5", "Mean confidence", "Mean rank"]],
        output_dir / "main_results_table.tex",
        caption="Activation maximization performance over generated images.",
        label="tab:main-results",
        float_format="%.3f",
    )
    write_latex_table(
        real_vs_generated,
        output_dir / "real_vs_generated_table.tex",
        caption="Generated activation compared with the real ImageNet subset baseline.",
        label="tab:real-generated",
        float_format="%.3f",
    )
    write_latex_table(
        confusion_table,
        output_dir / "confusion_distribution_table.tex",
        caption="Most frequent wrong top-1 predictions on generated inference images.",
        label="tab:confusion-distribution",
        float_format="%.3f",
    )
    write_latex_table(
        xai_table,
        output_dir / "xai_occlusion_summary_table.tex",
        caption="Selected XAI examples and aggregate occlusion sensitivity.",
        label="tab:xai-occlusion-summary",
        float_format="%.3f",
    )


def create_optimization_trajectory_plot(config: dict, bundle: dict, output_dir: Path) -> None:
    logs = bundle.get("optimization_logs")
    if logs is None or logs.empty:
        return

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for target_class in target_classes(config):
        class_logs = logs[logs["target_class"] == target_class].sort_values("iteration")
        if class_logs.empty:
            continue
        axis.plot(class_logs["iteration"], class_logs["target_confidence"], marker="o", label=target_class)

    axis.set_xlabel("TextGrad optimization step")
    axis.set_ylabel("Target confidence")
    axis.set_title("Optimization trajectory")
    axis.set_xticks(sorted(logs["iteration"].dropna().unique()))
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "optimization_trajectory.png", dpi=220)
    plt.close(figure)


def create_real_vs_generated_plot(config: dict, table: pd.DataFrame, output_dir: Path) -> None:
    if table.empty or table["Real Top-1"].isna().all():
        return

    labels = table["Class"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.bar(x - width / 2, table["Generated AMR@1"], width, label="Generated AMR@1")
    axis.bar(x + width / 2, table["Real Top-1"], width, label="Real Top-1")
    axis.set_ylabel("Rate")
    axis.set_title("Generated activation vs real-subset baseline")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylim(0, 1.0)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "generated_vs_real_top1.png", dpi=220)
    plt.close(figure)


def create_relative_occlusion_boxplot(config: dict, bundle: dict, output_dir: Path) -> None:
    occlusion_results = bundle.get("occlusion_results")
    if occlusion_results is None or occlusion_results.empty:
        return
    if "relative_max_occlusion_drop" not in occlusion_results.columns:
        return

    data = []
    labels = []
    for target_class in target_classes(config):
        values = occlusion_results[occlusion_results["target_class"] == target_class]["relative_max_occlusion_drop"]
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            continue
        data.append(values.to_numpy())
        labels.append(target_class)

    if not data:
        return

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.boxplot(data, labels=labels)
    axis.set_ylabel("Relative max occlusion drop")
    axis.set_title("Occlusion sensitivity by target class")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "xai_relative_occlusion_boxplot.png", dpi=220)
    plt.close(figure)


def create_xai_selected_examples_figure(config: dict, bundle: dict, output_dir: Path) -> None:
    selected = bundle.get("xai_selected_examples")
    metadata = bundle.get("xai_metadata")
    source = selected if selected is not None and not selected.empty else metadata
    if source is None or source.empty:
        return

    rows = []
    for target_class in target_classes(config):
        row = lookup_row(source, "target_class", target_class)
        if row:
            rows.append(row)

    if not rows:
        return

    n_rows = len(rows)
    figure, axes = plt.subplots(n_rows, 3, figsize=(9, 2.7 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row_axes, row in zip(axes, rows):
        target_class = row.get("target_class", "")
        case_type = row.get("case_type", "")
        title = (
            f"{target_class} ({case_type})\n"
            f"top-1: {row.get('top1_label', '')}, "
            f"rank: {format_rank(row.get('target_rank'))}, "
            f"conf: {format_float(row.get('target_confidence'), digits=3)}"
        )

        image_path = resolve_project_file(row.get("image_path"), bundle)
        image = load_image_or_none(image_path)
        if image is None:
            for axis in row_axes:
                axis.axis("off")
            row_axes[0].set_title(f"{target_class}: image missing")
            continue

        row_axes[0].imshow(image)
        row_axes[0].set_title("Original", fontsize=9)
        row_axes[0].axis("off")

        gradcam_array = load_npy_from_row(row, "gradcam_npy_path", bundle)
        occlusion_array = load_npy_from_row(row, "occlusion_npy_path", bundle)

        if gradcam_array is not None:
            row_axes[1].imshow(image)
            row_axes[1].imshow(resize_heatmap(gradcam_array, image.size), alpha=0.45)
        else:
            gradcam_png = resolve_project_file(row.get("gradcam_path"), bundle)
            gradcam_image = load_image_or_none(gradcam_png)
            row_axes[1].imshow(gradcam_image if gradcam_image is not None else image)
        row_axes[1].set_title("Target Grad-CAM", fontsize=9)
        row_axes[1].axis("off")

        if occlusion_array is not None:
            row_axes[2].imshow(image)
            row_axes[2].imshow(resize_heatmap(occlusion_array, image.size), alpha=0.45)
        else:
            occ_png = resolve_project_file(row.get("occlusion_path"), bundle)
            occ_image = load_image_or_none(occ_png)
            row_axes[2].imshow(occ_image if occ_image is not None else image)
        row_axes[2].set_title("Occlusion drop", fontsize=9)
        row_axes[2].axis("off")

        row_axes[1].text(
            0.5,
            -0.18,
            title,
            transform=row_axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(output_dir / "xai_selected_examples.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_text_summary(
    config: dict,
    output_dir: Path,
    main_table: pd.DataFrame,
    real_vs_generated: pd.DataFrame,
    confusion_table: pd.DataFrame,
    optimization_table: pd.DataFrame,
    guardrail_table: pd.DataFrame,
    prompt_table: pd.DataFrame,
    xai_table: pd.DataFrame,
    bundle: dict,
) -> str:
    lines = []
    lines.append("TextBack Final Results")
    lines.append("=" * 22)
    lines.append(f"Output folder: {output_dir}")
    lines.append("")

    lines.append("1. Activation maximization")
    for row in main_table.to_dict("records"):
        label = classify_case(row["AMR@1"], row["AMR@5"], row["Mean rank"])
        lines.append(
            f"  {row['Class']}: {label}; "
            f"AMR@1={format_float(row['AMR@1'])}, "
            f"AMR@5={format_float(row['AMR@5'])}, "
            f"mean_conf={format_float(row['Mean confidence'])}, "
            f"mean_rank={format_float(row['Mean rank'], digits=2)}"
        )
    lines.append("")

    lines.append("2. Real subset vs generated")
    for row in real_vs_generated.to_dict("records"):
        lines.append(
            f"  {row['Class']}: real_top1={format_float(row['Real Top-1'])}, "
            f"generated_AMR@1={format_float(row['Generated AMR@1'])}, "
            f"real_top5={format_float(row['Real Top-5'])}, "
            f"generated_AMR@5={format_float(row['Generated AMR@5'])}"
        )
    lines.append("")

    lines.append("3. Best prompt selection and guardrails")
    guardrail_lookup = {row["Class"]: row for row in guardrail_table.to_dict("records")}
    for row in optimization_table.to_dict("records"):
        guardrail = guardrail_lookup.get(row["Class"], {})
        lines.append(
            f"  {row['Class']}: best_step={format_rank(row['Best iteration'])}, "
            f"best_conf={format_float(row['Best confidence'])}, "
            f"best_rank={format_rank(row['Best rank'])}, "
            f"rejected={format_rank(guardrail.get('Rejected updates'))}/{format_rank(guardrail.get('Total updates'))}"
        )
    lines.append("")

    lines.append("4. Confusion patterns")
    for row in confusion_table.to_dict("records"):
        lines.append(f"  {row['Class']}: {row['Most frequent wrong top-1 predictions']}")
    lines.append("")

    lines.append("5. XAI summary")
    if xai_table.empty or xai_table["Selected top-1"].isna().all():
        lines.append("  XAI tables not found. Run the XAI computation before building final assets.")
    else:
        for row in xai_table.to_dict("records"):
            lines.append(
                f"  {row['Class']}: case={row.get('Case')}, "
                f"selected_top1={row.get('Selected top-1')}, "
                f"rank={format_rank(row.get('Selected rank'))}, "
                f"conf={format_float(row.get('Selected confidence'), digits=3)}, "
                f"mean_max_drop={format_float(row.get('Mean max occlusion drop'), digits=3)}, "
                f"mean_rel_drop={format_float(row.get('Mean relative max drop'), digits=3)}"
            )
    lines.append("")

    lines.append("6. Files written")
    for path in sorted(output_dir.glob("*")):
        lines.append(f"  {path.relative_to(PROJECT_ROOT)}")

    lines.append("")
    lines.append("Interpretation guardrail: these results support candidate classifier-salient cues, not definitive causal proof of spurious features.")
    return "\n".join(lines)


def write_latex_table(table: pd.DataFrame, path: Path, caption: str, label: str, float_format: str = "%.3f") -> None:
    latex = table.to_latex(index=False, escape=True, float_format=lambda x: float_format % x)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        latex.strip(),
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return paths[0] if paths else None


def target_classes(config: dict) -> list[str]:
    return list(config["experiment"]["target_classes"])


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path_string(value: str | None) -> str | None:
    if value is None or pd.isna(value) or value == "":
        return None
    return str(value).replace("\\", "/")


def resolve_project_file(value: str | None, bundle: dict) -> Path | None:
    value = normalize_path_string(value)
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_image_or_none(path: Path | None):
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("RGB")


def load_npy_from_row(row: dict, key: str, bundle: dict):
    path = resolve_project_file(row.get(key), bundle)
    if path is None or not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def resize_heatmap(array, image_size: tuple[int, int]):
    arr = np.asarray(array)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    arr = arr.astype(float)
    arr = arr - np.nanmin(arr)
    max_value = np.nanmax(arr)
    if max_value > 0:
        arr = arr / max_value
    heatmap_image = Image.fromarray(np.uint8(arr * 255), mode="L").resize(image_size, Image.BILINEAR)
    return np.asarray(heatmap_image) / 255.0


def lookup_real_row(real_subset: pd.DataFrame | None, target_class: str) -> dict:
    return lookup_row(real_subset, "target_class", target_class)


def lookup_row(table: pd.DataFrame | None, column: str, value: str) -> dict:
    if table is None or table.empty or column not in table.columns:
        return {}
    rows = table[table[column] == value]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def metric_or_compute(summary: dict, key: str, fallback):
    value = summary.get(key) if summary else None
    if value is not None:
        return to_float(value)
    if hasattr(fallback, "mean"):
        return float(fallback.mean())
    return to_float(fallback)


def safe_delta(a, b):
    a = to_float(a)
    b = to_float(b)
    if a is None or b is None:
        return None
    return a - b


def to_float(value):
    if value is None or value == "" or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value, digits: int = 4) -> str:
    value = to_float(value)
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_rank(value) -> str:
    value = to_float(value)
    if value is None:
        return "n/a"
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def classify_case(amr1, amr5, mean_rank) -> str:
    amr1 = to_float(amr1) or 0.0
    amr5 = to_float(amr5) or 0.0
    mean_rank = to_float(mean_rank) or 999.0
    if amr1 >= 0.30 and amr5 >= 0.60:
        return "success"
    if amr5 >= 0.30 or mean_rank <= 20:
        return "partial"
    return "failure"


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


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
