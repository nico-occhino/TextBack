import argparse
import json
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


DEFAULT_OUTPUT_DIR = "results/final_report_assets"
DEFAULT_XAI_DIR = "results/xai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final TextBack report tables, PNG figures, and CLI summary."
    )
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    config = load_config(args.config)

    results_dir = project_path(config["paths"]["results_dir"])
    output_dir = project_path(args.output_dir)
    xai_dir = project_path(DEFAULT_XAI_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_result_bundle(results_dir, xai_dir)

    main_table = build_main_results_table(config, bundle)
    real_vs_generated = build_real_vs_generated_table(config, bundle)
    confusion_table = build_confusion_table(config, bundle, top_n=5)
    optimization_table = build_optimization_table(config, bundle)
    guardrail_table = build_guardrail_table(config, bundle)
    prompt_table = build_prompt_table(config, bundle)
    gradcam_table = build_gradcam_selected_examples_table(config, bundle)

    write_tables(
        output_dir=output_dir,
        main_table=main_table,
        real_vs_generated=real_vs_generated,
        confusion_table=confusion_table,
        optimization_table=optimization_table,
        guardrail_table=guardrail_table,
        prompt_table=prompt_table,
        gradcam_table=gradcam_table,
    )

    create_optimization_trajectory_plot(config, bundle, output_dir)
    create_real_vs_generated_plot(real_vs_generated, output_dir)
    create_gradcam_presentation_figure(config, bundle, output_dir)

    report_text = build_text_summary(
        output_dir=output_dir,
        main_table=main_table,
        real_vs_generated=real_vs_generated,
        confusion_table=confusion_table,
        optimization_table=optimization_table,
        guardrail_table=guardrail_table,
        prompt_table=prompt_table,
        gradcam_table=gradcam_table,
    )
    (output_dir / "summary.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "summary.md").write_text(report_text, encoding="utf-8")

    print(report_text)


def load_result_bundle(results_dir: Path, xai_dir: Path) -> dict:
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
        "xai_selected_examples": read_optional_csv(xai_dir / "xai_selected_examples.csv"),
    }


def build_main_results_table(config: dict, bundle: dict) -> pd.DataFrame:
    inference_results = bundle["inference_results"]
    inference_summary = bundle.get("inference_summary") or {}
    activation_rates = bundle.get("activation_rates") or {}
    rows = []

    for target_class in target_classes(config):
        class_rows = inference_results[inference_results["target_class"] == target_class]
        summary = inference_summary.get(target_class, {})
        amr_at_1 = summary.get("top1_activation_rate", activation_rates.get(target_class))
        rows.append(
            {
                "Class": target_class,
                "AMR@1": (
                    to_float(amr_at_1)
                    if amr_at_1 is not None
                    else float((class_rows["target_rank"] == 1).mean())
                ),
                "AMR@5": metric_or_compute(summary, "top5_activation_rate", class_rows["target_rank"] <= 5),
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
                "Generated AMR@5": row["AMR@5"],
                "Real Top-5": real_top5,
            }
        )
    return pd.DataFrame(rows)


def build_confusion_table(config: dict, bundle: dict, top_n: int) -> pd.DataFrame:
    inference_results = bundle["inference_results"]
    rows = []
    for target_class in target_classes(config):
        class_rows = inference_results[inference_results["target_class"] == target_class]
        if "top1_correct" in class_rows.columns:
            wrong_rows = class_rows[~class_rows["top1_correct"].astype(str).str.lower().eq("true")]
        else:
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
    initial_metadata = bundle.get("initial_prompt_metadata") or {}
    descriptor_memory = bundle.get("descriptor_memory") or {}
    cue_synthesis = bundle.get("cue_synthesis") or {}
    rows = []
    for target_class in target_classes(config):
        initial = initial_metadata.get(target_class, {})
        descriptors = descriptor_memory.get(target_class, [])
        cues = cue_synthesis.get(target_class, {})
        rows.append(
            {
                "Class": target_class,
                "Initial prompt source": initial.get("source", ""),
                "Initial prompt": initial.get("prompt", ""),
                "Final prompt": final_prompts.get(target_class, ""),
                "Positive descriptors": "; ".join(descriptors[:12]),
                "Object/core cues": "; ".join(cues.get("object_or_core_cues", [])),
                "Candidate contextual cues": "; ".join(
                    cues.get("candidate_contextual_spurious_cues", [])
                ),
            }
        )
    return pd.DataFrame(rows)


def build_gradcam_selected_examples_table(config: dict, bundle: dict) -> pd.DataFrame:
    selected = bundle.get("xai_selected_examples")
    rows = []

    for target_class in target_classes(config):
        row = lookup_row(selected, "target_class", target_class)
        rows.append(
            {
                "Class": target_class,
                "Case": row.get("case_type"),
                "Selected top-1": row.get("top1_label"),
                "Selected rank": to_float(row.get("target_rank")),
                "Selected confidence": to_float(row.get("target_confidence")),
                "AMR@1": to_float(row.get("amr_at_1")),
                "AMR@5": to_float(row.get("amr_at_5")),
                "Grad-CAM path": row.get("gradcam_npy_path"),
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
    gradcam_table: pd.DataFrame,
) -> None:
    tables = {
        "main_results_table": main_table,
        "real_vs_generated_table": real_vs_generated,
        "confusion_distribution_table": confusion_table,
        "optimization_best_steps_table": optimization_table,
        "guardrail_rejections_table": guardrail_table,
        "final_prompts_table": prompt_table,
        "gradcam_selected_examples_table": gradcam_table,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    write_latex_table(
        main_table,
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
        gradcam_table,
        output_dir / "gradcam_selected_examples_table.tex",
        caption="Selected target-class Grad-CAM examples.",
        label="tab:gradcam-selected-examples",
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


def create_real_vs_generated_plot(table: pd.DataFrame, output_dir: Path) -> None:
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


def create_gradcam_presentation_figure(config: dict, bundle: dict, output_dir: Path) -> None:
    selected = bundle.get("xai_selected_examples")
    if selected is None or selected.empty:
        return

    rows = []
    for target_class in target_classes(config):
        row = lookup_row(selected, "target_class", target_class)
        if row:
            rows.append(row)

    if not rows:
        return

    n_rows = len(rows)
    figure, axes = plt.subplots(n_rows, 2, figsize=(7.0, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row_axes, row in zip(axes, rows):
        image_path = resolve_project_file(row.get("image_path"))
        image = load_image_or_none(image_path)
        if image is None:
            for axis in row_axes:
                axis.axis("off")
            row_axes[0].set_title(f"{row.get('target_class', '')}: image missing")
            continue

        row_axes[0].imshow(image)
        row_axes[0].set_title("Original", fontsize=9)
        row_axes[0].set_ylabel(row_label(row), fontsize=8, rotation=0, labelpad=62, va="center")
        row_axes[0].axis("off")

        heatmap_array = load_npy_from_row(row, "gradcam_npy_path")
        row_axes[1].imshow(image)
        if heatmap_array is not None:
            row_axes[1].imshow(resize_heatmap(heatmap_array, image.size), cmap="jet", alpha=0.45)
        row_axes[1].set_title("Target Grad-CAM", fontsize=9)
        row_axes[1].axis("off")

    figure.tight_layout(pad=0.5)
    figure.savefig(output_dir / "gradcam_presentation_grid.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def row_label(row: dict) -> str:
    return (
        f"{row.get('target_class', '')}\n"
        f"{row.get('case_type', '')}\n"
        f"top-1: {row.get('top1_label', '')}\n"
        f"rank: {format_rank(row.get('target_rank'))}\n"
        f"conf: {format_float(row.get('target_confidence'), digits=3)}"
    )


def build_text_summary(
    output_dir: Path,
    main_table: pd.DataFrame,
    real_vs_generated: pd.DataFrame,
    confusion_table: pd.DataFrame,
    optimization_table: pd.DataFrame,
    guardrail_table: pd.DataFrame,
    prompt_table: pd.DataFrame,
    gradcam_table: pd.DataFrame,
) -> str:
    lines = []
    lines.append("TextBack Final Results")
    lines.append("=" * 22)
    lines.append(f"Output folder: {output_dir}")
    lines.append("")

    lines.append("1. Activation maximization")
    for row in main_table.to_dict("records"):
        lines.append(
            f"  {row['Class']}: AMR@1={format_float(row['AMR@1'])}, "
            f"AMR@5={format_float(row['AMR@5'])}"
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
    prompt_lookup = {row["Class"]: row for row in prompt_table.to_dict("records")}
    for row in optimization_table.to_dict("records"):
        guardrail = guardrail_lookup.get(row["Class"], {})
        prompt = prompt_lookup.get(row["Class"], {})
        lines.append(
            f"  {row['Class']}: best_step={format_rank(row['Best iteration'])}, "
            f"best_conf={format_float(row['Best confidence'])}, "
            f"best_rank={format_rank(row['Best rank'])}, "
            f"rejected={format_rank(guardrail.get('Rejected updates'))}/"
            f"{format_rank(guardrail.get('Total updates'))}, "
            f"initial_source={prompt.get('Initial prompt source', 'n/a')}"
        )
    lines.append("")

    lines.append("4. Confusion patterns")
    for row in confusion_table.to_dict("records"):
        lines.append(f"  {row['Class']}: {row['Most frequent wrong top-1 predictions']}")
    lines.append("")

    lines.append("5. Grad-CAM summary")
    if gradcam_table.empty or gradcam_table["Selected top-1"].isna().all():
        lines.append("  Grad-CAM selected examples not found. Run the XAI computation before building final assets.")
    else:
        for row in gradcam_table.to_dict("records"):
            lines.append(
                f"  {row['Class']}: case={row.get('Case')}, "
                f"selected_top1={row.get('Selected top-1')}, "
                f"rank={format_rank(row.get('Selected rank'))}, "
                f"confidence={format_float(row.get('Selected confidence'), digits=3)}, "
                f"AMR@1={format_float(row.get('AMR@1'), digits=3)}, "
                f"AMR@5={format_float(row.get('AMR@5'), digits=3)}"
            )
        lines.append(
            "  Grad-CAM is a post-hoc diagnostic. It shows where the target-class logit is spatially supported, "
            "but it does not prove causal shortcut reliance."
        )
        lines.append(
            "  In failure cases, Grad-CAM may still highlight semantically relevant regions, but confidence and "
            "rank remain weak. Therefore Grad-CAM must be interpreted together with AMR, target confidence, "
            "and target rank."
        )
    lines.append("")

    lines.append("6. Files written")
    for name in final_asset_names():
        path = output_dir / name
        lines.append(f"  {path.relative_to(PROJECT_ROOT)}")

    lines.append("")
    lines.append("Interpretation guardrail: these results support candidate classifier-salient cues, not definitive causal proof of spurious features.")
    return "\n".join(lines)


def final_asset_names() -> list[str]:
    return [
        "main_results_table.csv",
        "main_results_table.tex",
        "real_vs_generated_table.csv",
        "real_vs_generated_table.tex",
        "confusion_distribution_table.csv",
        "confusion_distribution_table.tex",
        "optimization_best_steps_table.csv",
        "optimization_trajectory.png",
        "guardrail_rejections_table.csv",
        "final_prompts_table.csv",
        "generated_vs_real_top1.png",
        "gradcam_selected_examples_table.csv",
        "gradcam_selected_examples_table.tex",
        "gradcam_presentation_grid.png",
        "summary.txt",
        "summary.md",
    ]


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


def resolve_project_file(value: str | None) -> Path | None:
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


def load_npy_from_row(row: dict, key: str):
    path = resolve_project_file(row.get(key))
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
