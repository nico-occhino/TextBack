from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import LayerAttribution, LayerGradCam
from captum.attr import visualization as viz
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import build_classifier
from src.config import load_config


REPORT_CASES = [
    ("cowboy hat", "success"),
    ("volcano", "success"),
    ("sports car", "partial"),
    ("tabby", "failure"),
]

METADATA_COLUMNS = [
    "target_class",
    "case_type",
    "image_path",
    "gradcam_path",
    "occlusion_path",
    "top1_label",
    "target_rank",
    "target_confidence",
    "max_occlusion_drop",
    "mean_positive_occlusion_drop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TextBack XAI on selected generated examples.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output-dir", default="results/xai")
    parser.add_argument("--patch-size", type=int, default=56)
    parser.add_argument("--stride", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = build_classifier(config)
    target_layer, layer_name = find_gradcam_layer(classifier.model)
    print(f"Selected Grad-CAM layer: {layer_name}")

    inference_path = project_path(config["paths"]["results_dir"]) / "inference_results.csv"
    inference_results = pd.read_csv(inference_path)
    inference_results["target_rank"] = pd.to_numeric(inference_results["target_rank"])
    inference_results["target_confidence"] = pd.to_numeric(inference_results["target_confidence"])

    metadata_path = output_dir / "xai_metadata.csv"
    if metadata_path.exists():
        metadata_path.unlink()

    metadata_rows = []
    for target_class, case_type in REPORT_CASES:
        row = select_case_row(inference_results, target_class, case_type)
        if row is None:
            print(f"No inference example found for {target_class} ({case_type}).")
            continue

        image_path = project_path(row["image_path"])
        class_slug = slugify(target_class)
        gradcam_path = output_dir / class_slug / f"{class_slug}_gradcam.png"
        occlusion_path = output_dir / class_slug / f"{class_slug}_occlusion.png"

        gradcam_result = save_gradcam(classifier, image_path, target_class, target_layer, gradcam_path)
        occlusion_result = save_occlusion(
            classifier,
            image_path,
            target_class,
            occlusion_path,
            patch_size=args.patch_size,
            stride=args.stride,
        )

        metadata_row = {
            "target_class": target_class,
            "case_type": case_type,
            "image_path": str(image_path),
            "gradcam_path": str(gradcam_path),
            "occlusion_path": str(occlusion_path),
            "top1_label": gradcam_result["top1_label"],
            "target_rank": gradcam_result["target_rank"],
            "target_confidence": gradcam_result["target_confidence"],
            "max_occlusion_drop": occlusion_result["max_drop"],
            "mean_positive_occlusion_drop": occlusion_result["mean_positive_drop"],
        }
        append_metadata(metadata_path, metadata_row)
        metadata_rows.append(metadata_row)

    make_summary_figure(metadata_rows, "gradcam_path", output_dir / "gradcam_four_examples.png", output_dir / "gradcam_four_examples.pdf")
    make_summary_figure(metadata_rows, "occlusion_path", output_dir / "occlusion_four_examples.png", output_dir / "occlusion_four_examples.pdf")
    print(f"XAI outputs written to {output_dir}")


def select_case_row(results: pd.DataFrame, target_class: str, case_type: str):
    rows = results[results["target_class"] == target_class].copy()
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
    return preferred.sort_values("target_confidence", ascending=False).iloc[0]


def save_gradcam(classifier, image_path: Path, target_class: str, target_layer, output_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    input_tensor = classifier.preprocess(image).unsqueeze(0).to(classifier.device)
    target_index = target_class_index(classifier, target_class)

    with torch.enable_grad():
        layer_gradcam = LayerGradCam(classifier.model, target_layer)
        attributions = layer_gradcam.attribute(input_tensor, target=target_index)
        upsampled = LayerAttribution.interpolate(attributions, input_tensor.shape[2:])

    attribution = upsampled.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    visual_image = tensor_to_visual_image(input_tensor)

    figure, _ = viz.visualize_image_attr_multiple(
        attribution,
        visual_image,
        methods=["original_image", "blended_heat_map", "masked_image"],
        signs=["all", "positive", "positive"],
        titles=["Original", "Grad-CAM", "Masked"],
        show_colorbar=True,
        use_pyplot=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    return classifier.predict(image_path=image_path, target_class=target_class, top_k=5)


def save_occlusion(
    classifier,
    image_path: Path,
    target_class: str,
    output_path: Path,
    patch_size: int,
    stride: int,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    input_tensor = classifier.preprocess(image).unsqueeze(0).to(classifier.device)
    target_index = target_class_index(classifier, target_class)

    with torch.no_grad():
        base_probabilities = torch.softmax(classifier.model(input_tensor), dim=1)[0]
        base_confidence = float(base_probabilities[target_index].item())

    height, width = input_tensor.shape[2:]
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for y_start in range(0, height, stride):
        for x_start in range(0, width, stride):
            y_end = min(y_start + patch_size, height)
            x_end = min(x_start + patch_size, width)
            occluded = input_tensor.clone()
            occluded[:, :, y_start:y_end, x_start:x_end] = 0.0
            with torch.no_grad():
                probabilities = torch.softmax(classifier.model(occluded), dim=1)[0]
                occluded_confidence = float(probabilities[target_index].item())
            drop = base_confidence - occluded_confidence
            heatmap[y_start:y_end, x_start:x_end] += drop
            counts[y_start:y_end, x_start:x_end] += 1.0

    heatmap = heatmap / np.maximum(counts, 1.0)
    positive_heatmap = np.maximum(heatmap, 0.0)
    max_drop = float(positive_heatmap.max())
    mean_positive_drop = float(positive_heatmap[positive_heatmap > 0].mean()) if np.any(positive_heatmap > 0) else 0.0
    normalized = positive_heatmap / max_drop if max_drop > 0 else positive_heatmap
    visual_image = tensor_to_visual_image(input_tensor)

    figure, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(visual_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(visual_image)
    axes[1].imshow(normalized, cmap="hot", alpha=0.45)
    axes[1].set_title("Occlusion drop")
    axes[1].axis("off")
    heat = axes[2].imshow(normalized, cmap="hot")
    axes[2].set_title("Drop map")
    axes[2].axis("off")
    figure.colorbar(heat, ax=axes.ravel().tolist(), shrink=0.7)
    figure.suptitle(
        f"{target_class}: base target confidence {base_confidence:.3f}, max drop {max_drop:.3f}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)

    return {
        "base_confidence": base_confidence,
        "max_drop": max_drop,
        "mean_positive_drop": mean_positive_drop,
    }


def find_gradcam_layer(model):
    candidates = [
        ("model.layer4[-1].conv2", lambda item: item.layer4[-1].conv2),
        ("model.model.layer4[-1].conv2", lambda item: item.model.layer4[-1].conv2),
        ("model.module.layer4[-1].conv2", lambda item: item.module.layer4[-1].conv2),
        ("model.layer4[-1]", lambda item: item.layer4[-1]),
        ("model.model.layer4[-1]", lambda item: item.model.layer4[-1]),
        ("model.module.layer4[-1]", lambda item: item.module.layer4[-1]),
    ]
    for name, getter in candidates:
        try:
            return getter(model), name
        except (AttributeError, IndexError, TypeError):
            continue

    last_conv_name = None
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
            last_conv = module

    if last_conv is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")
    return last_conv, last_conv_name


def make_summary_figure(rows: list[dict], path_key: str, png_path: Path, pdf_path: Path) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, row in zip(axes.ravel(), rows):
        path = Path(row[path_key])
        image = Image.open(path).convert("RGB")
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{row['target_class']} ({row['case_type']})\n"
            f"top-1: {row['top1_label']}, rank: {row['target_rank']}, "
            f"conf: {float(row['target_confidence']):.3f}",
            fontsize=9,
        )
    for axis in axes.ravel()[len(rows):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)


def append_metadata(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def target_class_index(classifier, target_class: str) -> int:
    if hasattr(classifier, "label_to_index"):
        target_index = classifier.label_to_index.get(target_class)
        if target_index is not None:
            return int(target_index)
    return int(classifier.labels.index(target_class))


def tensor_to_visual_image(input_tensor) -> np.ndarray:
    image = input_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    return np.clip(image, 0.0, 1.0)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown_class"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
