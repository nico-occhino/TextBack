import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerAttribution, LayerGradCam
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import build_classifier
from src.config import load_config


GRADCAM_LAYER_PATH = "model.model.layer4[-1].conv2"

SELECTED_COLUMNS = [
    "target_class",
    "case_type",
    "image_path",
    "top1_label",
    "target_rank",
    "target_confidence",
    "amr_at_1",
    "amr_at_5",
    "gradcam_npy_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute TextBack Grad-CAM artifacts.")
    parser.add_argument("--config", default="configs/final.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = project_path("results/xai")
    selected_dir = output_dir / "selected"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    classifier = build_classifier(config)
    target_layer = resolve_layer(classifier.model, GRADCAM_LAYER_PATH)
    print(f"Selected Grad-CAM layer: {GRADCAM_LAYER_PATH}")

    inference_path = project_path("results/inference_results.csv")
    inference_results = pd.read_csv(inference_path)
    inference_results["target_rank"] = pd.to_numeric(inference_results["target_rank"], errors="coerce")
    inference_results["target_confidence"] = pd.to_numeric(
        inference_results["target_confidence"],
        errors="coerce",
    )

    selected_rows = []

    for target_class in config["experiment"]["target_classes"]:
        selected = select_representative_row(inference_results, target_class)
        if selected is None:
            print(f"No inference rows found for {target_class}.")
            continue

        selected_row, case_type, amr_at_1, amr_at_5 = selected
        image_path = project_path(selected_row["image_path"])
        class_slug = slugify(target_class)
        class_selected_dir = selected_dir / class_slug
        class_selected_dir.mkdir(parents=True, exist_ok=True)
        gradcam_path = class_selected_dir / f"{class_slug}_gradcam.npy"

        gradcam_heatmap = compute_gradcam_heatmap(classifier, image_path, target_class, target_layer)
        np.save(gradcam_path, gradcam_heatmap)

        selected_rows.append(
            {
                "target_class": target_class,
                "case_type": case_type,
                "image_path": project_relative_text(image_path),
                "top1_label": selected_row["top1_label"],
                "target_rank": int(selected_row["target_rank"]),
                "target_confidence": float(selected_row["target_confidence"]),
                "amr_at_1": amr_at_1,
                "amr_at_5": amr_at_5,
                "gradcam_npy_path": project_relative_text(gradcam_path),
            }
        )

    selected_examples = pd.DataFrame(selected_rows, columns=SELECTED_COLUMNS)
    selected_examples.to_csv(output_dir / "xai_selected_examples.csv", index=False)

    print(f"Grad-CAM artifacts written to {output_dir}")


def compute_gradcam_heatmap(classifier, image_path: Path, target_class: str, target_layer) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    input_tensor = classifier.preprocess(image).unsqueeze(0).to(classifier.device)
    input_tensor.requires_grad_(True)
    target_index = target_class_index(classifier, target_class)

    with torch.enable_grad():
        layer_gradcam = LayerGradCam(classifier.model, target_layer)
        attributions = layer_gradcam.attribute(input_tensor, target=target_index)
        upsampled = LayerAttribution.interpolate(attributions, input_tensor.shape[2:])

    heatmap = upsampled.squeeze().detach().cpu().numpy()
    if heatmap.ndim == 3:
        heatmap = heatmap.mean(axis=0)
    heatmap = np.maximum(heatmap, 0.0).astype(np.float32)
    return heatmap


def select_representative_row(results: pd.DataFrame, target_class: str):
    rows = results[results["target_class"] == target_class].copy()
    rows = rows.dropna(subset=["target_rank", "target_confidence"])
    if rows.empty:
        return None

    amr_at_1 = float((rows["target_rank"] == 1).mean())
    amr_at_5 = float((rows["target_rank"] <= 5).mean())
    if amr_at_1 >= 0.30:
        case_type = "success"
        preferred = rows[rows["target_rank"] == 1]
        selected = preferred.sort_values("target_confidence", ascending=False).iloc[0]
    elif amr_at_5 >= 0.30:
        case_type = "partial"
        selected = rows.sort_values(["target_rank", "target_confidence"], ascending=[True, False]).iloc[0]
    else:
        case_type = "failure"
        selected = rows.sort_values(["target_rank", "target_confidence"], ascending=[True, False]).iloc[0]

    return selected, case_type, amr_at_1, amr_at_5


def resolve_layer(model, layer_path: str):
    current = model
    parts = layer_path.split(".")
    if parts and parts[0] == "model":
        parts = parts[1:]

    for part in parts:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(-?\d+)\])?", part)
        if match is None:
            raise RuntimeError(f"Unsupported Grad-CAM layer path syntax: {layer_path}")

        attribute_name, index_text = match.groups()
        if not hasattr(current, attribute_name):
            raise RuntimeError(f"Configured Grad-CAM layer was not found: {layer_path}")
        current = getattr(current, attribute_name)
        if index_text is not None:
            try:
                current = current[int(index_text)]
            except (IndexError, TypeError, KeyError) as error:
                raise RuntimeError(f"Configured Grad-CAM layer was not found: {layer_path}") from error

    return current


def target_class_index(classifier, target_class: str) -> int:
    if hasattr(classifier, "label_to_index"):
        target_index = classifier.label_to_index.get(target_class)
        if target_index is not None:
            return int(target_index)
    return int(classifier.labels.index(target_class))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown_class"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative_text(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
