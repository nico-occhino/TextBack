"""Run post-hoc Grad-CAM analysis for TextBack results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from captum.attr import LayerAttribution, LayerGradCam
    from captum.attr import visualization as viz
    from PIL import Image
except ImportError as error:
    GRADCAM_IMPORT_ERROR = error
else:
    GRADCAM_IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import build_classifier
from src.config import load_config
from src.logging_utils import append_csv_row


METADATA_COLUMNS = [
    "source_type",
    "target_class",
    "image_path",
    "output_path",
    "top1_label",
    "target_rank",
    "target_confidence",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TextBack Grad-CAM analysis.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--max-generated-per-class", type=int, default=5)
    parser.add_argument("--max-real-per-class", type=int, default=5)
    parser.add_argument("--output-dir", default="results/gradcam")
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown_class"


def main() -> None:
    """Run Grad-CAM over selected generated and real images."""
    args = parse_args()
    if GRADCAM_IMPORT_ERROR is not None:
        print(f"Grad-CAM dependencies unavailable: {GRADCAM_IMPORT_ERROR}")
        return

    config = load_config(args.config)
    output_dir = _resolve_project_path(args.output_dir)
    metadata_path = output_dir / "gradcam_metadata.csv"
    if metadata_path.exists():
        metadata_path.unlink()

    try:
        classifier = build_classifier(config)
    except RuntimeError as error:
        print(f"Grad-CAM unavailable: {error}")
        return

    target_layer, layer_name = find_gradcam_layer(classifier.model)
    print(f"Selected Grad-CAM layer: {layer_name} ({type(target_layer).__name__})")

    generated_rows = _select_generated_rows(
        results_dir=_resolve_project_path(config["paths"]["results_dir"]),
        max_per_class=args.max_generated_per_class,
    )
    _run_generated_gradcam(classifier, generated_rows, target_layer, output_dir, metadata_path)
    _run_real_gradcam(
        classifier=classifier,
        config=config,
        target_layer=target_layer,
        output_dir=output_dir,
        metadata_path=metadata_path,
        max_per_class=args.max_real_per_class,
    )


def find_gradcam_layer(model):
    """Find a suitable final convolutional layer for Grad-CAM."""
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


def run_layer_gradcam(classifier, image_path: Path, target_class: str, target_layer, output_path: Path) -> dict:
    """Run LayerGradCam for one image and save an overlay figure."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = classifier.preprocess(image).unsqueeze(0).to(classifier.device)
    target_index = _target_index(classifier, target_class)

    with torch.enable_grad():
        layer_gradcam = LayerGradCam(classifier.model, target_layer)
        attributions = layer_gradcam.attribute(input_tensor, target=target_index)
        upsampled = LayerAttribution.interpolate(attributions, input_tensor.shape[2:])

    attribution = upsampled.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    visual_image = _tensor_to_visual_image(input_tensor)

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


def _select_generated_rows(results_dir: Path, max_per_class: int) -> list[dict]:
    """Select generated images from saved inference results."""
    path = results_dir / "inference_results.csv"
    if not path.exists():
        print(f"Generated inference results not found: {path}")
        return []

    rows_by_class = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows_by_class.setdefault(row["target_class"], []).append(row)

    selected = []
    for target_class, rows in rows_by_class.items():
        rows.sort(key=_generated_selection_key)
        selected.extend(rows[:max_per_class])
    return selected


def _generated_selection_key(row: dict) -> tuple[int, float]:
    """Sort generated rows by target rank preference and confidence."""
    rank = _to_int(row.get("target_rank"))
    confidence = _to_float(row.get("target_confidence"))
    if rank == 1:
        bucket = 0
    elif rank is not None and rank <= 5:
        bucket = 1
    else:
        bucket = 2
    return bucket, -confidence


def _run_generated_gradcam(classifier, rows: list[dict], target_layer, output_dir: Path, metadata_path: Path) -> None:
    """Generate Grad-CAM overlays for selected generated images."""
    counters = {}
    for row in rows:
        target_class = row["target_class"]
        class_slug = slugify(target_class)
        index = counters.get(target_class, 0)
        counters[target_class] = index + 1
        image_path = _resolve_project_path(row["image_path"])
        output_path = output_dir / "generated" / class_slug / f"generated_{index:03d}.png"
        result = run_layer_gradcam(classifier, image_path, target_class, target_layer, output_path)
        _append_metadata(metadata_path, "generated", target_class, image_path, output_path, result)


def _run_real_gradcam(
    classifier,
    config: dict,
    target_layer,
    output_dir: Path,
    metadata_path: Path,
    max_per_class: int,
) -> None:
    """Generate Grad-CAM overlays for correctly classified real subset images."""
    subset_root = _resolve_project_path(config["paths"]["imagenet_subset_root"])
    for target_class in config["experiment"]["target_classes"]:
        class_dir = subset_root / target_class
        if not class_dir.exists():
            print(f"Real subset directory not found for {target_class}: {class_dir}")
            continue

        kept = 0
        for image_path in _iter_image_files(class_dir):
            result = classifier.predict(image_path=image_path, target_class=target_class, top_k=5)
            if result["target_rank"] != 1:
                continue

            class_slug = slugify(target_class)
            output_path = output_dir / "real" / class_slug / f"real_{kept:03d}.png"
            result = run_layer_gradcam(classifier, image_path, target_class, target_layer, output_path)
            _append_metadata(metadata_path, "real", target_class, image_path, output_path, result)
            kept += 1
            if kept >= max_per_class:
                break

        if kept < max_per_class:
            print(
                f"Warning: only {kept}/{max_per_class} real images correctly "
                f"classified for {target_class}."
            )


def _append_metadata(
    metadata_path: Path,
    source_type: str,
    target_class: str,
    image_path: Path,
    output_path: Path,
    result: dict,
) -> None:
    """Append one Grad-CAM metadata row."""
    append_csv_row(
        metadata_path,
        {
            "source_type": source_type,
            "target_class": target_class,
            "image_path": str(image_path),
            "output_path": str(output_path),
            "top1_label": result.get("top1_label"),
            "target_rank": result.get("target_rank"),
            "target_confidence": result.get("target_confidence"),
        },
        METADATA_COLUMNS,
    )


def _target_index(classifier, target_class: str) -> int:
    """Return the classifier index for a target class."""
    if hasattr(classifier, "label_to_index"):
        target_index = classifier.label_to_index.get(target_class)
        if target_index is not None:
            return int(target_index)
    return int(classifier.labels.index(target_class))


def _tensor_to_visual_image(input_tensor) -> np.ndarray:
    """Convert a transformed input tensor into an HWC image for visualization."""
    image = input_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    return np.clip(image, 0.0, 1.0)


def _iter_image_files(directory: Path):
    """Yield image files under a directory."""
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve paths relative to the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _to_int(value) -> int | None:
    """Parse optional integer values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float:
    """Parse optional float values for sorting."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
