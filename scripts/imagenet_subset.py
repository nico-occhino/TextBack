"""Create a small ImageNet-1k subset with Hugging Face streaming.

This script does not download the full dataset.  It streams training examples
and saves only the requested number of images for the selected classes.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from torchvision.models import ResNet50_Weights


DEFAULT_CLASSES = ["tabby", "sports car", "cowboy hat", "volcano", "book jacket"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Stream a small ImageNet-1k subset.")
    parser.add_argument("--output-dir", default="data/imagenet_subset", help="Folder where images are saved.")
    parser.add_argument("--n-per-class", type=int, default=50, help="Number of images to save per class.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Exact ImageNet labels to collect.")
    return parser.parse_args()


def main() -> None:
    """Stream ImageNet and save images for the selected classes."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not found. Add HF_TOKEN=... to .env after accepting "
            "ImageNet-1k terms on Hugging Face."
        )

    args = parse_args()
    output_dir = Path(args.output_dir)
    class_to_id = build_class_id_map(args.classes)
    id_to_class = {class_id: class_name for class_name, class_id in class_to_id.items()}
    saved_counts = {class_name: 0 for class_name in args.classes}

    for class_name in args.classes:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, token=hf_token)
    
    for example in dataset:
        label = int(example["label"])
        if label not in id_to_class:
            continue

        class_name = id_to_class[label]
        if saved_counts[class_name] >= args.n_per_class:
            continue

        saved_counts[class_name] += 1
        image_number = saved_counts[class_name]
        image_path = output_dir / class_name / f"{image_number:03d}.png"

        # Convert to RGB so all saved files have the same image mode.
        image = example["image"].convert("RGB")
        image.save(image_path)
        print(f"Saved {class_name} {image_number}/{args.n_per_class}: {image_path}")

        if all(count >= args.n_per_class for count in saved_counts.values()):
            print("Finished: requested subset is complete.")
            return

    missing = {
        class_name: args.n_per_class - count
        for class_name, count in saved_counts.items()
        if count < args.n_per_class
    }
    if missing:
        print(f"Stream ended before all classes were complete: {missing}", file=sys.stderr)


def build_class_id_map(class_names: list[str]) -> dict[str, int]:
    """Map exact torchvision ImageNet labels to integer IDs.

    Args:
        class_names: Exact ImageNet labels requested by the user.

    Returns:
        Dictionary mapping class name to ImageNet class index.

    Raises:
        ValueError: If any class name is not an exact torchvision label.
    """
    categories = ResNet50_Weights.DEFAULT.meta["categories"]
    missing = [class_name for class_name in class_names if class_name not in categories]
    if missing:
        raise ValueError(
            f"Unknown ImageNet class labels: {missing}. "
            "Use scripts/list_imagenet_classes.py --query ... to find exact labels."
        )

    return {class_name: categories.index(class_name) for class_name in class_names}


if __name__ == "__main__":
    main()
