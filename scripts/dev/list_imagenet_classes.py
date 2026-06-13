"""List ImageNet class labels from torchvision ResNet50 weights."""

import argparse

from torchvision.models import ResNet50_Weights


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="List ImageNet class labels used by ResNet50.")
    parser.add_argument("--query", default=None, help="Optional case-insensitive label filter.")
    return parser.parse_args()


def main() -> None:
    """Print ImageNet labels, optionally filtered by a query."""
    args = parse_args()

    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]

    query = args.query.lower() if args.query else None
    for index, label in enumerate(categories):
        if query is not None and query not in label.lower():
            continue
        print(f"{index}: {label}")


if __name__ == "__main__":
    main()
