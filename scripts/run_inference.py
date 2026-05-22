"""Command line entry point for TextBack inference evaluation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_inference


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Run TextBack inference from final prompts.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load the config and run inference evaluation."""
    args = parse_args()
    config = load_config(args.config)
    metrics = run_inference(config)

    print("Inference finished. Activation maximization rates:")
    for class_name, result in metrics.items():
        rate = result["activation_maximization_rate"]
        print(f"- {class_name}: {rate:.3f}")


if __name__ == "__main__":
    main()
