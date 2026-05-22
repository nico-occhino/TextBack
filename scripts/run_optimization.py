"""Command line entry point for TextBack optimization."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_optimization


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Run TextBack prompt optimization.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load the config and run optimization."""
    args = parse_args()
    config = load_config(args.config)
    final_prompts = run_optimization(config)

    print("Optimization finished. Final prompts:")
    for class_name, prompt in final_prompts.items():
        print(f"- {class_name}: {prompt}")


if __name__ == "__main__":
    main()
