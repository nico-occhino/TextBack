"""Run TextBack prompt optimization from the command line."""

import argparse
import sys
from pathlib import Path

# Make imports work when launching from the project root:
# python scripts/run_optimization.py --config configs/default.yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.pipeline import TextBackPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run TextBack optimization.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load config, create the pipeline, and run optimization."""
    args = parse_args()
    config = load_config(args.config)

    pipeline = TextBackPipeline(config)
    try:
        final_prompts = pipeline.run_optimization()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(1) from error

    print("Final prompts saved to results/final_prompts.json")
    for target_class, prompt in final_prompts.items():
        print(f"- {target_class}: {prompt}")


if __name__ == "__main__":
    main()
