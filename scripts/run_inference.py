"""Run TextBack inference evaluation from the command line."""

import argparse
import sys
from pathlib import Path

# Make imports work when launching from the project root:
# python scripts/run_inference.py --config configs/default.yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.pipeline import TextBackPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run TextBack inference.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Load config, create the pipeline, and run inference."""
    args = parse_args()
    config = load_config(args.config)
    final_prompts_path = Path(config["paths"]["results_dir"]) / "final_prompts.json"
    if not final_prompts_path.exists():
        print("Missing results/final_prompts.json. Run optimization successfully before inference.")
        raise SystemExit(1)

    pipeline = TextBackPipeline(config)
    activation_rates = pipeline.run_inference()

    print("Inference results saved to results/inference_results.csv")
    for target_class, rate in activation_rates.items():
        print(f"- {target_class}: activation maximization rate = {rate:.3f}")


if __name__ == "__main__":
    main()
