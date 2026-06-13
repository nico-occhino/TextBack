"""Smoke-test the configured TextGrad LiteLLM backend."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the configured LLM backend.")
    parser.add_argument("--config", default="configs/final.yaml", help="Path to config YAML.")
    return parser.parse_args()


def main() -> None:
    """Call LiteLLM directly with the configured backend."""
    _load_env_file()
    args = parse_args()
    config = load_config(args.config)
    model = _litellm_model_name(config["textgrad"]["backward_engine"])

    from litellm import completion

    response = completion(
        model=model,
        messages=[{"role": "user", "content": "Reply only: ok"}],
        max_tokens=5,
    )
    print(repr(response.choices[0].message.content))


def _load_env_file() -> None:
    """Load .env if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _litellm_model_name(backend: str) -> str:
    """Convert a TextGrad backend name to a LiteLLM model name."""
    backend = backend.strip()
    if backend.startswith("experimental:"):
        backend = backend.removeprefix("experimental:")
    if backend.startswith("experimental/"):
        backend = backend.removeprefix("experimental/")
    if backend.startswith("gpt-"):
        return f"openai/{backend}"
    return backend


if __name__ == "__main__":
    main()
