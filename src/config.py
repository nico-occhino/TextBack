"""Configuration helpers for TextBack."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML config and validate the expected sections.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration values as a plain dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Check that all required top-level config sections exist.

    Args:
        config: Loaded configuration dictionary.

    Raises:
        ValueError: If a required section is missing.
    """
    required_sections = [
        "project",
        "paths",
        "experiment",
        "textual_optimizer",
        "textgrad",
        "image_generator",
        "classifier",
    ]

    found_sections = list(config.keys()) if isinstance(config, dict) else []
    for section in required_sections:
        if section not in found_sections:
            raise ValueError(
                f"Missing config section: {section}. Found sections: {found_sections}"
            )


def create_output_dirs(config: dict[str, Any]) -> None:
    """Create the output folders used by the project.

    Args:
        config: Loaded configuration dictionary.
    """
    paths = config["paths"]
    Path(paths["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["generated_images_dir"]).mkdir(parents=True, exist_ok=True)
