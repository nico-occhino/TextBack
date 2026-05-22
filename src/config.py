"""Configuration helpers for TextBack.

The project uses one YAML file to keep experiments easy to change during an
oral exam. This module intentionally stays small: load the YAML, create output
folders, and return a normal Python dictionary.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        A nested dictionary with all configuration values.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def prepare_output_dirs(config: Dict[str, Any]) -> None:
    """Create result folders used by the pipeline.

    Args:
        config: Loaded configuration dictionary.
    """
    paths = config["paths"]
    Path(paths["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["generated_images_dir"]).mkdir(parents=True, exist_ok=True)


def is_dry_run(config: Dict[str, Any]) -> bool:
    """Return whether the experiment should use dummy components.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        True when dummy components should be preferred.
    """
    return bool(config.get("project", {}).get("dry_run", True))
