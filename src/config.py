"""Configuration helpers for TextBack.

The project is driven by one YAML file.  The functions here stay intentionally
small so they are easy to explain and edit during an oral exam.
"""

from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return a plain dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration values as nested Python dictionaries.
    """
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")

    try:
        import yaml

        config = yaml.safe_load(text)
    except ModuleNotFoundError:
        # Small fallback for the simple default config.  Installing PyYAML is
        # still recommended, but this keeps the dry run usable on bare Python.
        config = _parse_simple_yaml(text)

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Check that the config contains all required top-level sections.

    Args:
        config: Loaded configuration dictionary.

    Raises:
        ValueError: If a required section is missing.
    """
    required_sections = [
        "project",
        "paths",
        "experiment",
        "llm",
        "image_generator",
        "classifier",
    ]

    # Fail early with a clear message. This is easier to debug during an exam
    # than getting a KeyError later inside the pipeline.
    found_sections = list(config.keys()) if isinstance(config, dict) else []
    for section in required_sections:
        if section not in found_sections:
            raise ValueError(
                f"Missing config section: {section}. Found sections: {found_sections}"
            )


def create_output_dirs(config: dict[str, Any]) -> None:
    """Create the output directories used by optimization and inference.

    Args:
        config: Loaded configuration dictionary.
    """
    paths = config["paths"]
    Path(paths["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["generated_images_dir"]).mkdir(parents=True, exist_ok=True)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small subset of YAML used by configs/default.yaml.

    Args:
        text: Raw YAML text.

    Returns:
        Nested dictionary with strings, numbers, booleans, and simple lists.
    """
    result: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith(" ") and stripped.endswith(":"):
            current_section = stripped[:-1]
            current_list_key = None
            result[current_section] = {}
            continue

        if current_section is None:
            continue

        if stripped.startswith("- ") and current_list_key is not None:
            result[current_section][current_list_key].append(_parse_scalar(stripped[2:]))
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value == "":
                result[current_section][key] = []
                current_list_key = key
            else:
                result[current_section][key] = _parse_scalar(value)
                current_list_key = None

    return result


def _parse_scalar(value: str) -> Any:
    """Convert one simple YAML scalar into a Python value.

    Args:
        value: Raw scalar text.

    Returns:
        Parsed Python value.
    """
    value = value.strip().strip('"').strip("'")

    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
