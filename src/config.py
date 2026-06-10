"""Configuration helpers for TextBack."""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML config and validate the expected sections.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration values as a plain dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        if yaml is not None:
            config = yaml.safe_load(file)
        else:
            config = _load_simple_yaml(file.read())

    validate_config(config)
    return config


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple YAML subset used by TextBack configs.

    This fallback keeps command-line scripts usable in minimal environments.
    Install PyYAML for general YAML support.
    """
    config: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        stripped = line_without_comment.strip()

        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1]
            current_list_key = None
            config[current_section] = {}
            continue

        if current_section is None:
            raise ValueError(f"Invalid config line outside a section: {raw_line}")

        section = config[current_section]
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"Invalid list item without a key: {raw_line}")
            section[current_list_key].append(_parse_scalar(stripped[2:]))
            continue

        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Invalid config line: {raw_line}")

        if value.strip() == "":
            section[key] = []
            current_list_key = key
        else:
            section[key] = _parse_scalar(value)
            current_list_key = None

    return config


def _parse_scalar(value: str) -> Any:
    """Parse simple YAML scalar values."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


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
        "textgrad",
        "descriptor_memory",
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
