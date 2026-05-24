"""Small utility functions shared across the project."""

import random
import re
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    """Set the Python random seed.

    Args:
        seed: Integer seed used for reproducible experiments.
    """
    random.seed(seed)


def slugify(text: str) -> str:
    """Convert a class name into a folder-friendly name.

    Args:
        text: Human readable class name.

    Returns:
        Lowercase string with spaces and punctuation replaced by underscores.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown_class"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path.

    Args:
        path: Directory path.

    Returns:
        The same path converted to a pathlib.Path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_nested(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    """Read a nested config value with a default.

    Args:
        config: Loaded configuration dictionary.
        section: Top-level key.
        key: Nested key inside the section.
        default: Value returned when the key is missing.

    Returns:
        The configured value or the default.
    """
    return config.get(section, {}).get(key, default)
