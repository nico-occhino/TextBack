"""Small utility functions shared across the project."""

import random
import re
from pathlib import Path


def set_seed(seed: int) -> None:
    """Seed common random number generators for reproducibility.

    This improves reproducibility, but full bitwise determinism is not
    guaranteed for all CUDA/Diffusers operations.

    Args:
        seed: Integer seed used for reproducible experiments.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


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
