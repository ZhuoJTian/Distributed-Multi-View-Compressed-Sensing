"""Filesystem helpers shared by experiment scripts."""
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create *path* and its parents when needed, then return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
