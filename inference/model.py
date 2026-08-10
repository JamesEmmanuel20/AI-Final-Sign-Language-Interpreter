"""Load Miriam's trained Random Forest (.pkl) for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

# Project root (parent of inference/), so the default path works from any cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "random_forest_model.pkl"


def load_model(model_path: str | Path | None = None) -> Any:
    """
    Load random_forest_model.pkl with joblib (same format used when saving).

    Defaults to the project-root pickle. Raises FileNotFoundError if missing,
    or RuntimeError if the file cannot be loaded.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Expected random_forest_model.pkl in the project root, "
            "or pass an explicit model_path."
        )

    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc
