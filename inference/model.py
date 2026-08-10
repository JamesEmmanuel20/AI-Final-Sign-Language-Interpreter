"""Load the trained ASL Random Forest artifact for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

# Project root = parent of the inference/ package directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "random_forest_model.pkl"


def load_model(model_path: str | Path | None = None) -> Any:
    """
    Load Miriam's trained RandomForestClassifier from disk.

    Problem this solves:
      The webcam/app layer should not know how the pickle was saved. This
      helper centralizes path resolution and clear load failures.

    Input:
      model_path — optional path to the .pkl file. If omitted, loads
      random_forest_model.pkl from the project root (next to this package).

    Output:
      The deserialized sklearn estimator (RandomForestClassifier).

    Why joblib:
      The training notebook saved the model with joblib.dump(...), so
      joblib.load(...) is the correct inverse and handles sklearn/numpy
      payloads reliably.
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
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 — surface any unpickle/IO failure clearly
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc

    return model
