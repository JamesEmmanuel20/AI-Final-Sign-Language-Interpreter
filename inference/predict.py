"""Predict an ASL letter from 21 normalized hand landmarks."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .model import load_model
from .normalize import FEATURE_NAMES, NUM_FEATURES, normalize_landmarks


def predict_letter(
    landmarks: Sequence[Any],
    model: Any | None = None,
) -> str | None:
    """
    Normalize 21 landmarks and predict a letter with the Random Forest.

    Pass an optional preloaded model from load_model() to avoid reloading.
    Returns a Python str letter, or None if normalization fails (degenerate hand).
    No label encoder — model.predict() already returns letter strings.
    """
    features = normalize_landmarks(landmarks)
    if features is None:
        return None

    if features.shape != (NUM_FEATURES,):
        raise ValueError(
            f"Expected normalized features shape ({NUM_FEATURES},), "
            f"got {features.shape}."
        )

    estimator = model if model is not None else load_model()

    # Use training column names when the estimator stored them
    if hasattr(estimator, "feature_names_in_"):
        x = pd.DataFrame([features], columns=FEATURE_NAMES)
    else:
        x = features.reshape(1, NUM_FEATURES)

    prediction = estimator.predict(x)
    return str(prediction[0])
