"""Letter prediction from 21 hand landmarks using Miriam's Random Forest."""

from __future__ import annotations

from typing import Any, Sequence

from .model import load_model
from .normalize import FEATURE_NAMES, NUM_FEATURES, normalize_landmarks


def predict_letter(
    landmarks: Sequence[Any],
    model: Any | None = None,
) -> str | None:
    """
    Predict an ASL letter from 21 MediaPipe-style landmarks.

    Problem this solves:
      Connects live (or test) landmarks to the trained Random Forest using the
      same normalization + 63-feature layout as training, without a label encoder.

    Input:
      landmarks — exactly 21 landmarks with x, y, z (MediaPipe objects or
      sequences). Landmark 0 is the wrist.
      model — optional preloaded estimator from load_model(). If omitted,
      load_model() is called so teammates can either inject a shared model
      or rely on the default loader.

    Output:
      Predicted letter as a Python str (e.g. "A"), or None if
      normalize_landmarks() returns None (degenerate / zero-scale hand).

    Pipeline:
      21 landmarks → normalize_landmarks() → (63,) vector → model.predict
      → letter string. No LabelEncoder; predict() already returns letters.
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

    # Shape (1, 63) with training column names when the estimator stored them.
    if hasattr(estimator, "feature_names_in_"):
        import pandas as pd

        x = pd.DataFrame([features], columns=FEATURE_NAMES)
    else:
        x = features.reshape(1, NUM_FEATURES)

    prediction = estimator.predict(x)
    return str(prediction[0])
