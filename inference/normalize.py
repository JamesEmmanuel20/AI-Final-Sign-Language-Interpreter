"""
Wrist-relative landmark normalization matching asl_normalizer.py.

Training used:
  translated = coords - wrist
  normalized = translated / max(||translated||)
  flattened as x0,y0,z0,...,x20,y20,z20
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NUM_LANDMARKS = 21
NUM_COORDS = 3
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63

FEATURE_NAMES = [
    name
    for i in range(NUM_LANDMARKS)
    for name in (f"x{i}", f"y{i}", f"z{i}")
]


def _landmark_to_xyz(landmark: Any) -> tuple[float, float, float]:
    """Extract (x, y, z) from a MediaPipe-like object or a 3-length sequence."""
    if hasattr(landmark, "x") and hasattr(landmark, "y") and hasattr(landmark, "z"):
        return float(landmark.x), float(landmark.y), float(landmark.z)

    if isinstance(landmark, (Sequence, np.ndarray)) and len(landmark) >= 3:
        return float(landmark[0]), float(landmark[1]), float(landmark[2])

    raise TypeError(
        "Each landmark must have x/y/z attributes or be a sequence of at least 3 numbers."
    )


def landmarks_to_coords(landmarks: Sequence[Any]) -> np.ndarray:
    """
    Convert 21 landmarks into a float array of shape (21, 3).

    Input:  exactly 21 landmarks (MediaPipe landmark objects or (x, y, z) values)
    Output: numpy array shape (21, 3), row i = landmark i = [x, y, z]
    """
    if len(landmarks) != NUM_LANDMARKS:
        raise ValueError(
            f"Expected exactly {NUM_LANDMARKS} landmarks, got {len(landmarks)}."
        )

    coords = np.array(
        [_landmark_to_xyz(landmark) for landmark in landmarks],
        dtype=float,
    )
    if coords.shape != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(
            f"Expected coords shape {(NUM_LANDMARKS, NUM_COORDS)}, got {coords.shape}."
        )
    return coords


def normalize_landmarks(landmarks: Sequence[Any]) -> np.ndarray | None:
    """
    Apply the same wrist translation + max-distance scale as asl_normalizer.py.

    Problem this solves:
      Raw MediaPipe coordinates depend on hand position in the frame and
      distance from the camera. Training removed both effects; live inference
      must do the same before calling the Random Forest.

    Input:
      Exactly 21 landmarks, each with x, y, z (MediaPipe objects or sequences).
      Landmark 0 is treated as the wrist.

    Output:
      - numpy array of shape (63,) in order x0,y0,z0,...,x20,y20,z20, or
      - None if the hand is degenerate (max wrist distance == 0), matching
        the training script's skip of invalid rows (avoids NaN/Inf).

    Why this matches training:
      Identical math to asl_normalizer.py:
        wrist = coords[0]
        translated = coords - wrist
        distances = ||translated|| per landmark
        normalized = translated / max(distances)
      then interleaved flatten. No LabelEncoder or extra features.
    """
    coords = landmarks_to_coords(landmarks)

    # Wrist = landmark 0 (same as asl_normalizer.py)
    wrist = coords[0].copy()
    translated = coords - wrist

    distances = np.linalg.norm(translated, axis=1)
    max_distance = float(distances.max())

    # Training skipped rows when max_distance == 0; do not divide.
    if max_distance == 0.0:
        return None

    normalized = translated / max_distance

    # Flatten in interleaved training order: x0,y0,z0,...,x20,y20,z20
    return normalized.reshape(NUM_FEATURES)
