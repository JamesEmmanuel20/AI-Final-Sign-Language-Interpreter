"""Normalize 21 hand landmarks the same way as asl_normalizer.py."""

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
    """Pull (x, y, z) from a MediaPipe landmark or a length-3 sequence."""
    if hasattr(landmark, "x") and hasattr(landmark, "y") and hasattr(landmark, "z"):
        return float(landmark.x), float(landmark.y), float(landmark.z)

    if isinstance(landmark, (Sequence, np.ndarray)) and len(landmark) >= 3:
        return float(landmark[0]), float(landmark[1]), float(landmark[2])

    raise TypeError(
        "Each landmark must have x/y/z attributes or be a sequence of at least 3 numbers."
    )


def landmarks_to_coords(landmarks: Sequence[Any]) -> np.ndarray:
    """Convert exactly 21 landmarks to a float array of shape (21, 3)."""
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
    Wrist-translate and max-distance scale 21 landmarks (matches training).

    Returns a (63,) vector in order x0,y0,z0,...,x20,y20,z20, or None if
    max distance from the wrist is 0 (avoids NaN/Inf; training skipped those rows).
    """
    coords = landmarks_to_coords(landmarks)

    # Landmark 0 is the wrist — same origin as asl_normalizer.py
    wrist = coords[0].copy()
    translated = coords - wrist

    distances = np.linalg.norm(translated, axis=1)
    max_distance = float(distances.max())
    if max_distance == 0.0:
        return None

    normalized = translated / max_distance

    # Interleaved training order: x0,y0,z0,...,x20,y20,z20
    return normalized.reshape(NUM_FEATURES)
