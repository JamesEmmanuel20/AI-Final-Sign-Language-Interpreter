"""MediaPipe HandLandmarker setup and detection for the live ASL demo."""

from __future__ import annotations
import urllib.request
from pathlib import Path
from typing import Any
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

_ROOT = Path(__file__).resolve().parent
_HAND_LANDMARKER_MODEL = _ROOT / "hand_landmarker.task"
_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_hand_landmarker_model() -> Path:
    if _HAND_LANDMARKER_MODEL.is_file():
        return _HAND_LANDMARKER_MODEL
    urllib.request.urlretrieve(_HAND_LANDMARKER_URL, _HAND_LANDMARKER_MODEL)
    return _HAND_LANDMARKER_MODEL


def create_landmarker():
    landmarker_model_path = _ensure_hand_landmarker_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hand(
    landmarker: Any,
    rgb: np.ndarray,
    timestamp_ms: int,
) -> tuple[Any | None, int]:
    timestamp_ms += 33
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None, timestamp_ms
    return result.hand_landmarks[0], timestamp_ms


def draw_hand_landmarks(frame: np.ndarray, landmarks: Any) -> None:
    drawing_utils.draw_landmarks(
        frame,
        landmarks,
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
        drawing_styles.get_default_hand_landmarks_style(),
        drawing_styles.get_default_hand_connections_style(),
    )
