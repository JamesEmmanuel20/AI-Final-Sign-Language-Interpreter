"""OpenCV webcam capture for the live ASL demo."""

from __future__ import annotations
import cv2
import numpy as np


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default webcam.")
    return cap


def read_bgr(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def release_camera(cap: cv2.VideoCapture) -> None:
    cap.release()
