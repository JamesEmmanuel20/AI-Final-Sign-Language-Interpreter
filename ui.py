"""OpenCV overlay and keyboard handling for the live ASL demo."""

from __future__ import annotations

import cv2
import numpy as np


def overlay(frame: np.ndarray, letter_display: str, current_text: str) -> None:
    cv2.putText(
        frame,
        f"Letter: {letter_display}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0) if letter_display != "No hand" else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Text: {current_text}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def show_and_read_key(frame: np.ndarray) -> str:
    cv2.imshow("ASL Sign Language Demo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q")):
        return "quit"
    if key in (ord("c"), ord("C")):
        return "clear"
    if key in (8, 127):  # Backspace (Windows) / Delete
        return "backspace"
    if key == ord(" "):
        return "space"
    return "none"


def destroy_windows() -> None:
    cv2.destroyAllWindows()
