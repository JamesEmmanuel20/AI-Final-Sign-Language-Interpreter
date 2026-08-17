"""Live webcam ASL letter demo using the existing inference package."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from camera import bgr_to_rgb, open_camera, read_bgr, release_camera
from hands import create_landmarker, detect_hand, draw_hand_landmarks
from inference import load_model, predict_letter
from ui import destroy_windows, overlay, show_and_read_key
from word_builder import WordBuilder


def main() -> None:
    model = load_model()
    words = WordBuilder()
    cap = open_camera()
    timestamp_ms = 0
    try:
        with create_landmarker() as landmarker:
            while True:
                frame = read_bgr(cap)
                if frame is None:
                    break
                rgb = bgr_to_rgb(frame)
                landmarks, timestamp_ms = detect_hand(landmarker, rgb, timestamp_ms)
                if landmarks is None:
                    letter_display = words.on_no_hand()
                else:
                    draw_hand_landmarks(frame, landmarks)
                    letter = predict_letter(landmarks, model=model)
                    letter_display = words.on_letter(letter)
                overlay(frame, letter_display, words.current_text)
                key = show_and_read_key(frame)
                if key == "quit":
                    break
                if key == "clear":
                    words.clear()
                elif key == "backspace":
                    words.backspace()
                elif key == "space":
                    words.add_space()
    finally:
        release_camera(cap)
        destroy_windows()


if __name__ == "__main__":
    main()
