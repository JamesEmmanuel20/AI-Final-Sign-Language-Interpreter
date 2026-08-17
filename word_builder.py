"""Stabilize letter predictions and build words for the live ASL demo."""

from __future__ import annotations

from collections import Counter, deque

PREDICTION_BUFFER_SIZE = 10
# Consecutive no-hand frames before a repeated letter (OO in BOOK) can be added.
NO_HAND_RESET_FRAMES = 15


def majority_letter(buffer: deque[str]) -> str | None:
    """Most common letter in the buffer, or None if the buffer is empty."""
    if not buffer:
        return None
    return Counter(buffer).most_common(1)[0][0]


def committed_letter(buffer: deque[str]) -> str | None:
    if len(buffer) != PREDICTION_BUFFER_SIZE:
        return None

    counts = Counter(buffer)
    letter, votes = counts.most_common(1)[0]

    if votes >= 9:
        return letter

    return None

class WordBuilder:
    def __init__(self) -> None:
        self.predictions: deque[str] = deque(maxlen=PREDICTION_BUFFER_SIZE)
        self.current_text = ""
        self.last_added_letter: str | None = None
        self.no_hand_frames = 0

    def on_no_hand(self) -> str:
        # Show B → B, lower/change hand, show O → BO, and so on
        # (the gap is what lets OO in BOOK be two letters, not one).
        self.no_hand_frames += 1
        self.predictions.clear()
        if self.no_hand_frames >= NO_HAND_RESET_FRAMES:
            self.last_added_letter = None
        return "No hand"

    def on_letter(self, letter: str | None) -> str:
        self.no_hand_frames = 0
        if letter is not None:
            self.predictions.append(letter)

        stable = majority_letter(self.predictions)
        # Unanimous full buffer: hold B, then change/lower the hand
        # before the next letter, including a second O for BOOK.
        to_add = committed_letter(self.predictions)
        if to_add is not None and to_add != self.last_added_letter:
            self.current_text += to_add
            self.last_added_letter = to_add
        return stable if stable is not None else "No hand"

    def clear(self) -> None:
        self.current_text = ""
        self.last_added_letter = majority_letter(self.predictions)

    def backspace(self) -> None:
        if self.current_text:
            self.current_text = self.current_text[:-1]
        # Avoid instantly re-adding the letter still on camera.
        self.last_added_letter = majority_letter(self.predictions)

    def add_space(self) -> None:
        self.current_text += " "
        self.last_added_letter = majority_letter(self.predictions)
