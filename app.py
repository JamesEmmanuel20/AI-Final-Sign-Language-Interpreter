import sys
import cv2

from collections import Counter
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hands import create_landmarker, detect_hand, draw_hand_landmarks
from inference import load_model, predict_letter
from word_builder import WordBuilder




class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    prediction_ready = Signal(str)
    hand_status_changed = Signal(bool)
    camera_started = Signal()
    camera_stopped = Signal()
    camera_error = Signal()

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = False
        self.camera = None

    def run(self):
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.camera_error.emit()
            return

        self.running = True
        self.camera_started.emit()

        try:
            with create_landmarker() as landmarker:
                timestamp_ms = 0

                while self.running:
                    success, frame = self.camera.read()

                    if not success:
                        continue

                    display_frame = frame.copy()

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    landmarks, timestamp_ms = detect_hand(
                        landmarker,
                        rgb,
                        timestamp_ms,
                    )

                    if landmarks is None:
                        self.hand_status_changed.emit(False)
                    else:
                        self.hand_status_changed.emit(True)
                        draw_hand_landmarks(display_frame, landmarks)

                        letter = predict_letter(
                            landmarks,
                            model=self.model,
                        )

                        if letter is not None:
                            self.prediction_ready.emit(letter)

                    display_frame = cv2.cvtColor(
                        display_frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    height, width, channels = display_frame.shape
                    bytes_per_line = channels * width

                    image = QImage(
                        display_frame.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGB888,
                    ).copy()

                    self.frame_ready.emit(image)
                    self.msleep(30)

        finally:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

            self.camera_stopped.emit()

    def stop(self):
        self.running = False


class ASLInterpreter(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASL Sign Language Interpreter")
        self.setMinimumSize(1200, 750)

        self.camera_worker = None
        self.model = load_model()
        self.word_builder = WordBuilder()

        self.build_ui()
        self.apply_styles()

        self.stop_button.setEnabled(False)

    # Build the interface
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(28, 20, 28, 20)
        main_layout.setSpacing(18)

        # Header
        header = QHBoxLayout()
        header.setSpacing(20)

        title_section = QVBoxLayout()
        title_section.setSpacing(3)

        title = QLabel("ASL Sign Language Interpreter")
        title.setObjectName("title")

        subtitle = QLabel("Learn  •  Communicate  •  Include")
        subtitle.setObjectName("subtitle")

        title_section.addWidget(title)
        title_section.addWidget(subtitle)

        header.addLayout(title_section)
        header.addStretch()

        self.camera_status = QLabel("●  Camera: OFF")
        self.camera_status.setObjectName("cameraStatus")

        header.addWidget(self.camera_status)

        main_layout.addLayout(header)

        # Main content
        content = QHBoxLayout()
        content.setSpacing(20)

        # Camera panel
        camera_panel = QFrame()
        camera_panel.setObjectName("panel")

        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(18, 16, 18, 16)
        camera_layout.setSpacing(12)

        camera_title = QLabel("▣  Live Camera")
        camera_title.setObjectName("panelTitle")

        camera_layout.addWidget(camera_title)

        self.camera_display = QLabel("Camera Preview")
        self.camera_display.setObjectName("cameraDisplay")
        self.camera_display.setAlignment(Qt.AlignCenter)
        self.camera_display.setMinimumSize(600, 420)
        self.camera_display.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        camera_layout.addWidget(self.camera_display, 1)

        # Detected letter
        self.detected_badge = QLabel("Detected:\n—", self.camera_display)
        self.detected_badge.setObjectName("detectedBadge")
        self.detected_badge.setAlignment(Qt.AlignCenter)
        self.detected_badge.setFixedSize(150, 92)

        self.detected_letter = self.detected_badge
        self.position_detected_badge()

        content.addWidget(camera_panel, 3)

        # Right side
        right_layout = QVBoxLayout()
        right_layout.setSpacing(14)

        # Translation
        translation_panel = QFrame()
        translation_panel.setObjectName("panel")

        translation_layout = QVBoxLayout(translation_panel)
        translation_layout.setContentsMargins(18, 16, 18, 16)
        translation_layout.setSpacing(12)

        translation_title = QLabel("▤  Translation")
        translation_title.setObjectName("panelTitle")

        translation_layout.addWidget(translation_title)

        self.translation_display = QLabel("No translation yet")
        self.translation_display.setObjectName("translationDisplay")
        self.translation_display.setAlignment(Qt.AlignCenter)
        self.translation_display.setWordWrap(True)

        translation_layout.addWidget(self.translation_display, 1)

        right_layout.addWidget(translation_panel, 1)

        # Buttons
        buttons = QGridLayout()
        buttons.setSpacing(12)

        self.start_button = QPushButton("▶  Start Camera")
        self.start_button.setObjectName("startButton")

        self.stop_button = QPushButton("■  Stop Camera")
        self.stop_button.setObjectName("stopButton")

        self.clear_button = QPushButton("▣  Clear")
        self.clear_button.setObjectName("secondaryButton")

        self.backspace_button = QPushButton("⌫  Backspace")
        self.backspace_button.setObjectName("secondaryButton")

        self.space_button = QPushButton("↔  Add Space")
        self.space_button.setObjectName("secondaryButton")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.clear_button.clicked.connect(self.clear_translation)
        self.backspace_button.clicked.connect(self.backspace_translation)
        self.space_button.clicked.connect(self.add_space)

        buttons.addWidget(self.start_button, 0, 0)
        buttons.addWidget(self.stop_button, 0, 1)
        buttons.addWidget(self.clear_button, 1, 0)
        buttons.addWidget(self.backspace_button, 1, 1)
        buttons.addWidget(self.space_button, 2, 0, 1, 2)

        right_layout.addLayout(buttons)

        # Help
        help_panel = QFrame()
        help_panel.setObjectName("helpPanel")

        help_layout = QVBoxLayout(help_panel)
        help_layout.setContentsMargins(16, 14, 16, 14)
        help_layout.setSpacing(6)

        help_title = QLabel("ⓘ  How to use")
        help_title.setObjectName("helpTitle")

        help_text = QLabel(
            "Start the camera and place your hand inside the frame.\n"
            "Hold an ASL sign until it is detected.\n"
            "Use the controls to edit your translation."
        )
        help_text.setObjectName("helpText")
        help_text.setWordWrap(True)

        help_layout.addWidget(help_title)
        help_layout.addWidget(help_text)

        right_layout.addWidget(help_panel)

        content.addLayout(right_layout, 2)

        main_layout.addLayout(content, 1)

        # Status bar
        status_panel = QFrame()
        status_panel.setObjectName("statusPanel")

        status_layout = QHBoxLayout(status_panel)
        status_layout.setContentsMargins(18, 12, 18, 12)

        model_status = QLabel("✦  Model: Random Forest")
        model_status.setObjectName("statusText")

        self.hand_status = QLabel("✋  No hand detected")
        self.hand_status.setObjectName("handStatus")

        status_layout.addWidget(model_status)
        status_layout.addStretch()
        status_layout.addWidget(self.hand_status)

        main_layout.addWidget(status_panel)

    # Position the detected badge inside the camera preview
    def position_detected_badge(self):
        if hasattr(self, "detected_badge"):
            margin = 16
            x = self.camera_display.width() - self.detected_badge.width() - margin
            y = margin
            self.detected_badge.move(max(0, x), y)
            self.detected_badge.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_detected_badge()

    # Start the camera
    def start_camera(self):
        if self.camera_worker is not None:
            return

        self.camera_status.setText("●  Camera: STARTING...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.camera_worker = CameraWorker(self.model)

        self.camera_worker.frame_ready.connect(self.display_frame)
        self.camera_worker.prediction_ready.connect(self.update_prediction)
        self.camera_worker.hand_status_changed.connect(self.update_hand_status)
        self.camera_worker.camera_started.connect(self.camera_started)
        self.camera_worker.camera_stopped.connect(self.camera_stopped)
        self.camera_worker.camera_error.connect(self.camera_error)

        self.camera_worker.start()

    # Display a camera frame
    @Slot(QImage)
    def display_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.camera_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.camera_display.setPixmap(pixmap)

    # Update the UI when the camera starts
    @Slot()
    def camera_started(self):
        self.camera_status.setText("●  Camera: ON")

    # Update the detected letter
    @Slot(str)
    def update_prediction(self, letter):
        self.word_builder.on_letter(letter)

        if self.word_builder.predictions:
            counts = Counter(self.word_builder.predictions)
            detected = counts.most_common(1)[0][0]
            self.detected_letter.setText(detected)

        text = self.word_builder.current_text

        self.translation_display.setText(
            text if text else "No translation yet"
        )

    # Update the hand status
    @Slot(bool)
    def update_hand_status(self, detected):
        if detected:
            self.hand_status.setText("✋  Hand detected")
        else:
            self.hand_status.setText("✋  No hand detected")
            self.detected_letter.setText("Detected:\n—")
            self.word_builder.on_no_hand()

    # Stop the camera
    def stop_camera(self):
        if self.camera_worker is None:
            return

        self.camera_status.setText("●  Camera: STOPPING...")
        self.stop_button.setEnabled(False)
        self.camera_worker.stop()

    # Update the UI after the camera stops
    @Slot()
    def camera_stopped(self):
        if self.camera_worker is not None:
            self.camera_worker.deleteLater()
            self.camera_worker = None

        self.camera_display.clear()
        self.camera_display.setText("Camera Preview")
        self.camera_status.setText("●  Camera: OFF")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.detected_letter.setText("Detected:\n—")
        self.hand_status.setText("✋  No hand detected")

    # Handle camera errors
    @Slot()
    def camera_error(self):
        if self.camera_worker is not None:
            self.camera_worker.deleteLater()
            self.camera_worker = None

        self.camera_status.setText("●  Camera: ERROR")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    # Clear the translation
    def clear_translation(self):
        self.word_builder.clear()
        self.translation_display.setText("No translation yet")

    # Remove the last letter
    def backspace_translation(self):
        self.word_builder.backspace()
        text = self.word_builder.current_text
        self.translation_display.setText(
            text if text else "No translation yet"
        )

    # Add a space
    def add_space(self):
        self.word_builder.add_space()
        text = self.word_builder.current_text
        self.translation_display.setText(
            text if text else "No translation yet"
        )

    # Close the application
    def closeEvent(self, event):
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.wait()
            self.camera_worker = None

        event.accept()

    # Apply the application styling
    def apply_styles(self):
        self.setStyleSheet("""
            /* Main window */

            QMainWindow {
                background-color: #0d0d16;
            }

            QWidget {
                color: #f5f5f7;
                font-family: "Segoe UI";
            }

            /* Header */

            #title {
                font-size: 32px;
                font-weight: 700;
                color: #f5f5f7;
            }

            #subtitle {
                font-size: 15px;
                color: #9997ad;
            }

            #cameraStatus {
                color: #4ade80;
                font-size: 15px;
                font-weight: 600;
            }

            /* Panels */

            #panel {
                background-color: #181722;
                border: 1px solid #302e42;
                border-radius: 16px;
            }

            #panelTitle {
                font-size: 18px;
                font-weight: 650;
                color: #f3f3f7;
            }

            /* Camera */

            #cameraDisplay {
                background-color: #09090f;
                border: 1px solid #302e42;
                border-radius: 12px;
                color: #77758c;
                font-size: 22px;
            }

            /* Detected letter */

            #detectedBadge {
                background-color: #39d982;
                color: white;
                border-radius: 14px;
                font-size: 20px;
                font-weight: 700;
                padding: 8px;
            }

            /* Translation */

            #translationDisplay {
                background-color: #11111a;
                border: 1px solid #302e42;
                border-radius: 12px;
                padding: 25px;
                font-size: 32px;
                font-weight: 600;
                color: white;
            }

            /* Buttons */

            QPushButton {
                min-height: 50px;
                border-radius: 11px;
                font-size: 15px;
                font-weight: 600;
            }

            #startButton {
                background-color: #7042f5;
                border: none;
                color: white;
            }

            #startButton:hover {
                background-color: #8055ff;
            }

            #startButton:disabled {
                background-color: #413568;
                color: #aaa5bb;
            }

            #stopButton {
                background-color: #df4c5c;
                border: none;
                color: white;
            }

            #stopButton:hover {
                background-color: #ec5d6c;
            }

            #stopButton:disabled {
                background-color: #49252b;
                color: #8e7276;
            }

            #secondaryButton {
                background-color: #252331;
                border: 1px solid #3b384d;
                color: #f1f0f5;
            }

            #secondaryButton:hover {
                background-color: #302d40;
            }

            /* Help */

            #helpPanel {
                background-color: #211d35;
                border: 1px solid #503c82;
                border-radius: 14px;
            }

            #helpTitle {
                color: #a987ff;
                font-size: 16px;
                font-weight: 700;
            }

            #helpText {
                color: #b9b6c8;
                font-size: 14px;
            }

            /* Status */

            #statusPanel {
                background-color: #181722;
                border: 1px solid #302e42;
                border-radius: 13px;
            }

            #statusText {
                color: #d2d0da;
                font-size: 15px;
                font-weight: 600;
            }

            #handStatus {
                color: #39df86;
                font-size: 15px;
                font-weight: 600;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ASLInterpreter()
    window.show()

    sys.exit(app.exec())