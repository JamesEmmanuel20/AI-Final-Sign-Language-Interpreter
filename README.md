# ASL Sign Language Interpreter
A real-time American Sign Language (ASL) letter recognition system that uses computer vision and machine learning to recognize hand signs from a webcam and build them into text.

## Overview
The ASL Sign Language Interpreter is an AI-based assistive application designed to recognize static ASL alphabet signs in real time.
The system uses MediaPipe Hand Landmarker to detect 21 hand landmarks from a webcam. These landmarks are normalized into 63 numerical features and passed to a Random Forest classifier, which predicts the corresponding ASL letter.
The predicted letters are then processed by a word-building component that stabilizes live predictions before adding them to the translation.

## Features
i. Real-time webcam-based hand detection
ii. Recognition of 24 ASL alphabet letters
iii. 21 hand landmarks extracted using MediaPipe
iv. 63 normalized features used for classification
v. Random Forest letter classification
vi. Prediction stabilization using a 10-frame buffer
vii. 9/10 prediction agreement before committing a letter
viii. Real-time translation display
ix. Clear translation
x. Backspace
xi. Add space
xii. Start and stop camera controls
xiii. PySide6 graphical user interface


## System Pipeline
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
Landmark Normalization
   ↓
63 Features
   ↓
Random Forest Classifier
   ↓
Predicted Letter
   ↓
Word Builder
   ↓
PySide6 Interface
   ↓
Translation

## Machine Learning Model
We use a Random Forest Classifier for ASL letter recognition.

The current dataset contains:
i. 1,846 samples
ii. 24 ASL letters
iii. 63 features per sample

The model was evaluated using stratified 5-fold cross-validation.

## Results
Metric                           Result

Average 5-Fold CV Accuracy         95.23%

Standard Deviation                 1.28%

Separate Test Accuracy              94.1%

While overall performance is strong, performance varies between individual letters. N, U, and V were among the weaker classes during evaluation.
The confusion matrix is used to identify class-specific misclassifications and guide future improvements.

## Prediction Stabilization
During live testing, individual frame predictions could occasionally fluctuate even when the user was holding the same sign.
To reduce this jumpiness, the word builder uses a 10-frame prediction buffer. A letter is committed only when 9 out of 10 predictions agree.
This provides a balance between prediction stability and real-time responsiveness.

## Technologies Used
i. Python
ii. MediaPipe
iii. OpenCV
iv. NumPy
v. Pandas
vi. Scikit-learn
vii. PySide6
viii. Git/GitHub

# Installation
## Clone the repository:

git clone https://github.com/JamesEmmanuel20/AI-Final-Sign-Language-Interpreter.git
cd AI-Final-Sign-Language-Interpreter

# Create and activate a virtual environment:
## Windows
python -m venv .venv
.venv\Scripts\Activate.ps1


## Install the dependencies:
pip install -r requirements.txt

# Running the Application
## Start the application with:
python app.py

Make sure your computer has a working webcam.
Click Start Camera, place your hand inside the camera frame, and hold an ASL letter until it is detected.

## Project Structure
AI-Final-Sign-Language-Interpreter/
│
├── app.py
├── camera.py
├── hands.py
├── ui.py
├── word_builder.py
│
├── inference/
│   ├── __init__.py
│   ├── model.py
│   ├── normalize.py
│   └── predict.py
│
├── normalized_dataset.csv
├── hand_landmarker.task
├── requirements.txt
└── README.md

# Limitations
The current system focuses on static ASL alphabet letters rather than complete ASL translation.
Other limitations include:

* Some letters are more difficult to classify than others.
* Performance can be affected by lighting and camera conditions.
* Different users may perform the same sign differently.
* The current dataset has limited diversity.
* Dynamic signs involving movement are not currently supported.
* The system should not be considered a complete ASL translator.
  
## Future Enhancements
Future development could include:

* Expanding and diversifying the training dataset
* Improving weaker classes such as N, U, and V
* Supporting dynamic ASL signs
* Recognizing complete words and phrases
* Improving real-time responsiveness
* Adding prediction confidence information
* Improving accessibility features
* Continuous evaluation across different users and environments
  
## Ethics and Fairness
The system was evaluated with consideration for bias, privacy, transparency, and human oversight.
The model's overall accuracy does not guarantee equal performance across all signs or users. The weaker classes identified during evaluation demonstrate the need for continued monitoring and more diverse training data.
The application is designed to process webcam information locally. Webcam data should not be unnecessarily stored or shared without appropriate user consent.
Users remain in control of the final translation through the application's editing controls, including Clear, Backspace, and Add Space.
