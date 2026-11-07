# Real-Time Sign Language Translator 🧠💻

A Python-based computer vision and machine learning project designed to recognize hand signs and convert them into text in real time, making communication more accessible[cite: 3, 4, 7].

## Features
- **Hand Gesture Detection:** Utilizes Google's MediaPipe framework to extract 3D hand landmarks instantly[cite: 3, 4].
- **Custom Data Collection:** Easily record and build your own dataset for different signs/gestures.
- **Machine Learning Classification:** Trains a high-accuracy Random Forest classifier using Scikit-Learn[cite: 7].
- **Real-Time Translation:** Live webcam inference script that processes frames and displays translated text dynamically[cite: 3].

---

## Tech Stack
- **Python**[cite: 3, 4, 7]
- **OpenCV** (Video handling and UI overlay)[cite: 3, 4, 5]
- **MediaPipe** (Hand tracking and landmark extraction)[cite: 3, 4, 5]
- **Scikit-Learn** (Classification model training)[cite: 5, 7]
- **Joblib** (Model serialization)[cite: 3, 5, 7]

---

## Project Structure
```text
├── collect_data.py    # Script to capture and save hand landmark data
├── train_model.py     # Script to train the ML classifier[cite: 7]
├── app.py             # Main real-time translation application[cite: 3]
├── requirements.txt   # Project dependencies
└── .gitignore         # Excludes virtual environments, datasets, and caches[cite: 2]
