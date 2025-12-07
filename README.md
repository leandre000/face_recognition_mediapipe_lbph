# Face Recognition Pipeline (MediaPipe + LBPH)

A classical face recognition system combining **MediaPipe Face Mesh** for detection and **OpenCV LBPH** (Local Binary Patterns Histograms) for recognition.

## 📂 Project Structure

```
face-det/
├── capture.py          # Capture face images from webcam
├── train.py            # Train LBPH model on captured faces
├── predict.py          # Recognize faces in real-time
├── dataset/            # Captured face images (auto-created)
│   └── <name>/
├── models/             # Trained models (auto-created)
│   ├── lbph_model.xml
│   └── label_map.json
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Webcam

### Install Dependencies

```bash
pip install opencv-python mediapipe opencv-contrib-python numpy<2
```

## 🚀 Quick Start

### Step 1: Capture Face Images (2 People)

```bash
python capture.py
```

The script will ask for **Person 1's name**, then **Person 2's name**.

For each person:
- Face yourself toward the camera
- Press **SPACE** to capture frames (capture 20–30 images per person)
- Press **Q** to finish and move to the next person

Images are saved to:
- `dataset/<person1_name>/`
- `dataset/<person2_name>/`

### Step 2: Train the Model

```bash
python train.py
```

This trains the LBPH recognizer and saves:
- `models/lbph_model.xml` - Trained model
- `models/label_map.json` - Name mappings

### Step 3: Run Face Recognition

```bash
python predict.py
```

The camera window displays:
- **Green rectangle** around detected faces
- **Predicted name**
- **Confidence score**

Press **Q** to exit.

## 📖 How It Works

1. **Detection**: MediaPipe Face Mesh detects facial landmarks
2. **Extraction**: Face regions are cropped from the frame
3. **Training**: LBPH algorithm learns facial patterns per person
4. **Recognition**: Predicts identity based on learned patterns

## 🎯 Features

- ✅ Multi-person support
- ✅ Real-time recognition
- ✅ No external ML models required
- ✅ Easy to use and extend

## 💡 Tips

- Capture 20–30 images **per person** for best results
- Ensure good lighting when capturing faces
- Vary face angles and distances slightly
- After adding new people, retrain the model with `python train.py`
- Press **Q** to quit the recognition script
