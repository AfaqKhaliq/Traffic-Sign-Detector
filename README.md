# 🚦 Traffic-Sign-Detector

A real-time **Traffic Sign Detection** system using **YOLOv5** and your **laptop webcam**.

This project demonstrates how to load a custom-trained YOLOv5 model and run inference on a live webcam feed to detect traffic signs in real-time.

---

## 📦 Features

- ✅ Real-time object detection using webcam
- 🚗 Trained on a custom traffic sign dataset
- 🧠 Uses YOLOv5 for fast and accurate detection
- 💻 Easy to run locally with minimal setup

---

## 📁 Repository Structure

Traffic-Sign-Detector/
│
├── yolov5/ # Cloned YOLOv5 repository
│ ├── best.pt # Trained weights file (you need to add this)
│ ├── requirements.txt # YOLOv5 dependencies
│ └── ... # Other YOLOv5 files and folders
│
├── app.py # Webcam detection script (already included)
└── README.md # This file

yaml
Copy
Edit

---

## 🔽 Download Trained Model

Download the trained model weights file (`best.pt`) from the Google Drive link below:

📥 **[Download best.pt](https://drive.google.com/drive/folders/1FAYX2EP78fVPvhz_D7nkGnJx7zUq53h3?usp=drive_link)**  
**Dataset Used**: [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or later
- Pip package manager
- A webcam connected to your computer

### Setup Instructions

1. **Clone the YOLOv5 Repository:**

   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   ```

2. **Navigate into the YOLOv5 Directory:**

   ```bash
   cd yolov5
   ```

3. **Install Dependencies:**
   Install the required Python packages using the `requirements.txt` file provided by YOLOv5, plus `opencv-python` for webcam access:

   ```bash
   pip install -r requirements.txt
   pip install opencv-python
   ```

4. **Add Your Trained Model:**
   Place the `best.pt` file (downloaded earlier) inside the `yolov5` directory.

---

## 🧪 Run Real-Time Detection

Go back to the root directory (where `app.py` is located), and run the detection script:

```bash
cd ..
python app.py
```
