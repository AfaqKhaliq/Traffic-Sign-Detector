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

It's recommended to structure your project like this:

Traffic-Sign-Detector/
│
├── yolov5/                 # Cloned YOLOv5 repository
│   ├── best.pt             # Trained weights file (you need to add this)
│   ├── webcam_detect.py    # Webcam inference script (you need to add this)
│   ├── requirements.txt    # YOLOv5 dependencies
│   └── ...                 # Other YOLOv5 files and folders
│
└── README.md               # This file

*(Note: You will clone the `yolov5` repository and then add your `best.pt` and `webcam_detect.py` files inside it.)*

---

## 🔽 Download Trained Model

Download the trained model weights file (`best.pt`) from the Google Drive link below:

📥 **[Download best.pt](https://drive.google.com/drive/folders/1FAYX2EP78fVPvhz_D7nkGnJx7zUq53h3?usp=drive_link)**

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8 or later
-   Pip package manager
-   A webcam connected to your computer

### Setup Instructions

1.  **Clone the YOLOv5 Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone [https://github.com/ultralytics/yolov5.git](https://github.com/ultralytics/yolov5.git)
    ```

2.  **Navigate into the Directory:**
    ```bash
    cd yolov5
    ```

3.  **Install Dependencies:**
    Install the required Python packages using the `requirements.txt` file provided by YOLOv5, plus `opencv-python` for webcam access:
    ```bash
    pip install -r requirements.txt
    pip install opencv-python
    ```
    *(Optional but recommended: Consider using a Python virtual environment)*

4.  **Add Your Trained Model:**
    Place the `best.pt` file (which you downloaded earlier) directly inside the `yolov5` directory you just cloned.

5.  **Add the Detection Script:**
    Create a Python file named `webcam_detect.py` inside the same `yolov5` directory. Paste the Python code for your webcam detection logic into this file.

    *(Make sure your `webcam_detect.py` script correctly loads the `best.pt` model and uses OpenCV to capture and process the webcam feed.)*

---

## 🧪 Run Real-Time Detection

Ensure you are still inside the `yolov5` directory in your terminal. Run the detection script using:

```bash
python webcam_detect.py










