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


---

## 🧠 Training Details

- **Model**: YOLOv5s
- **Input Size**: 416x416
- **Epochs**: 26 (trained incrementally)
- **Total Images**: ~39,000 (custom dataset)
- **Environment**: Google Colab (T4 GPU)

Weights are saved to `best.pt`.

---

## 🔽 Download Trained Model

Download the trained model weights file (`best.pt`) from this Google Drive link:

📥 [Download best.pt](https://drive.google.com/drive/folders/1FAYX2EP78fVPvhz_D7nkGnJx7zUq53h3?usp=drive_link)

Place the file inside the `yolov5/` directory after cloning.

---

## 🚀 Getting Started

### 1. Clone YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
⚙️ Setup Instructions
2. Install Dependencies
Install the required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
pip install opencv-python
3. Add Your Model
Place the best.pt file inside the yolov5 directory.

4. Add the Detection Script
Inside the same yolov5 directory, create a file named webcam_detect.py. You can find the sample code for this script later in the README.

🧪 Run Real-Time Detection
Run the detection script using the command below:

bash
Copy
Edit
python webcam_detect.py
The webcam will open and the model will start detecting traffic signs in real time.

Let me know if you'd like me to regenerate the full README with this updated formatting!









