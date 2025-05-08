from flask import Flask, render_template, request, redirect, url_for, send_file, Response
import torch
import pathlib
import sys
import os
import cv2
import numpy as np
from PIL import Image

# Fix PosixPath for Windows
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt', force_reload=True)
print("Model classes:", model.names)  # Debug: Check loaded classes

from PIL import Image as PILImage
import io
def allowed_file(filename):
    """Check if the uploaded file is a valid image (PNG/JPG/JPEG)."""
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
def is_valid_image(file_bytes):
    """Check if the uploaded file is a valid image (not just by extension)."""
    try:
        img = PILImage.open(io.BytesIO(file_bytes))
        img.verify()  # Verify if it's a real image
        return True
    except:
        return False

def process_image(filepath):
    """Process an image (PNG/JPG) and return detection results."""
    try:
        # Read file as bytes first to validate
        with open(filepath, 'rb') as f:
            file_bytes = f.read()
        
        if not is_valid_image(file_bytes):
            return None, "Uploaded file is not a valid image."

        # Now load with OpenCV
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "Failed to decode image (invalid format)."

        # Rest of the detection logic...
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        predictions = results.pandas().xyxy[0]

        conf_threshold = 0.2
        filtered = predictions[predictions['confidence'] > conf_threshold]

        detected_classes = ", ".join(filtered['name'].unique()) if len(filtered) > 0 else "No traffic sign detected."

        # Save detection result
        results.render()
        detected_img = Image.fromarray(results.ims[0])
        detected_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(filepath))
        detected_img.save(detected_path)

        return detected_classes, detected_path

    except Exception as e:
        return None, f"Error processing image: {str(e)}"
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect
        results = model(frame)
        results.render()
        frame = results.ims[0]  # rendered frame

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()    

@app.route('/', methods=['GET', 'POST'])
def index():
    detected_class = None
    input_image = None
    detected_image = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Check if the file is a real image
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
            
            if not is_valid_image(file_bytes):
                detected_class = "Error: Uploaded file is not a valid image."
                return render_template('index.html', detected_class=detected_class)

            # Process the image
            detected_class, detected_img_path = process_image(filepath)

            if detected_img_path:
                detected_image = os.path.basename(detected_img_path)
            else:
                detected_image = None

            return render_template(
                'index.html',
                detected_class=detected_class,
                uploaded=True,
                input_image=filename,
                detected_image=detected_image
            )
        else:
            detected_class = "Invalid file format. Please upload a PNG or JPG image."

    return render_template('index.html')
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)