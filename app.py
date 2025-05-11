from flask import Flask, render_template, request, redirect, url_for, send_file, Response
import torch
import pathlib
import sys
import os
import cv2
import numpy as np
from PIL import Image, Image as PILImage
import io

# Fix PosixPath for Windows
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLO models
model_og = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/bestog.pt', force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt', force_reload=True)

# Get class name mappings
old_names = model_og.names
new_names = model.names

# Create remapping dictionary
remap = {new_idx: old_names.index(name) for new_idx, name in new_names.items() if name in old_names}
model.names = old_names  # for consistent display

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def is_valid_image(file_bytes):
    try:
        img = PILImage.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except:
        return False

def process_image(filepath):
    try:
        with open(filepath, 'rb') as f:
            file_bytes = f.read()

        if not is_valid_image(file_bytes):
            return None, "Uploaded file is not a valid image."

        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "Failed to decode image (invalid format)."

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        predictions = results.pandas().xyxy[0]

        conf_threshold = 0.25
        filtered = predictions[predictions['confidence'] > conf_threshold]

        detected_indices = [remap.get(int(cls), int(cls)) for cls in filtered['class']]
        unique_indices = list(set(detected_indices))  # remove duplicates
        detected_classes = ", ".join([model.names[idx] for idx in unique_indices]) if unique_indices else "No traffic sign detected."


        results.render()
        detected_img = Image.fromarray(results.ims[0])
        detected_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(filepath))
        detected_img.save(detected_path)

        return detected_classes, detected_path

    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def get_available_camera_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def gen_frames():
    camera_index = get_available_camera_index()
    if camera_index is None:
        print("❌ No available camera found.")
        # Yield an error image or blank screen with message
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Webcam not found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    cap = cv2.VideoCapture(camera_index)
    print(f"✅ Webcam opened at index {camera_index}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect and render
        results = model(frame)
        results.render()
        frame = results.ims[0]  # rendered frame (NumPy array)

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

            with open(filepath, 'rb') as f:
                file_bytes = f.read()

            if not is_valid_image(file_bytes):
                detected_class = "Error: Uploaded file is not a valid image."
                return render_template('index.html', detected_class=detected_class)

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
