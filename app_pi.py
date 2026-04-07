import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
from flask_cors import CORS
import threading
import subprocess
import time
import shutil
import os
import re
import sys
from picamera2 import Picamera2

print("🚀 Starting AI Surveillance (Fixed Pipeline)...")

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TARGET_FPS = 12.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_RES = 640  # Model expects 640x640
DETECTION_THRESHOLD = 0.45

# --- Global State ---
frame_lock = threading.Lock()
raw_frame_rgb = None  # Store original RGB from camera
latest_detections = []
latest_has_fire = False
latest_jpeg = None

# --- AI Setup ---
try:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    session = ort.InferenceSession("fire_detector.onnx", sess_options=so, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"❌ Model Error: {e}")
    sys.exit(1)

# --- AI Worker ---
def ai_worker():
    global raw_frame_rgb, latest_detections, latest_has_fire

    while True:
        with frame_lock:
            # AI needs RGB
            img_in = raw_frame_rgb.copy() if raw_frame_rgb is not None else None

        if img_in is None:
            time.sleep(0.01)
            continue

        # 1. Resize to model input (640x640)
        img = cv2.resize(img_in, (MODEL_RES, MODEL_RES))
        
        # 2. Preprocess
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)   # Add batch dim

        outputs = session.run(None, {input_name: img})
        out = outputs[0][0]

        if out.shape[0] < out.shape[1]:
            out = out.T

        temp_detections = []
        has_fire = False

        for row in out:
            conf = np.max(row[4:])
            if conf > DETECTION_THRESHOLD:
                # Model returns coordinates relative to 640x640
                cx, cy, w, h = row[0:4]

                # Scale back to 640x480 (Camera Res)
                # x_scale = 640/640, y_scale = 480/640
                x = int((cx - w / 2)) 
                y = int((cy - h / 2) * (FRAME_HEIGHT / MODEL_RES))
                bw = int(w)
                bh = int(h * (FRAME_HEIGHT / MODEL_RES))

                temp_detections.append({"box": [x, y, bw, bh], "conf": float(conf)})
                has_fire = True

        # Non-Maximum Suppression to filter overlapping boxes
        if has_fire:
            boxes = [d["box"] for d in temp_detections]
            confs = [d["conf"] for d in temp_detections]
            indices = cv2.dnn.NMSBoxes(boxes, confs, DETECTION_THRESHOLD, 0.45)
            
            final_dets = []
            if len(indices) > 0:
                for i in indices.flatten():
                    final_dets.append(temp_detections[i])
            
            with frame_lock:
                latest_detections = final_dets
                latest_has_fire = len(final_dets) > 0
        else:
            with frame_lock:
                latest_has_fire = False
                latest_detections = []

        time.sleep(0.01)

# --- Camera Worker ---
def camera_worker():
    global raw_frame_rgb, latest_jpeg

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("🎥 Picamera2 Active.")

    while True:
        start_time = time.time()
        frame_rgb = picam2.capture_array()[:, :, :3]

        with frame_lock:
            raw_frame_rgb = frame_rgb.copy()
            current_detections = latest_detections
            current_has_fire = latest_has_fire

        # For display, convert RGB to BGR (OpenCV standard)
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if current_has_fire:
            for det in current_detections:
                x, y, w, h = det["box"]
                conf = det["conf"]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_frame, f"FIRE {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

        # Simple FPS lock
        elapsed = time.time() - start_time
        time.sleep(max(0, (1.0 / TARGET_FPS) - elapsed))

# --- Flask & Cloudflare ---
def start_cloudflare_background(port=5000):
    def run():
        path = shutil.which("cloudflared")
        if not path: return
        process = subprocess.Popen([path, "tunnel", "--url", f"http://localhost:{port}"], 
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            if "trycloudflare.com" in line:
                match = re.search(r"https?://[^\s']+", line)
                if match: print(f"\n✅ PUBLIC URL: {match.group(0)}\n")
    threading.Thread(target=run, daemon=True).start()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                jpeg = latest_jpeg
            if jpeg:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<body style="background:#000; display:flex; justify-content:center;"><img src="/video_feed" style="height:90vh;"></body>'

if __name__ == '__main__':
    start_cloudflare_background(5000)
    threading.Thread(target=ai_worker, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
