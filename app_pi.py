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
import signal
import sys
import re
from picamera2 import Picamera2

print("🚀 Starting AI Surveillance (Fixed Picamera Pipeline)...")

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TARGET_FPS = 12.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECTION_THRESHOLD = 0.45

# --- Global State ---
frame_lock = threading.Lock()
raw_frame_for_ai = None
latest_detections = []
latest_has_fire = False
latest_jpeg = None

# --- AI Setup ---
providers = ['CPUExecutionProvider']
try:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    session = ort.InferenceSession("fire_detector.onnx", sess_options=so, providers=providers)
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"❌ Model Error: {e}")
    sys.exit(1)

# --- Cloudflare ---
def start_cloudflare_background(port=5000):
    def run():
        path = shutil.which("cloudflared")
        if not path:
            return
        process = subprocess.Popen(
            [path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            if "trycloudflare.com" in line:
                match = re.search(r"https?://[^\s']+", line)
                if match:
                    print(f"\n✅ PUBLIC URL: {match.group(0)}\n")
    threading.Thread(target=run, daemon=True).start()

# --- AI Worker ---
def ai_worker():
    global raw_frame_for_ai, latest_detections, latest_has_fire

    while True:
        with frame_lock:
            frame = raw_frame_for_ai.copy() if raw_frame_for_ai is not None else None

        if frame is None:
            time.sleep(0.05)
            continue

        orig_h, orig_w = frame.shape[:2]

        # ✅ FIXED PREPROCESSING (MATCHES WORKING VERSION)
        img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = session.run(None, {input_name: img})
        out = outputs[0][0]

        if out.shape[0] < out.shape[1]:
            out = out.T

        boxes, confs = [], []
        temp_detections = []
        has_fire = False

        for row in out:
            conf = np.max(row[4:])
            if conf > DETECTION_THRESHOLD:
                cx, cy, w, h = row[0:4]

                x_scale, y_scale = orig_w / 640, orig_h / 640

                x = int((cx - w / 2) * x_scale)
                y = int((cy - h / 2) * y_scale)
                bw = int(w * x_scale)
                bh = int(h * y_scale)

                # ✅ FIX: boundary safety
                x = max(0, x)
                y = max(0, y)
                bw = min(orig_w - x, bw)
                bh = min(orig_h - y, bh)

                boxes.append([x, y, bw, bh])
                confs.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes, confs, DETECTION_THRESHOLD, 0.45)

        if len(indices) > 0:
            has_fire = True
            for i in indices.flatten():
                temp_detections.append({"box": boxes[i], "conf": confs[i]})

        with frame_lock:
            latest_detections = temp_detections
            latest_has_fire = has_fire

        time.sleep(0.01)

# --- Camera Worker (ONLY THIS PART UPDATED PROPERLY) ---
def camera_worker():
    global raw_frame_for_ai, latest_jpeg

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # ✅ FIX: Proper controls
    picam2.set_controls({
        "AwbMode": 0,
        "FrameDurationLimits": (83333, 83333)  # ~12 FPS lock
    })

    print("🎥 Picamera2 Active (Corrected).")

    frame_duration = 1.0 / TARGET_FPS

    while True:
        start_time = time.time()

        # ✅ CAPTURE FRAME
        frame_raw = picam2.capture_array()

        # ✅ FIX: remove alpha + convert RGB → BGR
        frame = frame_raw[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with frame_lock:
            raw_frame_for_ai = frame.copy()
            current_detections = latest_detections
            current_has_fire = latest_has_fire

        # Draw detections
        if current_has_fire:
            for det in current_detections:
                x, y, w, h = det["box"]
                conf = det["conf"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"FIRE {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ✅ Encode once (optimized)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

        # FPS control
        elapsed = time.time() - start_time
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# --- Flask ---
@app.route('/video_feed')
def video_feed():
    def generate():
        last_frame = None
        while True:
            with frame_lock:
                jpeg = latest_jpeg

            if jpeg is None or jpeg == last_frame:
                time.sleep(0.02)
                continue

            last_frame = jpeg

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<body style="background:#000; display:flex; justify-content:center;"><img src="/video_feed" style="height:90vh;"></body>'

# --- MAIN ---
if __name__ == '__main__':
    start_cloudflare_background(port=5000)
    threading.Thread(target=ai_worker, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
