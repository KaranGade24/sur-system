import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
from flask_cors import CORS
import threading
import time
import sys
from picamera2 import Picamera2

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TARGET_FPS = 12.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_RES = 640 
DETECTION_THRESHOLD = 0.45

# --- Global State ---
frame_lock = threading.Lock()
raw_frame_rgb = None 
latest_detections = []
latest_has_fire = False
latest_jpeg = None

# --- AI Setup ---
try:
    session = ort.InferenceSession("fire_detector.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"❌ Model Error: {e}")
    sys.exit(1)

# --- AI Worker ---
def ai_worker():
    global raw_frame_rgb, latest_detections, latest_has_fire

    while True:
        with frame_lock:
            # We take the RGB frame directly for the AI
            img_in = raw_frame_rgb.copy() if raw_frame_rgb is not None else None

        if img_in is None:
            time.sleep(0.01)
            continue

        # 1. Resize for AI (640x640)
        img = cv2.resize(img_in, (MODEL_RES, MODEL_RES))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) 
        img = np.expand_dims(img, axis=0)

        outputs = session.run(None, {input_name: img})
        out = outputs[0][0]
        if out.shape[0] < out.shape[1]: out = out.T

        temp_detections = []
        has_fire = False

        for row in out:
            conf = np.max(row[4:])
            if conf > DETECTION_THRESHOLD:
                cx, cy, w, h = row[0:4]
                # FIX: Scaling logic to map 640x640 back to 640x480
                x = int(cx - w / 2)
                y = int((cy - h / 2) * (FRAME_HEIGHT / MODEL_RES))
                bw = int(w)
                bh = int(h * (FRAME_HEIGHT / MODEL_RES))
                temp_detections.append({"box": [x, y, bw, bh], "conf": float(conf)})
                has_fire = True

        # NMS to clean up boxes
        if has_fire:
            boxes = [d["box"] for d in temp_detections]
            confs = [d["conf"] for d in temp_detections]
            indices = cv2.dnn.NMSBoxes(boxes, confs, DETECTION_THRESHOLD, 0.45)
            final_dets = [temp_detections[i] for i in indices.flatten()] if len(indices) > 0 else []
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

    print("🎥 Picamera2 Started (RGB Mode).")

    while True:
        start_time = time.time()
        # Capture direct RGB array
        frame_rgb = picam2.capture_array()[:, :, :3]

        with frame_lock:
            raw_frame_rgb = frame_rgb.copy()
            current_detections = latest_detections
            current_has_fire = latest_has_fire

        # --- THE FIX FOR BLUE COLOR ---
        # Convert RGB to BGR ONLY for the Flask display/drawing
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if current_has_fire:
            for det in current_detections:
                x, y, w, h = det["box"]
                # BGR Color (0, 0, 255) is RED
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_frame, "FIRE", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Encode the BGR frame for the web
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

        # Frame rate limiter
        elapsed = time.time() - start_time
        time.sleep(max(0, (1.0 / TARGET_FPS) - elapsed))

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                jpeg = latest_jpeg
            if jpeg:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<body style="background:#000;"><img src="/video_feed" style="width:100%;"></body>'

if __name__ == '__main__':
    threading.Thread(target=ai_worker, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
