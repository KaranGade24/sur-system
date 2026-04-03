print("start import")

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
import threading
import subprocess
import time
import shutil
import os
import signal
import sys
#from recorder import VideoRecorder
import re
from flask_cors import CORS


# ✅ NEW: Picamera2 import
from picamera2 import Picamera2

print("import done")

app = Flask(__name__)

CORS(app)


# --- Configuration for Low-End Hardware (<= 2GB RAM) ---
TARGET_FPS = 10.0  # 10 FPS is standard for CCTV, uses far less RAM/CPU
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# --- Global State for Thread Decoupling ---
frame_lock = threading.Lock()
raw_frame_for_ai = None      # The raw frame sent to the AI
latest_detections = []       # Bounding boxes updated by AI
latest_has_fire = False      # Fire status updated by AI
latest_jpeg = None           # Pre-encoded image for instant web streaming

#recorder = VideoRecorder(fps=TARGET_FPS)

# --- AI Setup ---
# OpenVINO can consume extra RAM. If on 2GB RAM, CPUExecutionProvider is safer and stable.
providers = ['CPUExecutionProvider'] 

try:
    # ✅ NEW: Optimize ONNX for Raspberry Pi
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1

    session = ort.InferenceSession("fire_detector.onnx", sess_options=so, providers=providers)
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

def ai_worker():
    """Background thread that strictly handles AI inference to prevent video lag."""
    global raw_frame_for_ai, latest_detections, latest_has_fire
    
    print("🧠 Background AI Engine Started.")
    while True:
        # 1. Grab the latest frame safely
        with frame_lock:
            if raw_frame_for_ai is None:
                frame = None
            else:
                frame = raw_frame_for_ai.copy()
        
        if frame is None:
            time.sleep(0.1)
            continue

        orig_h, orig_w = frame.shape[:2]
        
        # 2. Ultra-fast Preprocessing
        # INTER_LINEAR is faster than default cubic resizing
        img = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)  # ✅ reduced for Pi
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) 
        img = np.expand_dims(img, axis=0)  

        # 3. Run Inference
        outputs = session.run(None, {input_name: img})
        out = outputs[0][0]
        if out.shape[0] < out.shape[1] and len(out.shape) == 2: 
            out = out.T

        boxes, confidences, current_detections = [], [], []
        has_fire = False

        # 4. Fast Post-Processing
        for row in out:
            conf = np.max(row[4:])
            if conf > 0.45:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x_scale, y_scale = orig_w / 320, orig_h / 320
                
                x = int((cx - w / 2) * x_scale)
                y = int((cy - h / 2) * y_scale)
                bw, bh = int(w * x_scale), int(h * y_scale)

                boxes.append([x, y, bw, bh])
                confidences.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.45)

        if len(indices) > 0:
            has_fire = True
            for i in indices.flatten():
                bx, by, bw, bh = boxes[i]
                conf = confidences[i]
                current_detections.append({"box": [bx, by, bw, bh], "conf": conf})

        # 5. Push results back to global state
        with frame_lock:
            latest_detections = current_detections
            latest_has_fire = has_fire
            
        # Give the CPU a tiny rest to prevent thermal throttling
        time.sleep(0.01)

def camera_and_record_worker():
    """Handles camera capture, drawing, recording, and streaming at a strict FPS."""
    global raw_frame_for_ai, latest_jpeg
    
    # ✅ REPLACED OpenCV camera with Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
    picam2.configure(config)
    picam2.start()
    
    print("🎥 Camera & Recording Engine Started.")
    frame_duration = 1.0 / TARGET_FPS

    while True:
        start_time = time.time()

        # ✅ Capture frame from Pi camera
        frame = picam2.capture_array()
        success = True

        if not success: 
            time.sleep(0.1)
            continue

        # 1. Update the AI with the raw frame
        with frame_lock:
            raw_frame_for_ai = frame.copy()
            current_detections = latest_detections
            current_has_fire = latest_has_fire

        # 2. Draw the latest known bounding boxes
        display_frame = frame.copy()
        if current_has_fire:
            for det in current_detections:
                bx, by, bw, bh = det["box"]
                conf = det["conf"]
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
                cv2.putText(display_frame, f"FIRE {conf:.2f}", (bx, by-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. Save to Video File
        #recorder.write_frame(display_frame, current_has_fire, current_detections)

        # 4. Pre-encode JPEG ONCE (Saves massive CPU power for web clients)
        # Quality 70 is visually fine but heavily reduces network/RAM usage
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

        # 5. Strict Frame Pacing (Fixes the Fast-Forward Video Issue)
        elapsed_time = time.time() - start_time
        sleep_time = frame_duration - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

# --- Flask Routes ---
@app.route('/video_feed')
def video_feed():
    def generate():
        last_sent_frame = None
        while True:
            with frame_lock:
                jpeg = latest_jpeg
            
            # Don't waste bandwidth sending the exact same frame again
            if jpeg is None or jpeg == last_sent_frame:
                time.sleep(0.02)
                continue
                
            last_sent_frame = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <head><title>AI Surveillance</title><meta name="viewport" content="width=device-width, initial-scale=1"></head>
    <body style="background:#111; color:white; font-family:sans-serif; text-align:center; padding:10px;">
        <h2>🟢 Surveillance Active</h2>
        <img src='/video_feed' style="width:100%; max-width:640px; border:2px solid #333; border-radius:8px;">
    </body>
    """

def start_cloudflare_background(port=5000):
    def run():
        print("🚀 Starting Cloudflare Tunnel...")
        path = shutil.which("cloudflared")
        
        if not path:
            candidates = ["/usr/local/bin/cloudflared", "/usr/bin/cloudflared", "/snap/bin/cloudflared"]
            for c in candidates:
                if os.path.exists(c):
                    path = c
                    break
        
        if not path:
            print("❌ cloudflared not found. Please install it or add to PATH.")
            return

        process = subprocess.Popen(
            [path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            if "trycloudflare.com" in line:
                match = re.search(r"https?://[^\s']+", line)
                if match:
                    print("\n" + "="*50)
                    print(f"🌍 PUBLIC URL: {match.group(0)}")
                    print("="*50 + "\n")

    threading.Thread(target=run, daemon=True).start()

def signal_handler(sig, frame):
    print("\n🛑 Saving video and shutting down safely...")
    #recorder.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    start_cloudflare_background(port=5000)
    
    threading.Thread(target=ai_worker, daemon=True).start()
    threading.Thread(target=camera_and_record_worker, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
