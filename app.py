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
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIG  (tuned for Raspberry Pi 4/5, 2–4 GB)
# ─────────────────────────────────────────────
TARGET_FPS    = 10.0   # 10 FPS is the CCTV standard; easy on CPU/RAM
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
INFER_SIZE    = 320    # ← KEY: 320 instead of 640 — ~4× faster on ARM CPU
JPEG_QUALITY  = 60     # Lower = less RAM / network usage; 60 is fine for CCTV
CONF_THRESH   = 0.45
NMS_THRESH    = 0.45

# Limit OpenCV threads so they don't fight with Flask/AI threads on Pi's cores
cv2.setNumThreads(2)

# ─────────────────────────────────────────────
# GLOBAL SHARED STATE
# ─────────────────────────────────────────────
frame_lock        = threading.Lock()
raw_frame_for_ai  = None   # Latest raw frame → fed to AI thread
latest_detections = []     # Bounding boxes from AI thread
latest_has_fire   = False
latest_jpeg       = None   # Pre-encoded JPEG → served directly to browsers


# ─────────────────────────────────────────────
# ONNX MODEL LOAD
# ─────────────────────────────────────────────
# On 64-bit Pi OS: pip install onnxruntime   (works out of the box)
# On 32-bit Pi OS: pip install onnxruntime-armv7l  (third-party wheel)
providers = ['CPUExecutionProvider']
try:
    session    = ort.InferenceSession("fire_detector.onnx", providers=providers)
    input_name = session.get_inputs()[0].name
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────
# CAMERA HELPER — supports Pi Camera + USB cam
# ─────────────────────────────────────────────
def open_camera():
    """
    Try Pi Camera via libcamera (picamera2), then fall back to USB via OpenCV.
    Returns an object with .read() → (success, frame) interface.
    """
    # 1. Try picamera2 (Pi Camera Module on Pi OS Bullseye/Bookworm)
    try:
        from picamera2 import Picamera2
        import libcamera

        class PiCam2Wrapper:
            def __init__(self):
                self.cam = Picamera2()
                cfg = self.cam.create_video_configuration(
                    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                    controls={"FrameRate": TARGET_FPS}
                )
                self.cam.configure(cfg)
                self.cam.start()
                time.sleep(0.5)   # let sensor settle

            def read(self):
                frame = self.cam.capture_array()
                # picamera2 gives RGB — convert to BGR for OpenCV
                return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            def release(self):
                self.cam.stop()

            def set(self, *args):
                pass   # no-op; config is done at init

            def isOpened(self):
                return True

        print("📷 Using Pi Camera (picamera2)")
        return PiCam2Wrapper()

    except Exception:
        pass   # picamera2 not available → fall through to OpenCV

    # 2. Fall back to USB webcam via OpenCV
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # keep only newest frame
            cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
            print(f"📷 Using USB camera at index {idx}")
            return cap

    print("❌ No camera found.")
    sys.exit(1)


# ─────────────────────────────────────────────
# AI WORKER THREAD
# Runs inference in the background so camera
# capture is never blocked by slow ONNX calls.
# ─────────────────────────────────────────────
def ai_worker():
    global raw_frame_for_ai, latest_detections, latest_has_fire

    print("🧠 AI Engine started.")
    while True:
        # Grab latest raw frame
        with frame_lock:
            frame = raw_frame_for_ai.copy() if raw_frame_for_ai is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        orig_h, orig_w = frame.shape[:2]

        # ── Pre-process ──────────────────────────────
        # INFER_SIZE=320 is ~4× faster than 640 on ARM; still good accuracy
        img = cv2.resize(frame, (INFER_SIZE, INFER_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # ── Inference ────────────────────────────────
        outputs = session.run(None, {input_name: img})
        out = outputs[0][0]

        # Transpose if needed (YOLOv8 ONNX output can be [features, boxes])
        if out.shape[0] < out.shape[1] and len(out.shape) == 2:
            out = out.T

        boxes, confidences, current_detections = [], [], []
        has_fire = False

        # ── Post-process ─────────────────────────────
        x_scale = orig_w / INFER_SIZE
        y_scale = orig_h / INFER_SIZE

        for row in out:
            conf = float(np.max(row[4:]))
            if conf > CONF_THRESH:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x  = int((cx - w / 2) * x_scale)
                y  = int((cy - h / 2) * y_scale)
                bw = int(w * x_scale)
                bh = int(h * y_scale)
                boxes.append([x, y, bw, bh])
                confidences.append(conf)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

        if len(indices) > 0:
            has_fire = True
            for i in indices.flatten():
                current_detections.append({
                    "box":  boxes[i],
                    "conf": confidences[i]
                })

        # Push results
        with frame_lock:
            latest_detections = current_detections
            latest_has_fire   = has_fire

        # Brief rest to avoid thermal throttling on Pi
        time.sleep(0.02)


# ─────────────────────────────────────────────
# CAMERA WORKER THREAD
# Captures frames, draws bounding boxes, and
# encodes JPEG — no recording, pure streaming.
# ─────────────────────────────────────────────
def camera_worker():
    global raw_frame_for_ai, latest_jpeg

    cap            = open_camera()
    frame_duration = 1.0 / TARGET_FPS
    print("🎥 Camera worker started.")

    while True:
        loop_start = time.time()

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        # Share raw frame with AI thread
        with frame_lock:
            raw_frame_for_ai   = frame.copy()
            current_detections = latest_detections
            current_has_fire   = latest_has_fire

        # Draw bounding boxes on a copy (don't mutate the shared raw frame)
        display = frame.copy()
        if current_has_fire:
            for det in current_detections:
                bx, by, bw, bh = det["box"]
                conf = det["conf"]
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh),
                              (0, 0, 255), 3)
                cv2.putText(display, f"FIRE {conf:.2f}",
                            (bx, max(by - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add a status overlay in the corner
        status_text  = "🔥 FIRE DETECTED" if current_has_fire else "✅ Clear"
        status_color = (0, 0, 255) if current_has_fire else (0, 200, 0)
        cv2.putText(display, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Encode once → reused by all streaming clients (saves CPU)
        ret, buffer = cv2.imencode(
            '.jpg', display,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if ret:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

        # Strict frame pacing to prevent fast-forward / buffer overflow
        elapsed    = time.time() - loop_start
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route('/video_feed')
def video_feed():
    """MJPEG stream — works in any browser, VLC, or ffplay."""
    def generate():
        last_sent = None
        while True:
            with frame_lock:
                jpeg = latest_jpeg

            if jpeg is None or jpeg is last_sent:
                time.sleep(0.02)
                continue

            last_sent = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <title>🔥 Fire Detector — Live</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111;
      color: #eee;
      font-family: 'Segoe UI', sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 16px;
      min-height: 100vh;
    }
    h1 { font-size: 1.3rem; margin-bottom: 12px; letter-spacing: 1px; }
    #feed {
      width: 100%;
      max-width: 640px;
      border: 2px solid #333;
      border-radius: 10px;
      display: block;
    }
    footer {
      margin-top: 12px;
      font-size: 0.75rem;
      color: #555;
    }
  </style>
</head>
<body>
  <h1>🟢 Fire Detection — Live Stream</h1>
  <img id="feed" src="/video_feed" alt="Live feed">
  <footer>Powered by YOLOv8 ONNX · Raspberry Pi</footer>
</body>
</html>"""


# ─────────────────────────────────────────────
# CLOUDFLARE TUNNEL
# ─────────────────────────────────────────────
def start_cloudflare_tunnel(port=5000):
    """
    Starts a Cloudflare Quick Tunnel and prints the public URL.
    Install: sudo apt install cloudflared  OR
             curl -L https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /etc/apt/trusted.gpg.d/cloudflare.gpg
    """
    def run():
        print("🚀 Starting Cloudflare Tunnel...")

        # Find cloudflared binary
        path = shutil.which("cloudflared")
        if not path:
            for candidate in [
                "/usr/local/bin/cloudflared",
                "/usr/bin/cloudflared",
                "/snap/bin/cloudflared",
                os.path.expanduser("~/cloudflared"),
            ]:
                if os.path.isfile(candidate):
                    path = candidate
                    break

        if not path:
            print("❌ cloudflared not found.")
            print("   Install it: sudo apt install cloudflared")
            return

        process = subprocess.Popen(
            [path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            if "trycloudflare.com" in line:
                match = re.search(r"https?://[^\s'\"]+trycloudflare\.com[^\s'\"]*", line)
                if match:
                    url = match.group(0)
                    print("\n" + "=" * 55)
                    print(f"  🌍  PUBLIC URL: {url}")
                    print("=" * 55 + "\n")
            # Uncomment below to see all tunnel logs (useful for debugging):
            # print("[cloudflared]", line.strip())

    threading.Thread(target=run, daemon=True).start()


# ─────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────
def signal_handler(sig, frame):
    print("\n🛑 Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    start_cloudflare_tunnel(port=5000)

    # Start background workers
    threading.Thread(target=ai_worker,     daemon=True, name="AI-Worker").start()
    threading.Thread(target=camera_worker, daemon=True, name="Cam-Worker").start()

    # Small delay to let camera warm up before accepting connections
    time.sleep(1.5)

    print("🌐 Flask server starting on http://0.0.0.0:5000")
    # use_reloader=False is CRITICAL — reloader would fork and break threads
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
