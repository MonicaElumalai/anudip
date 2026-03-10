from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import winsound
import os
import threading
from deepface import DeepFace

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
RECORD_FOLDER = "recorded_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Alarm Path
ALARM_PATH = os.path.join(app.root_path, "static", "alarm.wav")

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Global variables
cap = None
video_path = None
latest_frame = None

# Recording variables
recording = False
frame_count = 0
lock = threading.Lock()

# ==============================
# VIDEO STREAM GENERATOR
# ==============================
def generate_frames():
    global cap, latest_frame, recording, frame_count

    while cap is not None and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        latest_frame = frame.copy()

        results = model(frame, conf=0.4)
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    heatmap[cy, cx] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

        cv2.putText(frame, f"People Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if recording:
            with lock:
                frame_count += 1
                filename = f"{RECORD_FOLDER}/frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ==============================
# IMAGE MATCH ROUTE (DeepFace)
# ==============================
@app.route("/match_image", methods=["POST"])
def match_image():

    uploaded_file = request.files["image"]

    upload_path = os.path.join("static", "temp_upload.jpg")
    uploaded_file.save(upload_path)

    match_found = False

    for file in os.listdir(RECORD_FOLDER):

        if not file.endswith(".jpg"):
            continue

        stored_path = os.path.join(RECORD_FOLDER, file)

        try:

            result = DeepFace.verify(
                img1_path=upload_path,
                img2_path=stored_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=False
            )

            distance = result["distance"]

            # threshold control
            if distance < 0.6:
                match_found = True
                break

        except:
            continue

    if match_found:

        winsound.PlaySound(
            ALARM_PATH,
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )

        result_text = "MATCHED"

    else:

        result_text = "NO MATCHES"

    return render_template(
        "upload_page.html",
        result=result_text,
        uploaded_image="temp_upload.jpg"
    )
@app.route("/upload_page")
def upload_page():
    return render_template("upload_page.html")

@app.route("/camera_page")
def camera_page():
    return render_template("camera.html")

@app.route("/video_page")
def video_page():
    return render_template("video.html")

@app.route("/open_camera")
def open_camera():
    global cap
    if cap:
        cap.release()
    cap = cv2.VideoCapture(0)
    return redirect(url_for("camera_page"))

@app.route("/upload_video", methods=["POST"])
def upload_video():
    global cap, video_path
    if cap:
        cap.release()

    file = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    return redirect(url_for("video_page"))

@app.route("/stop")
def stop():
    global cap
    if cap:
        cap.release()
        cap = None

    if video_path:
        return redirect(url_for("video_page"))
    else:
        return redirect(url_for("camera_page"))

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ==============================
# RECORDING ROUTES
# ==============================
@app.route("/start_recording")
def start_recording():
    global recording, frame_count
    recording = True
    frame_count = 0
    return jsonify({"status": "Recording Started"})

@app.route("/stop_recording")
def stop_recording():
    global recording
    recording = False
    return jsonify({"status": "Recording Stopped"})

if __name__ == "__main__":
    app.run(debug=True)