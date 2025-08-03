# --- Combined and Updated Code for Simultaneous Dual CCTV Camera Input ---
# This combines and adapts both of your original files with multithreading support for two CCTV cameras.

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, Response, jsonify, request
import os
import time
import csv
from datetime import datetime
import warnings
import threading

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

# Config
SIMILARITY_THRESHOLD = 0.4
student_embeddings = np.load('student_embeddings.npy', allow_pickle=True).item()
last_update_times = {}
recognized_students_cam1 = []
recognized_students_cam2 = []

# Setup face recognition
face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# Flask app
app = Flask(__name__)
recording_cam1 = False
recording_cam2 = False
video_writer_cam1 = None
video_writer_cam2 = None

# Create recordings folder
os.makedirs('recordings', exist_ok=True)

# CCTV camera inputs
cap1 = cv2.VideoCapture("rtsp://admin:admin%4012345@172.16.46.25:554/Streaming/Channels/101")
cap2 = cv2.VideoCapture("rtsp://admin:admin%4012345@172.16.46.116:554/Streaming/Channels/101")
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def log_attendance(name, roll_number, timestamp):
    with open('attendance_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, roll_number, timestamp])

def generate_frames(cam_id):
    cap = cap1 if cam_id == 1 else cap2
    recognized_students = recognized_students_cam1 if cam_id == 1 else recognized_students_cam2
    recording_flag = lambda: recording_cam1 if cam_id == 1 else recording_cam2
    get_writer = lambda: video_writer_cam1 if cam_id == 1 else video_writer_cam2

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = face_app.get(frame)
        recognized_students.clear()

        for face in faces:
            face_embedding = face.embedding
            similarities = {
                name: cosine_similarity([face_embedding], [embedding])[0][0]
                for name, embedding in student_embeddings.items()
            }

            if similarities:
                predicted_name = max(similarities, key=similarities.get)
                max_similarity = similarities[predicted_name]

                if max_similarity >= SIMILARITY_THRESHOLD:
                    name_parts = predicted_name.split('_')
                    name = " ".join(name_parts[:-1])
                    roll_number = name_parts[-1]
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    student_key = f"{name}_{roll_number}"
                    if student_key not in last_update_times or \
                            (datetime.now() - last_update_times[student_key]).total_seconds() > 3:
                        log_attendance(name, roll_number, timestamp)
                        last_update_times[student_key] = datetime.now()

                    recognized_students.append({
                        "name": name,
                        "roll_number": roll_number,
                        "timestamp": timestamp
                    })

                    x1, y1, x2, y2 = map(int, face.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Name: {name}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Roll: {roll_number}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(frame, f"Time: {timestamp}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if recording_flag() and get_writer():
            get_writer().write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index_recording.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/student_info1')
def student_info1():
    return jsonify(recognized_students_cam1)

@app.route('/student_info2')
def student_info2():
    return jsonify(recognized_students_cam2)

@app.route('/start_recording/<int:cam_id>', methods=['POST'])
def start_recording(cam_id):
    global video_writer_cam1, video_writer_cam2, recording_cam1, recording_cam2

    filename = datetime.now().strftime(f"cam{cam_id}_%Y%m%d_%H%M%S.avi")
    filepath = os.path.join("recordings", filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if cam_id == 1 and not recording_cam1:
        video_writer_cam1 = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))
        recording_cam1 = True
        return jsonify({"status": "started", "file": filepath})
    elif cam_id == 2 and not recording_cam2:
        video_writer_cam2 = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))
        recording_cam2 = True
        return jsonify({"status": "started", "file": filepath})
    return jsonify({"status": "already recording"})

@app.route('/stop_recording/<int:cam_id>', methods=['POST'])
def stop_recording(cam_id):
    global video_writer_cam1, video_writer_cam2, recording_cam1, recording_cam2

    if cam_id == 1 and recording_cam1:
        recording_cam1 = False
        video_writer_cam1.release()
        video_writer_cam1 = None
        return jsonify({"status": "stopped"})
    elif cam_id == 2 and recording_cam2:
        recording_cam2 = False
        video_writer_cam2.release()
        video_writer_cam2 = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not recording"})

if __name__ == '__main__':
    try:
        with open('attendance_log.csv', 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Roll Number', 'Timestamp'])
    except FileExistsError:
        pass
    app.run(debug=True, threaded=True)
