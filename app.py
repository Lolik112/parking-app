import cv2
import pickle
import numpy as np
import json
from flask import Flask, Response, render_template, jsonify
import threading
import time
import os

app = Flask(__name__)

# Глобальні змінні
cap = None
posList = []
width, height = 107, 48
latest_frame = None
latest_stats = {"free": 0, "total": 0, "occupied": 0}
lock = threading.Lock()

def load_data():
    global posList, cap
    # Шлях відносно місця де лежить app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, 'CarParkPos'), 'rb') as f:
        posList = pickle.load(f)
    
    cap = cv2.VideoCapture(os.path.join(base_dir, 'carPark.mp4'))

def check_parking_space(img_pro, img):
    space_counter = 0
    spaces = []

    for pos in posList:
        x, y = pos
        img_crop = img_pro[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)

        is_free = count < 900
        if is_free:
            color = (0, 200, 80)
            thickness = 3
            space_counter += 1
        else:
            color = (0, 60, 220)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

        spaces.append({"x": x, "y": y, "free": is_free, "count": int(count)})

    return space_counter, spaces

def process_video():
    global latest_frame, latest_stats

    load_data()

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
        img_threshold = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 16
        )
        img_median = cv2.medianBlur(img_threshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_median, kernel, iterations=1)

        free_count, spaces = check_parking_space(img_dilate, img)
        total = len(posList)

        # Overlay статистики
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (320, 75), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, f"Вільно: {free_count}/{total}", (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 100), 3)

        with lock:
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            latest_frame = buffer.tobytes()
            latest_stats = {
                "free": free_count,
                "total": total,
                "occupied": total - free_count,
                "percent_free": round(free_count / total * 100) if total > 0 else 0
            }

        time.sleep(0.033)  # ~30 fps

def generate_frames():
    while True:
        with lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with lock:
        return jsonify(latest_stats)

if __name__ == '__main__':
    thread = threading.Thread(target=process_video, daemon=True)
    thread.start()
    app.run(debug=False, host='0.0.0.0', port=5000)
