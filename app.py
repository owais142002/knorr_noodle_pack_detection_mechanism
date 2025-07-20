from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
from pupil_apriltags import Detector
from ultralytics import YOLO
import threading
import time
from collections import Counter

import serial
# open and give Arduino a moment to boot
ser = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)

EXTERNAL_CAMERA = True 
app = Flask(__name__)
socketio = SocketIO(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600

model = YOLO('runs/detect/train/weights/best.pt')

detector = Detector(
    families='tag36h11',
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

cap = cv2.VideoCapture(1 if EXTERNAL_CAMERA else 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

PACK_LABELS = {
    "black_noodle_pack": "black",
    "green_noodle_pack": "green",
    "orange_noodle_pack": "orange",
    "red_noodle_pack": "red"
}
PACK_CHARS = {'black': 'a', 'green': 'b', 'orange': 'c', 'red': 'd'}

DEFAULT_STATE = {"playing": False, "current": "default"}
STATE = DEFAULT_STATE.copy()

SESSION_LABELS = set()  # Will be filled after calibration


def calibrate_objects():
    print("[Calibration] Starting 10-second object setup phase...")
    detected_labels = []

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, imgsz=640, conf=0.7, verbose=False)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        detected = set(results[0].names[i] for i in classes)
        for label in PACK_LABELS:
            if label in detected:
                detected_labels.append(PACK_LABELS[label])

        # Show detection overlay in OpenCV window during calibration
        annotated = results[0].plot()
        cv2.imshow("Calibration View", annotated)
        cv2.waitKey(1)
        time.sleep(1)

    cv2.destroyAllWindows()
    counted = Counter(detected_labels)
    most_common = set(counted.keys())
    print(f"[Calibration] Detected objects: {most_common}")
    return most_common


def detect_loop():
    global SESSION_LABELS

    SESSION_LABELS = calibrate_objects()

    # Notify frontend of detected objects
    socketio.emit('video_update', {
        'type': 'info',
        'name': ', '.join(SESSION_LABELS)
    })

    time.sleep(5)

    tag_stable_count = 0
    last_seen_tag_time = time.time()
    tag_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        if tags:
            tag_stable_count += 1
            last_seen_tag_time = time.time()

            if tag_stable_count >= 3 and not tag_active:
                tag_active = True
                tag_stable_count = 0

                missing_votes = []

                for _ in range(5):
                    time.sleep(0.2)
                    ret, stable_frame = cap.read()
                    if not ret:
                        continue

                    results = model.predict(source=stable_frame, imgsz=640, conf=0.5, verbose=False)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    detected = set(results[0].names[i] for i in classes)

                    for label in PACK_LABELS:
                        # Check only SESSION_LABELS
                        if PACK_LABELS[label] in SESSION_LABELS and label not in detected:
                            missing_votes.append(PACK_LABELS[label])
                            break

                if missing_votes:
                    print("[Detection] Missing votes:", missing_votes)
                    most_common = Counter(missing_votes).most_common(1)
                    print(f"[Detection] The missing pack is: {most_common}")
                    if most_common:
                        chosen_pack = most_common[0][0]
                        if STATE["current"] != chosen_pack:
                            STATE["playing"] = True
                            STATE["current"] = chosen_pack
                            # blink that LED
                            ser.write(PACK_CHARS[chosen_pack].encode())
                            socketio.emit('video_update', {'type': 'video', 'name': chosen_pack})
        else:
            tag_stable_count = 0
            if tag_active and (time.time() - last_seen_tag_time > 1):
                tag_active = False
                STATE.update(DEFAULT_STATE)
                # stop blinking
                ser.write(b's')
                socketio.emit('video_update', {'type': 'image', 'name': 'default'})

        time.sleep(0.1)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    thread = threading.Thread(target=detect_loop)
    thread.daemon = True
    thread.start()
    socketio.run(app, host='0.0.0.0', port=5000)
