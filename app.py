from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
from pupil_apriltags import Detector
from ultralytics import YOLO
import threading
import time
from collections import Counter

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

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

PACK_LABELS = {"black_noodle_pack": "black", "green_noodle_pack": "green",
               "orange_noodle_pack": "orange", "red_noodle_pack": "red"}

DEFAULT_STATE = {"playing": False, "current": "default"}
STATE = DEFAULT_STATE.copy()

def detect_loop():
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
                    detected = set(results[0].names[i] for i in results[0].boxes.cls.cpu().numpy().astype(int))

                    for label in PACK_LABELS:
                        if label not in detected:
                            missing_votes.append(PACK_LABELS[label])
                            break

                if missing_votes:
                    print(missing_votes)
                    most_common = Counter(missing_votes).most_common(1)
                    print(f"The missing pack is: {most_common}")
                    if most_common:
                        chosen_pack = most_common[0][0]
                        if STATE["current"] != chosen_pack:
                            STATE["playing"] = True
                            STATE["current"] = chosen_pack
                            socketio.emit('video_update', {'type': 'video', 'name': chosen_pack})
        else:
            tag_stable_count = 0
            if tag_active and (time.time() - last_seen_tag_time > 1):
                tag_active = False
                STATE.update(DEFAULT_STATE)
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
