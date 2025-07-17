# 🎯 Noodle Pack Detection & Advertisement System

This system uses a webcam, AprilTags, and a custom-trained YOLOv8 model to detect which noodle pack has been picked from a shelf. When a pack is removed, the corresponding promotional video is played in a fullscreen browser. The display returns to a default image when the pack is placed back.

---

## 📦 Features

- 🔍 **AprilTag detection** to detect pack interaction.
- 🧠 **YOLOv8 model** to identify which pack is missing.
- 🗳️ **Voting over 5 frames** to ensure robust missing-pack detection.
- 🎥 **Automatic video playback** based on detected pack.
- 🖼️ **Default fallback image** when no pack is picked.
- 🧩 **Web-based display** using Flask and Socket.IO (offline-ready).

---

## 🧰 Requirements

- Python 3.8+
- OpenCV
- Flask
- Flask-SocketIO
- Eventlet
- Ultralytics YOLOv8
- pupil_apriltags

### ✅ Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 How to Run

- Make sure your webcam is connected and available at cv2.VideoCapture(1). 
    - You can change it to 0 or 2 if needed.
- Start the app:

```bash
cd final_app
python app.py
```
- Access the display:
    - Open a browser and go to http://localhost:5000

Ensure it's fullscreen and stays open during operation.

### 🧠 How It Works
- AprilTags are placed below each noodle pack.
- When an AprilTag is detected stably (3+ frames), the system:
    - Captures 5 stable frames.
    - Runs YOLOv8 on each frame.
    - Determines which pack is missing based on majority vote.
    - Trigers the appropriate promotional video.

- When the AprilTag disappears, the system resets and shows the default image.

### 📌 Notes
- All videos and images are served locally, no internet required.
- Model weights are loaded from the runs/ folder, not exposed to frontend.
- Designed for commercial offline use with smooth transitions and low flicker.

### 🤝 Contributing
Have improvements or want to add new packs or features? Open an issue or pull request!

### 🛡️ License
This project is private and not licensed for public distribution unless specified otherwise.