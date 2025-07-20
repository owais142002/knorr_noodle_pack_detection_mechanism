import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

EXTERNAL_CAMERA = True

cap = cv2.VideoCapture(1 if EXTERNAL_CAMERA else 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# # Optional: Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()