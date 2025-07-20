import cv2
from pupil_apriltags import Detector

EXTERNAL_CAMERA = True

cap = cv2.VideoCapture(1 if EXTERNAL_CAMERA else 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = Detector(
    families='tag36h11',
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    for tag in tags:
        corners = [(int(x), int(y)) for x, y in tag.corners]
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i + 1) % 4]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw center
        cX, cY = int(tag.center[0]), int(tag.center[1])
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # Display Tag ID
        cv2.putText(frame, f"ID: {tag.tag_id}", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("AprilTag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()