import cv2
import numpy as np
import requests

API_URL = "http://localhost:8000/detect/image"
CONFIDENCE = 0.4
CLASSES = ""

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        continue
    try:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    "frame.jpg",
                    buffer.tobytes(),
                    "image/jpeg",
                )
            },
            params={
                "confidence": CONFIDENCE,
                "classes": CLASSES,
            },
            timeout=5,
        )
        if response.status_code == 200:
            array = np.frombuffer(
                response.content,
                np.uint8,
            )
            annotated = cv2.imdecode(
                array,
                cv2.IMREAD_COLOR,
            )
        else:
            annotated = frame
    except requests.RequestException:
        annotated = frame
    cv2.imshow("YOLOv8 Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()