import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def create_dummy_image():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "Test",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    _, encoded = cv2.imencode(".jpg", img)
    return io.BytesIO(encoded.tobytes())


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "model" in json_data


def test_detect_image_endpoint():
    dummy_image = create_dummy_image()
    files = {"file": ("test.jpg", dummy_image, "image/jpeg")}
    response = client.post("/detect/image", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_detect_video_endpoint():
    dummy_image = create_dummy_image()
    files = {"file": ("test.mp4", dummy_image, "video/mp4")}
    response = client.post("/detect/video", files=files)
    assert response.status_code in [200, 400, 422]