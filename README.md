# Real-Time Object Detection API with YOLOv8 & FastAPI

A production-ready FastAPI service that integrates Ultralytics YOLOv8 for real-time object detection across single images, video streams, and live webcam feeds.

## Project Structure

```text
.
├── main.py
├── webcam.py
├── test_api.py
├── requirements.txt
├── README.md
├── samples/
│   ├── test_image.jpg
│   └── test_video.mp4
└── screenshots/
    ├── swagger.png
    ├── image_detection.png
    └── webcam_detection.png

```

## Features

* **FastAPI Core**: Lightweight HTTP server providing clean routing and automatic Swagger documentation.
* **YOLOv8 Model Inference**: High-accuracy object detection powered by Ultralytics (`yolov8n.pt`).
* **Image & Video Detection**: Endpoints for detecting objects in uploaded static images and video clips.
* **Live Webcam Integration**: Real-time camera feed capture and annotation using OpenCV.
* **Automated Testing Suite**: Endpoint verification via `pytest` and `TestClient`.

## Installation & Setup

1. **Clone the repository & enter project directory**:
```bash
cd "YOLO object detection"

```


2. **Create and activate virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate

```


3. **Install dependencies**:
```bash
pip install -r requirements.txt

```



## Running the Application

### 1. Start the FastAPI Server

```bash
uvicorn main:app --reload

```

The server will run at `http://127.0.0.1:8000`.

### 2. Run Live Webcam Client

In a secondary terminal window (with `venv` activated):

```bash
python webcam.py

```

Press `q` to exit the webcam view.

### 3. Run Automated Tests

```bash
pytest test_api.py

```

## API Endpoints

* `GET /` - Health check and model metadata
* `GET /classes` - List all supported detection classes
* `POST /detect/image` - Detect objects in an uploaded image
* `POST /detect/video` - Detect objects in an uploaded video

## Interactive Documentation

Access the automatic OpenAPI docs at:

* Swagger UI: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`

```
