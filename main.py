import io
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 Object Detection API")

model = YOLO("yolov8n.pt")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "yolov8n.pt",
    }


@app.get("/classes")
def get_classes():
    return {"classes": list(model.names.values())}


def get_class_ids(classes: str):
    selected = [name.strip() for name in classes.split(",") if name.strip()]
    if not selected:
        return None
    return [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in selected
    ]


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.4, ge=0.0, le=1.0),
    classes: str = Query(""),
):
    contents = await file.read()
    array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )
    class_ids = get_class_ids(classes)
    results = model(
        frame,
        conf=confidence,
        classes=class_ids,
        verbose=False,
    )
    annotated = results[0].plot()
    counts = {}
    boxes = results[0].boxes
    if boxes is not None:
        for class_id in boxes.cls.tolist():
            name = model.names[int(class_id)]
            counts[name] = counts.get(name, 0) + 1
    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode output image",
        )
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"X-Detections": json.dumps(counts)},
    )


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Query(0.4, ge=0.0, le=1.0),
    classes: str = Query(""),
):
    suffix = Path(file.filename or "video.mp4").suffix
    if not suffix:
        suffix = ".mp4"
    class_ids = get_class_ids(classes)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / f"input{suffix}"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(await file.read())
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Invalid video file",
            )
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model(
                frame,
                conf=confidence,
                classes=class_ids,
                verbose=False,
            )
            annotated = results[0].plot()
            writer.write(annotated)
        cap.release()
        writer.release()
        video_bytes = output_path.read_bytes()
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="detected.mp4"'},
    )