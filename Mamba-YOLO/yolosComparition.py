#!/usr/bin/env python3
"""
Comparación entre YOLOv8n y modelo personalizado
"""

import cv2
import os
from ultralytics import YOLO
from ultralytics.solutions.object_counter import ObjectCounter
from ultralytics.engine.results import Boxes


def init_counter(class_name: str = "person") -> ObjectCounter:
    return ObjectCounter(
        classes_names={0: class_name},
        reg_pts=[(200, 400), (1000, 400)],
        view_img=False,
        draw_tracks=True,
        view_in_counts=True,
        view_out_counts=True,
    )


def process_frame(model: YOLO, counter: ObjectCounter, frame):
    results = model.track(frame, persist=True, verbose=False)

    if results and results[0].boxes.id is not None:
        result = results[0]
        boxes = result.boxes
        person_mask = boxes.cls == 0

        if person_mask.sum() > 0:
            filtered_data = boxes.data[person_mask]
            boxes_filtered = Boxes(filtered_data, boxes.orig_shape)
            result.boxes = boxes_filtered
            frame = counter.start_counting(frame, [result])

    banner_h = 40
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_h), (0, 0, 0), -1)
    total_count = counter.in_counts + counter.out_counts
    cv2.putText(frame,
                f"Total Persons: {total_count}",
                (20, int(banner_h * 0.75)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    return frame


def resize_to_screen(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


# new directory for results
os.makedirs("resultados", exist_ok=True)

# models
model_a = YOLO("yolov8n.pt")
model_b = YOLO("output_dir/mscoco/mambayolo33/weights/best.pt")

counter_a = init_counter("person")
counter_b = init_counter("person")

# input video path
video_path = "/root/deepLearning/test/video.mp4"
cap = cv2.VideoCapture(video_path)

# output video path
output_path = os.path.join("resultados", "output_comparacion.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_a = frame.copy()
    frame_b = frame.copy()

    processed_a = process_frame(model_a, counter_a, frame_a)
    processed_b = process_frame(model_b, counter_b, frame_b)

    h = min(processed_a.shape[0], processed_b.shape[0])
    w = min(processed_a.shape[1], processed_b.shape[1])
    processed_a = cv2.resize(processed_a, (w, h))
    processed_b = cv2.resize(processed_b, (w, h))

    combined = cv2.hconcat([processed_a, processed_b])

    cv2.putText(combined, "YOLOv8n", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(combined, "MambaYolo", (w + 50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    combined_resized = resize_to_screen(combined, 1280, 720)

    cv2.imshow("Model Comparison", combined_resized)
    out.write(combined_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved in: {output_path}")
