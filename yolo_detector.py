from typing import TypedDict

import cv2
import torch
from ultralytics import YOLOWorld


class Detection(TypedDict):
    class_name: str
    confidence: float
    bbox: list[float]


class YOLODetector:
    def __init__(self, confidence: float = 0.1):
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.model.set_classes(["red ball", "red sphere"])

    def detect(self, frame) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        return [
            {
                "class_name": result.names[int(box.cls[0].item())],
                "confidence": float(box.conf[0].item()),
                "bbox": box.xyxy[0].tolist(),
            }
            for box in result.boxes
        ]

    def annotate(self, frame, detections: list[Detection]):
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox"])
            center_x = int(round((detection["bbox"][0] + detection["bbox"][2]) / 2))
            center_y = int(round((detection["bbox"][1] + detection["bbox"][3]) / 2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(annotated, (center_x, center_y), 4, (0, 255, 0), -1)
            label = (f'{detection["class_name"]} {detection["confidence"]:.2f} ' f'({center_x}, {center_y})')
            cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return annotated
