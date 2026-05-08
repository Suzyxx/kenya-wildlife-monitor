from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

ANIMAL_COCO_IDS = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

DISPLAY_NAMES = {
    "bird":     "Bird",
    "cat":      "Wildcat/Leopard",
    "dog":      "African Wild Dog",
    "horse":    "Horse/Donkey",
    "sheep":    "Sheep/Antelope",
    "cow":      "Cattle/Buffalo",
    "elephant": "Elephant",
    "bear":     "Monkey/Baboon",
    "zebra":    "Zebra",
    "giraffe":  "Giraffe",
}

CONSERVATION_STATUS = {
    "zebra":    "Endangered",
    "elephant": "Vulnerable",
    "giraffe":  "Vulnerable",
    "dog":      "Endangered",
    "cat":      "Vulnerable",
}

@dataclass
class Detection:
    label: str
    display_name: str
    confidence: float
    bbox: tuple
    area_fraction: float
    conservation_status: Optional[str] = None


class WildlifeDetector:

    def __init__(self, confidence: float = 0.35, model_size: str = "n"):
        self.confidence = confidence
        self._model = None
        self._model_size = model_size
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_name = f"yolov8{self._model_size}.pt"
            print(f"[DETECTOR] Loading YOLOv8{self._model_size} on CPU...")
            self._model = YOLO(model_name)
            self._model.to("cpu")
            print("[DETECTOR] Model ready.")
        except ImportError:
            print("[DETECTOR] ultralytics not installed – falling back to mock detector.")
            self._model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is None:
            return self._mock_detect(frame)
        h, w = frame.shape[:2]
        results = self._model(frame, conf=self.confidence, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in ANIMAL_COCO_IDS:
                continue
            conf  = float(box.conf[0])
            label = ANIMAL_COCO_IDS[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
            detections.append(Detection(
                label               = label,
                display_name        = DISPLAY_NAMES.get(label, label),
                confidence          = round(conf, 3),
                bbox                = (x1, y1, x2, y2),
                area_fraction       = round(area_frac, 4),
                conservation_status = CONSERVATION_STATUS.get(label),
            ))
        return detections

    def annotate_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated = frame.copy()
        COLOR_MAP = {
            "elephant": (0, 140, 255),
            "zebra":    (255, 200, 0),
            "giraffe":  (0, 200, 255),
            "bird":     (0, 255, 180),
        }
        default_color = (50, 255, 50)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = COLOR_MAP.get(d.label, default_color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"{d.display_name} {d.confidence:.0%}"
            cv2.putText(annotated, label_text, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return annotated

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        import random
        h, w = frame.shape[:2]
        animals = random.choices(
            ["elephant", "zebra", "giraffe", "bird", "cow"],
            weights=[0.2, 0.25, 0.15, 0.3, 0.1],
            k=random.randint(0, 4)
        )
        result = []
        for animal in animals:
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = min(x1 + random.randint(50, 200), w)
            y2 = min(y1 + random.randint(50, 200), h)
            result.append(Detection(
                label               = animal,
                display_name        = DISPLAY_NAMES.get(animal, animal),
                confidence          = round(random.uniform(0.36, 0.95), 3),
                bbox                = (x1, y1, x2, y2),
                area_fraction       = round(((x2-x1)*(y2-y1))/(w*h), 4),
                conservation_status = CONSERVATION_STATUS.get(animal),
            ))
        return result