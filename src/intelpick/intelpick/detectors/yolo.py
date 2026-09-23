"""Ultralytics YOLO backend.

Pretrained COCO weights know "bottle" and "cup" but not "cube" or "bottle cap", and no YOLO
class encodes colour. For this project train a small custom model (see docs/ARCHITECTURE.md)
and optionally tag each box with its dominant colour from the ColorDetector.
"""

from . import Blob


class YoloDetector:

    def __init__(self, model_path, conf=0.5, color_tagger=None):
        from ultralytics import YOLO  # heavy import, only when this backend is selected
        self.model = YOLO(model_path)
        self.conf = conf
        self.color_tagger = color_tagger

    def detect(self, bgr):
        result = self.model.predict(bgr, conf=self.conf, verbose=False)[0]
        blobs = []
        for box in result.boxes:
            x0, y0, x1, y1 = (float(t) for t in box.xyxy[0])
            label = result.names[int(box.cls)]
            if self.color_tagger:
                color = self.color_tagger.dominant_color(bgr[int(y0):int(y1), int(x0):int(x1)])
                if color:
                    label = f'{color}_{label}'
            # Axis-aligned boxes carry no rotation; train an OBB model if the grasp needs it.
            blobs.append(Blob(label, float(box.conf), (x0 + x1) / 2, (y0 + y1) / 2,
                              0.0, (x1 - x0) * (y1 - y0)))
        return blobs
