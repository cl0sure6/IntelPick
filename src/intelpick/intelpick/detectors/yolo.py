"""YOLO localisation with learned colour classes or HSV colour recognition.

class: use trained labels directly (red, green, blue, yellow for colour sorting).
color: localise with YOLO, recognise colour with HSV, output just the colour.
color_class: output red_cube etc. for bins configured with combined labels.
"""

import math
from pathlib import Path

from . import Blob


class YoloDetector:

    def __init__(self, model_path, conf=0.5, color_tagger=None, label_mode=None,
                 min_area=300, max_area=40000, device='cpu', imgsz=640,
                 color_min_fraction=0.2, color_min_margin=0.05):
        self.label_mode = label_mode or ('color' if color_tagger is not None else 'class')
        if self.label_mode not in ('class', 'color', 'color_class'):
            raise ValueError('yolo_label_mode must be class, color or color_class')
        if self.label_mode != 'class' and color_tagger is None:
            raise ValueError('color and color_class modes require an HSV color tagger')
        if not 0 < conf <= 1 or not 0 <= min_area <= max_area or imgsz <= 0:
            raise ValueError('invalid YOLO confidence, area limits or image size')
        if not 0 <= color_min_fraction <= 1 or not 0 <= color_min_margin <= 1:
            raise ValueError('color fraction and margin must be in [0, 1]')
        # Runtime must use an explicit local model; a typo must not download generic COCO weights.
        path = Path(model_path).expanduser()
        if not model_path or not path.is_file():
            raise ValueError(f'YOLO weights not found: {model_path!r}. Train with train_yolo, '
                             'then pass yolo_model:=/absolute/path/to/weights/best.pt')
        try:
            from ultralytics import YOLO  # heavy import only for this backend
        except ImportError as e:
            raise RuntimeError('Ultralytics is missing; see docs/YOLO.md for installation') from e
        self.model = YOLO(str(path))
        if self.model.task != 'detect':
            raise ValueError('IntelPick requires a YOLO object detection model (task=detect)')
        self.conf, self.device, self.imgsz = conf, device, imgsz
        self.min_area, self.max_area = min_area, max_area
        self.color_tagger = color_tagger
        self.color_min_fraction, self.color_min_margin = color_min_fraction, color_min_margin

    def detect(self, bgr):
        result = self.model.predict(bgr, conf=self.conf, device=self.device,
                                    imgsz=self.imgsz, verbose=False)[0]
        if result.boxes is None:
            raise RuntimeError('YOLO returned no bounding-box output; use a detection model')
        blobs = []
        height, width = bgr.shape[:2]
        for box in result.boxes:
            coords = [float(t) for t in box.xyxy[0]]
            confidence = float(box.conf.item())
            if not all(math.isfinite(t) for t in coords + [confidence]):
                continue
            x0, y0, x1, y1 = coords
            x0, x1 = max(0.0, x0), min(float(width), x1)
            y0, y1 = max(0.0, y0), min(float(height), y1)
            area = (x1 - x0) * (y1 - y0)
            if x1 <= x0 or y1 <= y0 or confidence < self.conf or not (
                    self.min_area <= area <= self.max_area):
                continue
            label = result.names[int(box.cls.item())]
            if self.label_mode != 'class':
                crop = bgr[math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1)]
                color = self.color_tagger.dominant_color(
                    crop, self.color_min_fraction, self.color_min_margin)
                if color is None:
                    continue
                label = color if self.label_mode == 'color' else f'{color}_{label}'
            blobs.append(Blob(label, confidence, (x0 + x1) / 2, (y0 + y1) / 2, 0.0, area))
        return blobs
