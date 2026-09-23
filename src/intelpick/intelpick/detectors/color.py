"""Classic HSV thresholding. Sorting by colour needs no neural network, and this is the baseline
the YOLO backend has to beat."""

import math

import cv2
import numpy as np

from . import Blob

# Starting points only: tune under your own lighting (see docs, "HSV tuning").
DEFAULT_COLORS = {
    'red': [0, 120, 70, 10, 255, 255, 170, 120, 70, 179, 255, 255],
    'green': [40, 80, 50, 85, 255, 255],
    'blue': [95, 120, 50, 130, 255, 255],
    'yellow': [20, 120, 100, 35, 255, 255],
}


class ColorDetector:

    def __init__(self, colors, min_area=300, max_area=40000):
        """
        colors: {name: [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi, ...]}, OpenCV HSV (H in 0..179).
                Several ranges per colour are allowed (red wraps around H=0, so it needs two).
        """
        self.ranges = {}
        for name, flat in colors.items():
            if len(flat) == 0 or len(flat) % 6:
                raise ValueError(f'color {name!r}: need a multiple of 6 values, got {len(flat)}')
            self.ranges[name] = [
                (np.array(flat[i:i + 3], np.uint8), np.array(flat[i + 3:i + 6], np.uint8))
                for i in range(0, len(flat), 6)
            ]
        self.min_area = min_area
        self.max_area = max_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def mask(self, hsv, name):
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self.ranges[name]:
            m |= cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._kernel)

    def detect(self, bgr):
        hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
        blobs = []
        for name in self.ranges:
            contours, _ = cv2.findContours(self.mask(hsv, name), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not self.min_area <= area <= self.max_area:
                    continue
                (u, v), (w, h), deg = cv2.minAreaRect(c)
                # How well the blob fills its bounding rectangle: ~1 for cubes, ~0.785 for caps.
                fill = area / max(w * h, 1.0)
                blobs.append(Blob(name, min(fill, 1.0), u, v, math.radians(deg), area))
        return blobs

    def dominant_color(self, bgr):
        """Colour covering the most pixels of a crop, or None. Used to colour-tag YOLO boxes."""
        if bgr.size == 0:
            return None
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        counts = {name: cv2.countNonZero(self.mask(hsv, name)) for name in self.ranges}
        best = max(counts, key=counts.get, default=None)
        if best is None or counts[best] < 0.2 * bgr.shape[0] * bgr.shape[1]:
            return None
        return best
