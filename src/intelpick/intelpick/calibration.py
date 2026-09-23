"""Camera pixel -> table plane mapping.

The camera looks down at a flat table, so a single homography maps image pixels to (x, y) in the
arm base frame. The table height (z, arm frame) is stored alongside so pick heights can be given
relative to the table surface. So is the image size: the mapping is only valid at the
resolution it was measured at.
"""

import cv2
import numpy as np
import yaml


class TableCalibration:

    def __init__(self, H, table_z, rms_error=float('nan'), image_size=None):
        self.H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        self.table_z = float(table_z)    # metres, arm frame
        self.rms_error = float(rms_error)  # metres, fit residual
        self.image_size = tuple(image_size) if image_size else None  # (width, height) px

    @classmethod
    def fit(cls, pixels, table_xy, table_z, image_size=None):
        """pixels: Nx2 image points, table_xy: Nx2 arm-frame points in metres, N >= 4."""
        src = np.asarray(pixels, dtype=np.float64)
        dst = np.asarray(table_xy, dtype=np.float64)
        if len(src) < 4 or len(src) != len(dst):
            raise ValueError('need at least 4 matching point pairs')
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            raise ValueError('degenerate points (collinear?)')
        calib = cls(H, table_z, image_size=image_size)
        err = np.linalg.norm(calib.pixels_to_table(src) - dst, axis=1)
        calib.rms_error = float(np.sqrt(np.mean(err ** 2)))
        return calib

    def pixels_to_table(self, uv):
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def pixel_to_table(self, u, v):
        x, y = self.pixels_to_table([[u, v]])[0]
        return float(x), float(y)

    def save(self, path):
        data = {
            'homography': self.H.flatten().tolist(),
            'table_z': self.table_z,
            'rms_error_m': self.rms_error,
            'image_size': list(self.image_size) if self.image_size else None,
        }
        with open(path, 'w') as f:
            yaml.safe_dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data['homography'], data['table_z'], data.get('rms_error_m', float('nan')),
                   data.get('image_size'))

    def matches(self, width, height):
        """False if the camera resolution differs from the one calibrated at."""
        return self.image_size is None or tuple(self.image_size) == (width, height)
