"""Camera pixel -> table plane mapping.

The camera looks down at a flat table, so a single homography maps image pixels to (x, y) in the
arm base frame. The table height (z, arm frame) is stored alongside so pick heights can be given
relative to the table surface. So is the image size: the mapping is only valid at the
resolution it was measured at.

The mapping is also only trustworthy inside the area the calibration points covered; outside it
errors grow quickly. That area (the convex hull of the points, table metres) is saved as the
workspace, and the detector ignores objects outside it.
"""

import cv2
import numpy as np
import yaml


class TableCalibration:

    def __init__(self, H, table_z, rms_error=float('nan'), image_size=None, workspace=None):
        self.H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        self.table_z = float(table_z)    # metres, arm frame
        self.rms_error = float(rms_error)  # metres, fit residual
        self.image_size = tuple(image_size) if image_size else None  # (width, height) px
        # Convex polygon on the table (metres, arm frame); None = unknown, nothing is excluded.
        self.workspace = None
        if workspace is not None:
            self.workspace = np.asarray(workspace, dtype=np.float32).reshape(-1, 2)

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
        hull = cv2.convexHull(dst.astype(np.float32)).reshape(-1, 2)
        calib = cls(H, table_z, image_size=image_size, workspace=hull)
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
            'workspace': None if self.workspace is None else self.workspace.tolist(),
        }
        with open(path, 'w') as f:
            yaml.safe_dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data['homography'], data['table_z'], data.get('rms_error_m', float('nan')),
                   data.get('image_size'), data.get('workspace'))

    def in_workspace(self, x, y, margin=0.0):
        """True if (x, y) metres lies inside the calibrated area grown by `margin` metres."""
        if self.workspace is None:
            return True
        dist = cv2.pointPolygonTest(self.workspace, (float(x), float(y)), True)  # >0 inside
        return dist >= -margin

    def workspace_pixels(self):
        """Workspace outline in image pixels (Nx2 int), for drawing, or None."""
        if self.workspace is None:
            return None
        pts = self.workspace.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2).round().astype(int)

    def matches(self, width, height):
        """False if the camera resolution differs from the one calibrated at."""
        return self.image_size is None or tuple(self.image_size) == (width, height)
