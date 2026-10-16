"""Stable, recent targets for the sorter, independent of ROS and arm hardware."""

import math
from collections import deque

from .grasp import pick_waypoints


class StableTargets:

    def __init__(self, labels, stable_frames=5, stable_tol=0.01, max_age=1.0,
                 min_reach=0.12, max_reach=0.40, approach='radial', standoff=0.04,
                 offset=0.0, min_confidence=0.5):
        if not all(math.isfinite(v) for v in (stable_tol, max_age, min_reach, max_reach, standoff)):
            raise ValueError('tracking tolerances and reach must be finite')
        if stable_frames < 1 or stable_tol < 0 or max_age <= 0:
            raise ValueError('stable_frames >= 1, stable_tol >= 0 and detection_max_age > 0 required')
        if not 0 < min_reach <= max_reach or standoff < 0 or not 0 <= min_confidence <= 1:
            raise ValueError('invalid reach, standoff or minimum confidence')
        if approach not in ('radial', 'vertical') or not math.isfinite(offset):
            raise ValueError('invalid approach or grasp offset')
        self.labels = set(labels)
        self.stable_frames, self.stable_tol, self.max_age = stable_frames, stable_tol, max_age
        self.min_reach, self.max_reach = min_reach, max_reach
        self.approach, self.standoff, self.offset = approach, standoff, offset
        self.min_confidence = min_confidence
        self.frames = deque(maxlen=stable_frames)
        self.last_frame = None

    def pickable(self, detection):
        d = detection
        if d.label not in self.labels or not all(math.isfinite(v) for v in
                                                (d.x, d.y, d.confidence)):
            return False
        if not self.min_confidence <= d.confidence <= 1:
            return False
        if not self.min_reach <= math.hypot(d.x, d.y) <= self.max_reach:
            return False
        # Check the actual grasp and stand-off points, including any configured offset.
        points = pick_waypoints(d.x, d.y, 0.0, 0.0, self.approach, self.standoff, self.offset)
        return all(self.min_reach <= math.hypot(x, y) <= self.max_reach for x, y, _, _ in points)

    def clear(self):
        self.frames.clear()
        self.last_frame = None

    def update(self, detections, now):
        if self.last_frame is not None and now - self.last_frame > self.max_age:
            self.clear()
        self.frames.append([d for d in detections if self.pickable(d)])
        self.last_frame = now

    def target(self, now):
        if self.last_frame is None or now - self.last_frame > self.max_age:
            self.clear()
            return None
        if len(self.frames) < self.stable_frames:
            return None
        for d in sorted(self.frames[-1], key=lambda d: math.hypot(d.x, d.y)):
            if all(any(o.label == d.label and
                       math.hypot(o.x - d.x, o.y - d.y) <= self.stable_tol for o in frame)
                   for frame in list(self.frames)[:-1]):
                return d
        return None
