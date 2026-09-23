"""/camera/image_raw -> /detections (pixels + table metres) and /detections/image (overlay)."""

import math
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from intelpick_interfaces.msg import Detection, DetectionArray

from .calibration import TableCalibration
from .detectors.color import DEFAULT_COLORS, ColorDetector


class DetectorNode(Node):

    def __init__(self):
        super().__init__('detector')
        backend = self.declare_parameter('backend', 'color').value
        names = self.declare_parameter('color_names', list(DEFAULT_COLORS)).value
        colors = {n: self.declare_parameter(f'colors.{n}', DEFAULT_COLORS.get(n, [0])).value
                  for n in names}
        min_area = self.declare_parameter('min_area', 300).value
        max_area = self.declare_parameter('max_area', 40000).value
        # Objects must lie inside the calibrated area (convex hull of the calibration points),
        # grown by this margin in metres. Negative shrinks it.
        self.workspace_margin = self.declare_parameter('workspace_margin', 0.01).value
        # Optional extra pixel rectangle [u0, v0, u1, v1]; all zeros = off.
        self.roi = list(self.declare_parameter('roi', [0, 0, 0, 0]).value)
        calib_file = self.declare_parameter('calibration_file', '').value

        self.colors = ColorDetector(colors, min_area, max_area)
        if backend == 'color':
            self.detector = self.colors
        elif backend == 'yolo':
            from .detectors.yolo import YoloDetector
            model = self.declare_parameter('yolo_model', 'yolo26n.pt').value
            conf = self.declare_parameter('yolo_conf', 0.5).value
            tag = self.declare_parameter('yolo_color_tag', True).value
            self.detector = YoloDetector(model, conf, self.colors if tag else None)
        else:
            raise ValueError(f'unknown backend {backend!r}')

        self.calib = None
        if calib_file and os.path.exists(calib_file):
            self.calib = TableCalibration.load(calib_file)
            self.get_logger().info(
                f'calibration {calib_file}: rms {self.calib.rms_error * 1000:.1f} mm')
            if self.calib.workspace is None:
                self.get_logger().warn('calibration has no workspace outline (made by an older '
                                       'version): re-run calibrate to limit picks to the mat')
        else:
            self.get_logger().warn('no calibration loaded, x/y will be NaN (run calibrate)')

        self.bridge = CvBridge()
        self.pub = self.create_publisher(DetectionArray, 'detections', 10)
        self.pub_img = self.create_publisher(Image, 'detections/image', 1)
        self.create_subscription(Image, 'camera/image_raw', self.on_image, 1)
        self.get_logger().info(f'detector backend: {backend}')

    def in_roi(self, u, v):
        u0, v0, u1, v1 = self.roi
        return u1 <= u0 or (u0 <= u <= u1 and v0 <= v <= v1)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        calib = self.calib
        if calib and not calib.matches(msg.width, msg.height):
            self.get_logger().error(
                f'image is {msg.width}x{msg.height} but calibration was made at '
                f'{calib.image_size[0]}x{calib.image_size[1]}: fix the camera resolution or '
                're-run calibrate. Positions disabled.', throttle_duration_sec=5.0)
            calib = None
        out = DetectionArray(header=msg.header)
        rejected = []
        for b in self.detector.detect(frame):
            x, y = calib.pixel_to_table(b.u, b.v) if calib else (math.nan, math.nan)
            outside = calib is not None and not calib.in_workspace(x, y, self.workspace_margin)
            if outside or not self.in_roi(b.u, b.v):
                rejected.append(b)
                continue
            out.detections.append(Detection(
                label=b.label, confidence=float(b.confidence), u=float(b.u), v=float(b.v),
                angle=float(b.angle), area_px=float(b.area), x=float(x), y=float(y)))
        self.pub.publish(out)

        if self.pub_img.get_subscription_count():
            img = self.draw(frame, out, rejected, calib)
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def draw(self, frame, out, rejected, calib):
        outline = calib.workspace_pixels() if calib else None
        if outline is not None:
            cv2.polylines(frame, [outline.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        u0, v0, u1, v1 = self.roi
        if u1 > u0:
            cv2.rectangle(frame, (u0, v0), (u1, v1), (255, 255, 255), 1)
        for b in rejected:  # seen but ignored: outside the workspace / roi
            cv2.circle(frame, (int(b.u), int(b.v)), 5, (128, 128, 128), 2)
        for d in out.detections:
            c = (int(d.u), int(d.v))
            cv2.circle(frame, c, 5, (255, 255, 255), -1)
            text = d.label
            if not math.isnan(d.x):
                text += f' ({d.x * 100:.0f}, {d.y * 100:.0f}) cm'
            cv2.putText(frame, text, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return frame


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
