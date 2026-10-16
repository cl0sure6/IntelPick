"""Real ROS messages and services; synthetic images and the dry-run arm only."""

import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('intelpick_interfaces.msg')
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from intelpick_interfaces.msg import Detection, DetectionArray
from intelpick.arm_node import ArmNode
from intelpick.calibration import TableCalibration
from intelpick.detector_node import DetectorNode
from intelpick.sorter_node import SorterNode


@pytest.mark.parametrize('backend', ['color', 'yolo'])
def test_camera_image_through_detector_sorter_and_arm_services(tmp_path, monkeypatch, backend):
    calib_path = tmp_path / 'calibration.yaml'
    pixels = [(100, 100), (540, 100), (540, 380), (100, 380)]
    table = [(0.25 - (v - 240) * 0.0005, -(u - 320) * 0.0005) for u, v in pixels]
    TableCalibration.fit(pixels, table, -0.105, image_size=(640, 480)).save(calib_path)
    weights = tmp_path / 'best.pt'
    weights.touch()

    class FakeYolo:
        task = 'detect'

        def __init__(self, path):
            pass

        def predict(self, image, **kwargs):
            box = SimpleNamespace(xyxy=np.array([[290, 210, 350, 270]]),
                                  conf=np.array([0.9]), cls=np.array([0]))
            return [SimpleNamespace(names={0: 'red'}, boxes=[box])]

    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=FakeYolo))
    rclpy.init(args=['--ros-args', '-p', 'dry_run:=true', '-p', 'settle_time:=0.0',
                    '-p', 'stable_frames:=2', '-p', f'calibration_file:={calib_path}',
                    '-p', f'backend:={backend}', '-p', f'yolo_model:={weights}'])
    nodes = []
    executor = MultiThreadedExecutor(num_threads=4)
    thread = None
    try:
        arm, detector, sorter, observer = ArmNode(), DetectorNode(), SorterNode(), Node('test_camera')
        nodes.extend([arm, detector, sorter, observer])
        publisher = observer.create_publisher(Image, 'camera/image_raw', 1)
        overlays = []
        observer.create_subscription(Image, 'detections/image', overlays.append, 10)
        for node in nodes:
            executor.add_node(node)
        thread = threading.Thread(target=executor.spin)
        thread.start()
        sorter.home()
        frame = np.full((480, 640, 3), 180, np.uint8)
        cv2.rectangle(frame, (290, 210), (350, 270), (0, 0, 220), -1)
        msg = CvBridge().cv2_to_imgmsg(frame, 'bgr8')
        msg.header.frame_id = 'camera'
        deadline = time.monotonic() + 5
        target = None
        while time.monotonic() < deadline:
            msg.header.stamp = observer.get_clock().now().to_msg()
            publisher.publish(msg)
            time.sleep(0.06)
            target = sorter.stable_target()
            if target is not None and overlays:
                break
        assert target is not None
        assert target.label == 'red'
        assert (target.x, target.y) == pytest.approx((0.25, 0.0), abs=0.001)
        assert overlays and overlays[-1].header.frame_id == 'camera'
        assert overlays[-1].header.stamp.sec > 0
        sorter.pick_and_place(target)
        sorter.home()
        moves = [c for c in arm.arm.sent if c['T'] == 104]
        assert any((c['x'], c['y'], c['z']) == pytest.approx((250, 0, -90)) for c in moves)
        assert any((c['x'], c['y'], c['z']) == pytest.approx((50, 250, -45)) for c in moves)
        assert sorter.stable_target() is None
        # A changed camera resolution must not yield a target with usable coordinates.
        msg = CvBridge().cv2_to_imgmsg(np.zeros((240, 320, 3), np.uint8), 'bgr8')
        msg.header.stamp = observer.get_clock().now().to_msg()
        publisher.publish(msg)
        time.sleep(0.1)
        assert sorter.stable_target() is None
    finally:
        executor.shutdown()
        if thread is not None:
            thread.join(timeout=5)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.try_shutdown()


def test_sorter_discards_pre_home_old_and_duplicate_camera_frames():
    rclpy.init(args=['--ros-args', '-p', 'stable_frames:=2'])
    sorter = SorterNode()
    try:
        msg = DetectionArray()
        msg.detections = [Detection(label='red', confidence=0.9, x=0.25, y=0.0)]
        msg.header.stamp = sorter.get_clock().now().to_msg()
        sorter.on_detections(msg)
        sorter.on_detections(msg)
        assert sorter.stable_target() is None
        msg.header.stamp = sorter.get_clock().now().to_msg()
        sorter.on_detections(msg)
        assert sorter.stable_target() is not None
        msg.header.stamp = (sorter.get_clock().now() - Duration(seconds=2)).to_msg()
        sorter.on_detections(msg)
        assert sorter.stable_target() is None
        before_home = sorter.get_clock().now().to_msg()
        sorter.ignore_detections_for(0)
        msg.header.stamp = before_home
        sorter.on_detections(msg)
        assert not sorter.targets.frames
    finally:
        sorter.destroy_node()
        rclpy.try_shutdown()
