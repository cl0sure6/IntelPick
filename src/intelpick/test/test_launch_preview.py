"""Launch the installed preview against a local stream; no arm process or device access."""

import os
import signal
import subprocess
import tempfile
import time

import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('intelpick_interfaces.msg')
from rclpy.node import Node

from intelpick_interfaces.msg import DetectionArray
from test_camera_source import MjpegServer


def test_vision_only_launch_uses_config_and_publishes_from_stream():
    server = MjpegServer()
    process = None
    node = None
    rclpy.init(args=[])
    try:
        node = Node('test_preview')
        received = []
        node.create_subscription(DetectionArray, 'detections', received.append, 10)
        with tempfile.TemporaryFile(mode='w+') as output:
            process = subprocess.Popen(
                ['ros2', 'launch', 'intelpick', 'intelpick.launch.py',
                 f'camera:={server.url}', 'sorter:=false', 'arm:=false'],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 10
            while not received and time.monotonic() < deadline and process.poll() is None:
                rclpy.spin_once(node, timeout_sec=0.1)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=10)
            output.seek(0)
            log = output.read()
        assert received, log
        assert received[-1].header.stamp.sec > 0
        assert 'detector backend: color' in log
        assert 'arm_node-' not in log and 'sorter_node-' not in log
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        server.stop()
