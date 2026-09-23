"""FrameSource against a local MJPEG server that behaves like a phone camera app."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from intelpick.calibration import TableCalibration
from intelpick.camera_source import FrameSource, redact


class MjpegServer:
    """Streams uniform grey frames whose brightness is the frame counter, at ~30 fps."""

    def __init__(self, port=0):
        self.count = 0
        self.running = True
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while outer.running:
                        outer.count += 1
                        img = np.full((240, 320, 3), outer.count, np.uint8)
                        jpg = cv2.imencode('.jpg', img)[1].tobytes()
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                         b'Content-Length: %d\r\n\r\n' % len(jpg) + jpg + b'\r\n')
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}/video'

    def stop(self):
        self.running = False
        self.httpd.shutdown()
        self.httpd.server_close()


def wait_frame(src, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        f = src.read()
        if f is not None:
            return f
        time.sleep(0.02)
    return None


@pytest.fixture
def server():
    s = MjpegServer()
    yield s
    s.stop()


def test_stream_gives_newest_frame_without_backlog(server):
    src = FrameSource(server.url, log=lambda s: None)
    try:
        assert wait_frame(src) is not None
        time.sleep(1.0)  # a queueing reader would now be ~30 frames behind
        frame = wait_frame(src)
        assert frame.shape == (240, 320, 3)
        assert abs(float(frame.mean()) - server.count) <= 6
        assert src.read() is None  # the same frame is never handed out twice
    finally:
        src.close()


def test_stream_goes_stale_then_reconnects():
    s = MjpegServer()
    src = FrameSource(s.url, stale_after=0.5, log=lambda msg: None)
    try:
        assert wait_frame(src) is not None
        s.stop()
        time.sleep(1.0)
        assert src.read() is None  # old frames are not passed off as live
        s = MjpegServer(port=s.port)  # "Wi-Fi comes back"
        assert wait_frame(src, timeout=15) is not None
    finally:
        src.close()
        s.stop()


def test_redact_hides_password_only():
    assert redact('rtsp://admin:hunter2@10.0.0.5:554/stream1') == 'rtsp://admin:***@10.0.0.5:554/stream1'
    assert redact('http://192.168.1.23:4747/video') == 'http://192.168.1.23:4747/video'


def test_calibration_remembers_image_size(tmp_path):
    px = [(0, 0), (640, 0), (640, 480), (0, 480)]
    xy = [(0.4, 0.16), (0.4, -0.16), (0.16, -0.16), (0.16, 0.16)]
    TableCalibration.fit(px, xy, -0.04, image_size=(640, 480)).save(tmp_path / 'c.yaml')
    calib = TableCalibration.load(tmp_path / 'c.yaml')
    assert calib.matches(640, 480)
    assert not calib.matches(1280, 720)
    assert TableCalibration(np.eye(3), 0.0).matches(1280, 720)  # old files without a size
