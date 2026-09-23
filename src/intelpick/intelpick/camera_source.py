"""One way to open a camera, shared by camera_node and the calibrate tool.

device can be:
  /dev/video0 or 0                       USB webcam forwarded into WSL with usbipd
  http://192.168.1.23:4747/video         phone app (DroidCam, IP Webcam) or IP camera, MJPEG
  rtsp://user:pass@192.168.1.40/stream1  Wi-Fi security camera

Network streams are read on a background thread that keeps only the newest frame. Without it
OpenCV queues frames and the picture falls seconds behind the arm. The thread also reconnects
after Wi-Fi drops.
"""

import os
import re
import threading
import time

import cv2


def redact(url):
    """Hide the password in rtsp://user:pass@host URLs before logging them."""
    return re.sub(r'://([^:/@]+):[^@/]+@', r'://\1:***@', url)


class FrameSource:

    def __init__(self, device, width=640, height=480, stale_after=1.0, log=print):
        self.device = str(device).strip()
        self.is_stream = '://' in self.device
        self.width, self.height = width, height
        self.stale_after = stale_after  # network: a frame older than this is not handed out
        self.log = log
        self.name = redact(self.device)

        self._lock = threading.Lock()
        self._frame, self._stamp, self._seq, self._last_seq = None, 0.0, 0, 0
        self._running = True

        if self.is_stream:
            self._cap = None
            self._thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._thread.start()
        else:
            self._cap = self._open_local()

    # --- opening ----------------------------------------------------------------------------

    def _open_local(self):
        dev = int(self.device) if self.device.isdigit() else self.device
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f'cannot open camera {self.name} (usbipd attached? in video group?)')
        # MJPG keeps USB bandwidth low, which matters over usbipd.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _open_stream(self):
        if self.device.startswith('rtsp://'):
            # TCP: no smeared frames from lost UDP packets on Wi-Fi.
            os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
        cap = cv2.VideoCapture(self.device, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        return cap if cap.isOpened() else None

    def _stream_loop(self):
        backoff = 1.0
        while self._running:
            cap = self._open_stream()
            if cap is None:
                self.log(f'camera {self.name}: cannot connect, retrying in {backoff:.0f}s '
                         '(same Wi-Fi? app running? URL right?)')
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            self.log(f'camera {self.name}: connected')
            backoff = 1.0
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.log(f'camera {self.name}: stream lost, reconnecting')
                    break
                with self._lock:
                    self._frame, self._stamp = frame, time.monotonic()
                    self._seq += 1
            cap.release()

    # --- reading ----------------------------------------------------------------------------

    def read(self):
        """Newest frame not handed out before, or None."""
        if not self.is_stream:
            ok, frame = self._cap.read()
            return frame if ok else None
        with self._lock:
            if self._seq == self._last_seq or time.monotonic() - self._stamp > self.stale_after:
                return None
            self._last_seq = self._seq
            return self._frame

    def close(self):
        self._running = False
        if self.is_stream:
            self._thread.join(timeout=6.0)
        elif self._cap is not None:
            self._cap.release()
