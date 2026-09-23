"""Camera -> arm calibration by touching points on the table with the gripper tip.

Run with the arm_node stopped (this tool opens the serial port itself):
    ros2 run intelpick calibrate --port /dev/ttyUSB0 --model m2 --out ~/.intelpick/calibration.yaml
    ros2 run intelpick calibrate --camera http://192.168.1.23:4747/video     # Wi-Fi camera

Put 4-9 small markers (paper dots) spread over the pick area, then for each one:
  1. click the marker in the image window (arm out of the way),
  2. move the limp arm by hand until the gripper tip touches the marker,
  3. press SPACE to record the arm position.
Keys: SPACE record, U undo, ENTER fit and save, Q/ESC quit without saving.
"""

import argparse
import os

import cv2
import numpy as np

from .calibration import TableCalibration
from .camera_source import FrameSource
from .roarm import RoArm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--camera', default='/dev/video0', help='device, index or stream URL')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--host', default='', help='arm IP, to use Wi-Fi/HTTP instead of serial')
    ap.add_argument('--model', default='m2', choices=['m2', 'm3'])
    ap.add_argument('--out', default=os.path.expanduser('~/.intelpick/calibration.yaml'))
    args, _ = ap.parse_known_args()  # tolerate --ros-args from ros2 run

    try:
        cam = FrameSource(args.camera, args.width, args.height)
    except RuntimeError as e:
        raise SystemExit(str(e))

    arm = RoArm(port=args.port, host=args.host, model=args.model)
    arm.connect()
    arm.set_torque(False)
    print('Torque OFF: support the arm, it is now limp.')

    pixels, points = [], []  # recorded pairs: (u, v) and arm (x, y, z) in mm
    pending = None
    latest = None  # newest camera frame; streams don't deliver one every loop

    def on_click(event, u, v, *_):
        nonlocal pending
        if event == cv2.EVENT_LBUTTONDOWN:
            pending = (u, v)

    win = 'intelpick calibrate'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    saved = False
    try:
        while True:
            new = cam.read()
            if new is not None:
                latest = new
            if latest is None:
                if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                    break
                continue
            frame = latest.copy()
            size = (frame.shape[1], frame.shape[0])
            for i, (u, v) in enumerate(pixels):
                cv2.circle(frame, (u, v), 6, (0, 255, 0), 2)
                cv2.putText(frame, str(i + 1), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            if pending:
                cv2.drawMarker(frame, pending, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            status = f'{len(pixels)} points | click marker, touch it with tip, SPACE'
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(win, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                if pending is None:
                    print('click the marker in the image first')
                    continue
                p = arm.get_pose()
                if p is None:
                    print('no feedback from arm, check the connection')
                    continue
                pixels.append(pending)
                points.append((p.x, p.y, p.z))
                print(f'#{len(pixels)}: pixel {pending} -> arm ({p.x:.1f}, {p.y:.1f}, {p.z:.1f}) mm')
                pending = None
            elif key in (ord('u'), ord('U')) and pixels:
                pixels.pop()
                points.pop()
            elif key == 13:  # Enter
                if len(pixels) < 4:
                    print('need at least 4 points')
                    continue
                pts = np.array(points) / 1000.0
                calib = TableCalibration.fit(pixels, pts[:, :2], table_z=float(pts[:, 2].mean()),
                                             image_size=size)
                os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
                calib.save(args.out)
                print(f'saved {args.out}: rms {calib.rms_error * 1000:.1f} mm, '
                      f'table_z {calib.table_z * 1000:.1f} mm, image {size[0]}x{size[1]}')
                if calib.rms_error > 0.005:
                    print('rms above 5 mm: re-check markers or camera focus before trusting it')
                saved = True
                break
            elif key in (ord('q'), 27):
                break
    finally:
        arm.set_torque(True)
        arm.close()
        cam.close()
        cv2.destroyAllWindows()
    if not saved:
        print('not saved')
