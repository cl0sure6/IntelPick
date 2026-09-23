"""Minimal client for Waveshare RoArm-M2-S / RoArm-M3 using the ESP32 JSON protocol.

No ROS in here, so the calibration tool and unit tests can use it directly.

Protocol (Waveshare wiki "RoArm-M3-S Robotic Arm Control", "RoArm-M2-S JSON Command Meaning"):
  {"T":100}                              move all joints to the init pose (blocking)
  {"T":104,"x":..,"y":..,"z":..,...}     interpolated move of the tool tip, millimetres (blocking)
  {"T":105}                              request feedback -> JSON with x, y, z (mm) and joint angles
  {"T":106,"cmd":rad,"spd":0,"acc":0}    gripper angle; smaller = more open (1.08 .. 3.14)
  {"T":107,"tor":200}                    clamp torque limit, 200 = 20 % .. 1000 = 100 %
  {"T":210,"cmd":0|1}                    torque off/on (off lets you move the arm by hand)
Frame: origin at the base, +X forward, +Y left, +Z up. XYZ is the clamp tip.

T:104 differs by model: on the 4-DOF M2-S "t" is the gripper angle, on the M3 "t" is wrist
pitch, "r" is wrist roll and "g" is the gripper angle. Feedback (T:1051) reports the measured
gripper angle under the same key.
"""

import json
import math
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

GRIPPER_MIN = 1.08
GRIPPER_MAX = 3.14


@dataclass
class Pose:
    x: float  # mm
    y: float
    z: float


class RoArm:

    def __init__(self, port='/dev/ttyUSB0', model='m2', baud=115200, host='',
                 dry_run=False, m3_pitch=1.57):
        """
        port:     serial device, used when host is empty.
        host:     arm IP (e.g. 192.168.4.1 on the arm's own Wi-Fi AP) to use HTTP instead of serial.
        m3_pitch: wrist pitch (rad) sent with every M3 move; pick a value that points the gripper
                  down. Verify on hardware, it is not documented.
        """
        if model not in ('m2', 'm3'):
            raise ValueError(f'model must be m2 or m3, got {model!r}')
        self.model = model
        self.port = port
        self.baud = baud
        self.host = host
        self.dry_run = dry_run
        self.m3_pitch = m3_pitch
        self.gripper_angle = GRIPPER_MAX
        self.sent = []  # every command sent, for tests and debugging

        self._serial = None
        self._reader = None
        self._running = False
        self._feedback = None
        self._feedback_event = threading.Event()
        self._write_lock = threading.Lock()
        self._sim_pose = Pose(235.0, 0.0, 234.0)  # dry-run pose, matches the documented init pose
        self.sim_jaw_stop = 2.6  # dry-run: jaws stall here as if always closing on an object

    # --- connection -------------------------------------------------------------------------

    def connect(self):
        if self.dry_run or self.host:
            return
        import serial  # imported lazily so dry-run and tests do not need pyserial
        self._serial = serial.Serial()
        self._serial.port = self.port
        self._serial.baudrate = self.baud
        self._serial.timeout = 0.1
        self._serial.rts = False  # the ESP32 board resets when RTS/DTR toggle on open
        self._serial.dtr = False
        self._serial.open()
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def close(self):
        self._running = False
        if self._reader:
            self._reader.join(timeout=1.0)
        if self._serial:
            self._serial.close()

    def _read_loop(self):
        buf = b''
        while self._running:
            buf += self._serial.read(256)
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                self._handle_line(line.decode(errors='ignore').strip())

    def _handle_line(self, line):
        if not line.startswith('{'):
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        if all(k in msg for k in ('x', 'y', 'z')):
            self._feedback = msg
            self._feedback_event.set()

    # --- raw commands -----------------------------------------------------------------------

    def send(self, cmd):
        self.sent.append(cmd)
        payload = json.dumps(cmd, separators=(',', ':'))
        if self.dry_run:
            self._simulate(cmd)
            return
        if self.host:
            url = f'http://{self.host}/js?json=' + urllib.parse.quote(payload)
            with urllib.request.urlopen(url, timeout=10) as resp:
                self._handle_line(resp.read().decode(errors='ignore').strip())
            return
        with self._write_lock:
            self._serial.write(payload.encode() + b'\n')

    @property
    def _gripper_key(self):
        return 't' if self.model == 'm2' else 'g'

    def _simulate(self, cmd):
        t = cmd.get('T')
        if t == 100:
            self._sim_pose = Pose(235.0, 0.0, 234.0)
        elif t in (104, 1041):
            self._sim_pose = Pose(cmd['x'], cmd['y'], cmd['z'])
        elif t == 105:
            p = self._sim_pose
            self._handle_line(json.dumps(
                {'T': 1051, 'x': p.x, 'y': p.y, 'z': p.z,
                 self._gripper_key: min(self.gripper_angle, self.sim_jaw_stop)}))

    # --- high level -------------------------------------------------------------------------

    def get_feedback(self, timeout=1.0):
        """Raw T:1051 dict (x, y, z, joint angles, loads, voltage) or None."""
        self._feedback_event.clear()
        self.send({'T': 105})
        if not self._feedback_event.wait(timeout):
            return None
        return self._feedback

    def get_pose(self, timeout=1.0):
        fb = self.get_feedback(timeout)
        return None if fb is None else Pose(float(fb['x']), float(fb['y']), float(fb['z']))

    def home(self, timeout=10.0):
        """Go to the firmware init pose and wait until the arm stops moving."""
        self.send({'T': 100})
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            time.sleep(0.3)
            p = self.get_pose(timeout=1.0)
            if p and last and math.dist((p.x, p.y, p.z), (last.x, last.y, last.z)) < 1.0:
                return True
            last = p
        return False

    def xyz_cmd(self, x, y, z, spd=0.25):
        """T:104 for this model, keeping the gripper where it is."""
        cmd = {'T': 104, 'x': round(x, 1), 'y': round(y, 1), 'z': round(z, 1)}
        if self.model == 'm2':
            cmd['t'] = self.gripper_angle
        else:
            cmd.update(t=self.m3_pitch, r=0, g=self.gripper_angle)
        cmd['spd'] = spd
        return cmd

    def move_to(self, x, y, z, spd=0.25, tol=8.0, timeout=10.0):
        """Move the tool tip to (x, y, z) mm and wait until feedback says it arrived."""
        self.send(self.xyz_cmd(x, y, z, spd))
        return self._wait_until_near(Pose(x, y, z), tol, timeout)

    def set_gripper(self, angle, settle=0.6):
        """Command the gripper and return the angle it actually reached (None if unknown).

        When closing on an object the jaws stall early, so the measured angle stays below the
        commanded one: that is how a caller can tell whether something was grabbed.
        """
        self.gripper_angle = min(max(angle, GRIPPER_MIN), GRIPPER_MAX)
        self.send({'T': 106, 'cmd': self.gripper_angle, 'spd': 0, 'acc': 0})
        if not self.dry_run:
            time.sleep(settle)  # the gripper has no completion feedback
        fb = self.get_feedback()
        if fb is None or self._gripper_key not in fb:
            return None
        return float(fb[self._gripper_key])

    def set_grip_torque(self, percent):
        """Limit the squeeze so the servo doesn't overheat or crush a printed part."""
        self.send({'T': 107, 'tor': int(min(max(percent, 1), 100) * 10)})

    def set_torque(self, on):
        self.send({'T': 210, 'cmd': 1 if on else 0})

    def _wait_until_near(self, target, tol, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            p = self.get_pose(timeout=1.0)
            if p and math.dist((p.x, p.y, p.z), (target.x, target.y, target.z)) <= tol:
                return True
            time.sleep(0.1)
        return False
