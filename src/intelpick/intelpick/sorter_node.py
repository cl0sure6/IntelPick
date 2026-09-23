"""The task logic: wait for a stable detection, pick it, drop it in the bin for its label, repeat.

The arm hides part of the table while it moves, so detections are only trusted once the arm is
back home and `settle_time` has passed, and only if the same object shows up in `stable_frames`
consecutive frames.
"""

import math
import os
import threading
import time
from collections import deque

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from intelpick_interfaces.msg import DetectionArray
from intelpick_interfaces.srv import MoveTo, SetGripper

from .calibration import TableCalibration
from .grasp import min_pick_radius, pick_waypoints


class SorterNode(Node):

    def __init__(self):
        super().__init__('sorter')
        labels = self.declare_parameter('bin_labels', ['red', 'green', 'blue', 'yellow']).value
        flat = self.declare_parameter(  # x0, y0, x1, y1, ... metres
            'bin_positions', [0.05, 0.25, 0.15, 0.25, 0.05, -0.25, 0.15, -0.25]).value
        if len(flat) != 2 * len(labels):
            raise ValueError('bin_positions must hold one (x, y) pair per bin label')
        self.bins = {lbl: (flat[2 * i], flat[2 * i + 1]) for i, lbl in enumerate(labels)}

        calib_file = self.declare_parameter('calibration_file', '').value
        table_z = self.declare_parameter('table_z', -0.05).value
        if calib_file and os.path.exists(calib_file):
            table_z = TableCalibration.load(calib_file).table_z
        else:
            self.get_logger().warn(f'no calibration, using table_z={table_z}')
        # Heights are relative to the table surface.
        self.z_grasp = table_z + self.declare_parameter('grasp_height', 0.015).value
        self.z_place = table_z + self.declare_parameter('place_height', 0.06).value
        self.z_hover = table_z + self.declare_parameter('hover_height', 0.10).value
        self.min_reach = self.declare_parameter('min_reach', 0.12).value
        self.max_reach = self.declare_parameter('max_reach', 0.40).value
        self.stable_frames = self.declare_parameter('stable_frames', 5).value
        self.stable_tol = self.declare_parameter('stable_tol', 0.01).value
        self.settle_time = self.declare_parameter('settle_time', 1.0).value
        self.max_picks = self.declare_parameter('max_picks', 0).value  # 0 = run forever
        self.approach = self.declare_parameter('approach', 'radial').value  # radial | vertical
        self.standoff = self.declare_parameter('standoff', 0.04).value
        self.grasp_offset = self.declare_parameter('grasp_offset', 0.0).value
        self.fast = self.declare_parameter('speed', 0.0).value  # 0 = arm_node default
        self.slow = self.declare_parameter('approach_speed', 0.1).value
        # After closing, a clamp angle within this margin of fully closed means nothing is held.
        self.check_grasp = self.declare_parameter('check_grasp', True).value
        self.empty_margin = self.declare_parameter('empty_grip_margin', 0.08).value
        self.gripper_closed = self.declare_parameter('gripper_closed', 3.1).value
        self.min_pick_r = min_pick_radius(self.min_reach, self.approach, self.standoff)
        for lbl, (bx, by) in self.bins.items():
            if not self.min_reach <= math.hypot(bx, by) <= self.max_reach:
                raise ValueError(f'bin {lbl!r} at ({bx}, {by}) is outside the reach limits')

        self.move_cli = self.create_client(MoveTo, 'arm/move_to')
        self.grip_cli = self.create_client(SetGripper, 'arm/set_gripper')
        self.home_cli = self.create_client(Trigger, 'arm/home')

        self._lock = threading.Lock()
        self._frames = deque(maxlen=self.stable_frames)
        self._accept_after = 0.0
        self.create_subscription(DetectionArray, 'detections', self.on_detections, 10)

    # --- perception -------------------------------------------------------------------------

    def on_detections(self, msg):
        if time.monotonic() < self._accept_after:
            return
        with self._lock:
            self._frames.append([d for d in msg.detections if self.pickable(d)])

    def pickable(self, d):
        if d.label not in self.bins or math.isnan(d.x):
            return False
        return self.min_pick_r <= math.hypot(d.x, d.y) <= self.max_reach

    def stable_target(self):
        """Closest object to the arm that appears in every one of the last N frames."""
        with self._lock:
            if len(self._frames) < self.stable_frames:
                return None
            frames = list(self._frames)
        for d in sorted(frames[-1], key=lambda d: math.hypot(d.x, d.y)):
            if all(any(o.label == d.label and
                       math.hypot(o.x - d.x, o.y - d.y) <= self.stable_tol for o in f)
                   for f in frames[:-1]):
                return d
        return None

    def ignore_detections_for(self, seconds):
        with self._lock:
            self._frames.clear()
        self._accept_after = time.monotonic() + seconds

    # --- motion -----------------------------------------------------------------------------

    def call(self, client, request, timeout=20.0):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'service {client.srv_name} not available')
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f'{client.srv_name} timed out')
            time.sleep(0.02)
        res = future.result()
        if not res.success:
            raise RuntimeError(f'{client.srv_name}: {res.message}')
        return res

    def move(self, x, y, z, slow=False):
        speed = self.slow if slow else self.fast
        self.call(self.move_cli, MoveTo.Request(x=float(x), y=float(y), z=float(z),
                                                speed=float(speed)))

    def grip(self, open_):
        return self.call(self.grip_cli, SetGripper.Request(open=open_)).angle

    def home(self):
        self.ignore_detections_for(1e9)
        self.call(self.home_cli, Trigger.Request())
        self.ignore_detections_for(self.settle_time)

    def pick_and_place(self, d):
        bx, by = self.bins[d.label]
        self.get_logger().info(
            f'{d.label} at ({d.x:.3f}, {d.y:.3f}) m -> bin ({bx:.3f}, {by:.3f})')
        self.ignore_detections_for(1e9)
        self.grip(True)
        waypoints = pick_waypoints(d.x, d.y, self.z_grasp, self.z_hover, self.approach,
                                   self.standoff, self.grasp_offset)
        for x, y, z, slow in waypoints:
            self.move(x, y, z, slow)
        angle = self.grip(False)
        gx, gy = waypoints[-1][:2]
        self.move(gx, gy, self.z_hover, slow=True)  # lift before bailing out, never drag
        if self.check_grasp and not math.isnan(angle) and \
                angle >= self.gripper_closed - self.empty_margin:
            raise RuntimeError(f'nothing in the gripper (clamp closed to {angle:.2f} rad)')
        self.move(bx, by, self.z_hover)
        self.move(bx, by, self.z_place)
        self.grip(True)
        self.move(bx, by, self.z_hover)

    def run(self):
        self.home()
        picks = 0
        while rclpy.ok() and (not self.max_picks or picks < self.max_picks):
            target = self.stable_target()
            if target is None:
                time.sleep(0.1)
                continue
            try:
                self.pick_and_place(target)
                picks += 1
            except RuntimeError as e:
                self.get_logger().error(f'pick failed: {e}')
            self.home()
        self.get_logger().info(f'done, {picks} objects sorted')


def main():
    rclpy.init()
    node = SorterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
