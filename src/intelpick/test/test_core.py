"""Hardware-free tests: colour detection, calibration, grasp paths, RoArm command encoding."""

import cv2
import numpy as np
import pytest

from intelpick.calibration import TableCalibration
from intelpick.detectors.color import DEFAULT_COLORS, ColorDetector
from intelpick.grasp import min_pick_radius, pick_waypoints
from intelpick.roarm import RoArm


def test_color_detector_finds_squares():
    img = np.full((480, 640, 3), 200, np.uint8)          # light grey table
    cv2.rectangle(img, (100, 100), (160, 160), (0, 0, 220), -1)   # red (BGR)
    cv2.rectangle(img, (400, 300), (450, 350), (220, 60, 0), -1)  # blue
    cv2.circle(img, (300, 400), 30, (0, 220, 220), -1)           # yellow cap

    blobs = {b.label: b for b in ColorDetector(DEFAULT_COLORS).detect(img)}

    assert set(blobs) == {'red', 'blue', 'yellow'}
    assert blobs['red'].u == pytest.approx(130, abs=2)
    assert blobs['red'].v == pytest.approx(130, abs=2)
    assert blobs['blue'].u == pytest.approx(425, abs=2)
    assert blobs['red'].confidence > blobs['yellow'].confidence  # square fills its box better


def test_color_detector_ignores_small_noise():
    img = np.full((480, 640, 3), 200, np.uint8)
    cv2.rectangle(img, (10, 10), (14, 14), (0, 0, 220), -1)
    assert ColorDetector(DEFAULT_COLORS).detect(img) == []


def test_calibration_roundtrip(tmp_path):
    # Camera looking straight down: 1 px = 0.5 mm, image centre over (0.25, 0) m.
    pixels = [(100, 100), (540, 100), (540, 380), (100, 380), (320, 240)]
    table = [(0.25 - (v - 240) * 0.0005, -(u - 320) * 0.0005) for u, v in pixels]
    calib = TableCalibration.fit(pixels, table, table_z=-0.05)
    assert calib.rms_error < 1e-6

    path = tmp_path / 'calib.yaml'
    calib.save(path)
    loaded = TableCalibration.load(path)
    x, y = loaded.pixel_to_table(320, 140)
    assert (x, y) == pytest.approx((0.30, 0.0), abs=1e-6)
    assert loaded.table_z == -0.05


def test_calibration_needs_four_points():
    with pytest.raises(ValueError):
        TableCalibration.fit([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)], 0.0)


@pytest.mark.parametrize('model,extra', [('m2', {'t'}), ('m3', {'t', 'r', 'g'})])
def test_roarm_move_command_per_model(model, extra):
    arm = RoArm(model=model, dry_run=True)
    arm.set_gripper(2.0)
    assert arm.move_to(200, 50, 30)
    cmd = [c for c in arm.sent if c['T'] == 104][0]
    assert set(cmd) == {'T', 'x', 'y', 'z', 'spd'} | extra
    assert (cmd['x'], cmd['y'], cmd['z']) == (200, 50, 30)
    gripper_key = 't' if model == 'm2' else 'g'
    assert cmd[gripper_key] == 2.0


def test_roarm_gripper_is_clamped():
    arm = RoArm(dry_run=True)
    arm.set_gripper(0.0)
    assert {'T': 106, 'cmd': 1.08, 'spd': 0, 'acc': 0} in arm.sent


def test_roarm_gripper_reports_measured_angle():
    arm = RoArm(dry_run=True)
    assert arm.set_gripper(1.6) == pytest.approx(1.6)
    assert arm.set_gripper(3.1) == pytest.approx(arm.sim_jaw_stop)  # stalled on a "held" object


def test_roarm_grip_torque_encoding():
    arm = RoArm(dry_run=True)
    arm.set_grip_torque(30)
    arm.set_grip_torque(250)
    assert arm.sent[-2:] == [{'T': 107, 'tor': 300}, {'T': 107, 'tor': 1000}]


def test_radial_approach_slides_outward_along_base_line():
    x, y = 0.2, 0.2
    wps = pick_waypoints(x, y, z_grasp=-0.03, z_hover=0.05, approach='radial', standoff=0.04)
    (sx, sy, sz, _), (dx, dy, dz, slow_down), (gx, gy, gz, slow_in) = wps
    r = np.hypot(x, y)
    assert np.hypot(sx, sy) == pytest.approx(r - 0.04)
    assert (sx, sy) == pytest.approx((dx, dy)) and sz == 0.05 and dz == -0.03
    assert (gx, gy, gz) == pytest.approx((x, y, -0.03))
    assert np.cross([sx, sy], [x, y]) == pytest.approx(0, abs=1e-9)  # same bearing
    assert slow_down and slow_in


def test_vertical_approach_and_offset():
    wps = pick_waypoints(0.3, 0.0, -0.03, 0.05, approach='vertical', offset=0.01)
    assert [w[:3] for w in wps] == [pytest.approx((0.31, 0, 0.05)), pytest.approx((0.31, 0, -0.03))]


def test_min_pick_radius_accounts_for_standoff():
    assert min_pick_radius(0.12, 'radial', 0.04) == pytest.approx(0.16)
    assert min_pick_radius(0.12, 'vertical', 0.04) == pytest.approx(0.12)
