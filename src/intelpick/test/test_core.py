"""Hardware-free tests: colour detection, calibration, grasp paths, RoArm command encoding."""

import cv2
import numpy as np
import pytest

from intelpick.calibration import TableCalibration
from intelpick.detectors.color import DEFAULT_COLORS, ColorDetector
from intelpick.grasp import (StuckHolding, is_empty_grip, lift_point, min_pick_radius,
                             pick_steps, pick_waypoints, place_steps, run_pick_place)
from intelpick.roarm import Pose, RoArm


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


def _mat_calibration():
    # Markers on the corners of a 20 x 30 cm mat, 1 px = 1 mm, image centre over (0.25, 0).
    pixels = [(170, 140), (470, 140), (470, 340), (170, 340), (320, 240)]  # + centre marker
    table = [(0.25 - (v - 240) * 0.001, -(u - 320) * 0.001) for u, v in pixels]
    return TableCalibration.fit(pixels, table, table_z=-0.05, image_size=(640, 480)), pixels


def test_workspace_is_hull_of_calibration_points():
    calib, _ = _mat_calibration()
    assert len(calib.workspace) == 4  # centre marker is inside, not a corner
    assert calib.in_workspace(0.25, 0.0)
    assert calib.in_workspace(0.34, 0.14)             # near a corner, inside
    assert not calib.in_workspace(0.25, 0.20)         # 5 cm beyond the side
    assert calib.in_workspace(0.25, 0.155, margin=0.01)       # 5 mm out, within margin
    assert not calib.in_workspace(0.25, 0.145, margin=-0.01)  # shrunk zone


def test_workspace_outline_maps_back_to_marker_pixels(tmp_path):
    calib, pixels = _mat_calibration()
    calib.save(tmp_path / 'c.yaml')
    outline = TableCalibration.load(tmp_path / 'c.yaml').workspace_pixels()
    assert sorted(map(tuple, outline.tolist())) == sorted(pixels[:4])


def test_old_calibration_without_workspace_excludes_nothing():
    calib = TableCalibration(np.eye(3), 0.0)
    assert calib.in_workspace(5.0, 5.0)
    assert calib.workspace_pixels() is None


def test_command_echo_is_not_position_feedback():
    # Real firmware echoes each command; the T:104 echo holds the *target* coordinates.
    arm = RoArm(dry_run=True)
    arm._handle_line('{"T":104,"x":400,"y":0,"z":203,"t":1.6,"spd":0.15}')
    assert arm._feedback is None
    arm._handle_line('{"T":1051,"x":392.3,"y":-10.8,"z":177.0,"t":1.61}')
    assert arm._feedback['x'] == 392.3


def test_move_waits_for_stop_and_reports_miss():
    arm = RoArm(dry_run=True)
    simulate = arm._simulate

    def stops_short(cmd):  # arm parks 20 mm short of every target
        simulate(cmd)
        if cmd.get('T') == 104:
            arm._sim_pose = Pose(cmd['x'] - 20, cmd['y'], cmd['z'])

    arm._simulate = stops_short
    assert not arm.move_to(300, 0, 200, tol=15, timeout=3)
    assert arm.last_error == pytest.approx(20)
    assert arm.move_to(300, 0, 200, tol=25, timeout=3)


def test_pick_steps_open_approach_close_then_lift():
    steps = pick_steps(0.25, 0.0, z_grasp=-0.09, z_hover=-0.005, approach='radial', standoff=0.04)
    # the last close re-reads the grip after the lift (catches objects slipping out)
    assert [s.kind for s in steps] == ['open', 'move', 'move', 'move', 'close', 'move', 'close']
    slide, lift = steps[3], steps[5]
    assert (slide.x, slide.z) == pytest.approx((0.25, -0.09))
    assert (lift.x, lift.y, lift.z) == pytest.approx((0.25, 0.0, -0.005))  # straight up
    assert all(s.what for s in steps)


def test_place_steps_release_at_the_bottom():
    steps = place_steps(0.05, 0.25, z_place=-0.045, z_hover=-0.005)
    assert [s.kind for s in steps] == ['move', 'move', 'open', 'move']
    assert steps[1].z == pytest.approx(-0.045)


def test_empty_grip_threshold():
    # measured on the real M2-S: empty 3.094, holding a 2-3 cm object 2.988
    assert is_empty_grip(3.094, 3.1, 0.06)
    assert not is_empty_grip(2.988, 3.1, 0.06)
    assert not is_empty_grip(float('nan'), 3.1, 0.06)
    assert not is_empty_grip(None, 3.1, 0.06)


def test_grasp_trial_puts_the_object_back_where_it_was(monkeypatch):
    from intelpick import grasp_trial
    monkeypatch.setattr('sys.argv', ['grasp_trial', '--dry-run', '--auto', '--x', '0.25',
                                     '--table-z', '-0.105', '--grasp-height', '0.015'])
    monkeypatch.setattr('builtins.input', lambda prompt='': 'y')  # "yes, it's holding"
    trial = grasp_trial.Trial(grasp_trial.parse_args())
    trial.main()
    moves = [(c['x'], c['y'], c['z']) for c in trial.arm.sent if c['T'] == 104]
    assert all(np.hypot(x, y) > 150 for x, y, _ in moves)  # never towards the base
    at_grasp_spot = [m for m in moves if m == pytest.approx((250, 0, -90))]
    assert len(at_grasp_spot) == 2  # slide onto it, then set it back down there


def test_only_moves_at_grasp_height_are_strict():
    from intelpick.grasp import LOOSE_TOL
    for approach in ('radial', 'vertical'):
        for m in [s for s in pick_steps(0.33, 0.0, -0.09, -0.005, approach) if s.kind == 'move']:
            assert m.tol == (0.0 if m.z == -0.09 else LOOSE_TOL), (approach, m.what)
    assert all(s.tol == LOOSE_TOL for s in place_steps(0.05, 0.25, -0.045, -0.005)
               if s.kind == 'move')


def test_grasp_trial_accepts_sag_while_carrying(monkeypatch, capsys):
    # Real M2-S at 33 cm: the lift stopped 16 mm low under load (strict tolerance is 15).
    from intelpick import grasp_trial
    monkeypatch.setattr('sys.argv', ['grasp_trial', '--dry-run', '--auto', '--x', '0.33',
                                     '--table-z', '-0.105', '--tolerance', '0.015'])
    monkeypatch.setattr('builtins.input', lambda prompt='': 'y')
    trial = grasp_trial.Trial(grasp_trial.parse_args())
    simulate = trial.arm._simulate

    def sags_high_up(cmd):
        simulate(cmd)
        if cmd.get('T') == 104 and cmd['z'] > -50:
            trial.arm._sim_pose = Pose(cmd['x'], cmd['y'], cmd['z'] - 16)

    trial.arm._simulate = sags_high_up
    trial.main()
    assert '1/1 held' in capsys.readouterr().out


class FakeArm:
    """run_step stand-in: records steps, fails the ones named in `fail` as (what, x) pairs."""

    def __init__(self, fail=(), close_angle=2.7):
        self.fail, self.close_angle, self.done = set(fail), close_angle, []

    def run_step(self, step):
        if (step.what, round(step.x, 3)) in self.fail:
            raise RuntimeError(f'move "{step.what}" did not arrive')
        self.done.append((step.what, round(step.x, 3)))
        return self.close_angle if step.kind == 'close' else 1.6


def _cycle(arm):
    pick = pick_steps(0.25, 0.0, -0.09, -0.005)
    lift = lift_point(pick)
    run_pick_place(arm.run_step, pick, place=place_steps(0.05, 0.25, -0.045, -0.005),
                   put_back=place_steps(lift.x, lift.y, -0.09, -0.005),
                   grip_is_empty=lambda a: is_empty_grip(a, 3.1, 0.06), log=lambda m: None)


def test_pick_place_success_never_puts_back():
    arm = FakeArm()
    _cycle(arm)
    assert ('release', 0.0) in arm.done and ('over the drop point', 0.25) not in arm.done


def test_failure_while_holding_sets_object_back_down():
    arm = FakeArm(fail=[('over the drop point', 0.05)])  # bin unreachable
    with pytest.raises(RuntimeError) as e:
        _cycle(arm)
    assert not isinstance(e.value, StuckHolding)
    assert arm.done[-4:] == [('over the drop point', 0.25), ('down to release height', 0.25),
                             ('release', 0.0), ('back up', 0.25)]


def test_failure_before_grasp_or_empty_grip_does_not_put_back():
    for arm in (FakeArm(fail=[('slide out onto the object', 0.25)]), FakeArm(close_angle=3.09)):
        with pytest.raises(RuntimeError):
            _cycle(arm)
        assert ('over the drop point', 0.25) not in arm.done


def test_put_back_failing_too_is_stuck_holding():
    arm = FakeArm(fail=[('over the drop point', 0.05), ('down to release height', 0.25)])
    with pytest.raises(StuckHolding):
        _cycle(arm)
