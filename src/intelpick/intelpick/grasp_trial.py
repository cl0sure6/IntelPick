"""Pick an object from a known spot with the sorter's exact sequence, check the grip, put it back.

Tunes the grasp on the real arm before the camera is involved (milestone M1):

    ros2 run intelpick grasp_trial --x 0.25 --y 0.0
    ros2 run intelpick grasp_trial --x 0.25 --grasp-height 0.01 --standoff 0.05 --open 1.3

Defaults come from config/intelpick.yaml, so settings that work go straight back there.
Stop arm_node first: this tool opens the serial port itself.
Each step waits for ENTER (--auto runs through). Ctrl-C stops; the arm holds its pose.
"""

import argparse
import math
import os

import yaml

from .grasp import Step, is_empty_grip, pick_steps, place_steps
from .roarm import RoArm


def load_config():
    """(shared, arm, sorter) parameter dicts from the installed config, empty if not found."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('intelpick'), 'config', 'intelpick.yaml')
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception:  # not built / not sourced: fall back to the built-in defaults
        data = {}
    return tuple((data.get(k) or {}).get('ros__parameters', {}) for k in ('/**', 'arm', 'sorter'))


def table_z_default(sorter):
    path = os.path.expanduser('~/.intelpick/calibration.yaml')
    if os.path.exists(path):
        with open(path) as f:
            return float(yaml.safe_load(f)['table_z'])
    return sorter.get('table_z', -0.105)


def parse_args():
    shared, arm, sorter = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add = ap.add_argument
    add('--x', type=float, default=0.25, help='object position, metres forward of the base')
    add('--y', type=float, default=0.0, help='object position, metres to the left (+) / right (-)')
    add('--tries', type=int, default=1)
    add('--auto', action='store_true', help="don't wait for ENTER before each step")
    add('--table-z', type=float, default=table_z_default(sorter),
        help='table height, m (default: calibration file, else config)')
    add('--grasp-height', type=float, default=sorter.get('grasp_height', 0.015))
    add('--hover-height', type=float, default=sorter.get('hover_height', 0.10))
    add('--approach', choices=['radial', 'vertical'], default=sorter.get('approach', 'radial'))
    add('--standoff', type=float, default=sorter.get('standoff', 0.04))
    add('--offset', type=float, default=sorter.get('grasp_offset', 0.0), help='grasp_offset, m')
    add('--open', type=float, default=arm.get('gripper_open', 1.6), help='gripper_open, rad')
    add('--closed', type=float, default=shared.get('gripper_closed', 3.1))
    add('--margin', type=float, default=sorter.get('empty_grip_margin', 0.06))
    add('--torque', type=int, default=arm.get('grip_torque', 30), help='grip_torque, %%')
    add('--speed', type=float, default=sorter.get('speed') or arm.get('speed', 0.25))
    add('--approach-speed', type=float, default=sorter.get('approach_speed', 0.1))
    add('--tolerance', type=float, default=arm.get('arrive_tolerance', 0.015), help='metres')
    add('--port', default='/dev/ttyUSB0')
    add('--host', default='', help='arm IP, to use Wi-Fi/HTTP instead of serial')
    add('--model', default='m2', choices=['m2', 'm3'])
    add('--dry-run', action='store_true', help='simulated arm, to rehearse the steps')
    args, _ = ap.parse_known_args()  # tolerate --ros-args from ros2 run
    return args


class Trial:

    def __init__(self, args):
        self.a = args
        self.arm = RoArm(port=args.port, host=args.host, model=args.model, dry_run=args.dry_run)

    def prompt(self, text):
        if self.a.auto:
            print(f'>>> {text}')
        else:
            input(f'>>> {text} [ENTER] ')

    def run(self, step):
        """Execute one step; returns the measured clamp angle for open/close."""
        self.prompt(step.what)
        if step.kind == 'move':
            spd = self.a.approach_speed if step.slow else self.a.speed
            ok = self.arm.move_to(step.x * 1000, step.y * 1000, step.z * 1000, spd=spd,
                                  tol=self.a.tolerance * 1000)
            p, miss = self.arm.last_pose, self.arm.last_error
            where = f'at ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) mm' if p else 'no feedback'
            print(f'    {"ok" if ok else "MISSED"}, {miss:.0f} mm off, {where}')
            if not ok:
                raise RuntimeError(f'move "{step.what}" did not arrive')
            return None
        angle = self.arm.set_gripper(self.a.open if step.kind == 'open' else self.a.closed)
        print(f'    clamp at {angle:.3f} rad' if angle is not None else '    clamp angle unknown')
        return angle

    def one_try(self, n):
        a = self.a
        z_grasp = a.table_z + a.grasp_height
        z_hover = a.table_z + a.hover_height
        pick = pick_steps(a.x, a.y, z_grasp, z_hover, a.approach, a.standoff, a.offset)
        side = 'left' if a.y >= 0 else 'right'
        input(f'\n--- try {n}/{a.tries}: put the object {a.x * 100:.0f} cm in front of the base, '
              f'{abs(a.y) * 100:.0f} cm {side}. Ready? [ENTER] ')

        angle = math.nan
        for step in pick:
            result = self.run(step)
            if step.kind == 'close':
                angle = result
        empty = is_empty_grip(angle, a.closed, a.margin)
        print(f'grip check: {"EMPTY" if empty else "HOLDING"} '
              f'(clamp {angle:.3f} rad, empty above {a.closed - a.margin:.3f})')
        held = input('>>> is it really holding the object? [y/n] ').strip().lower().startswith('y')

        # Put it back where it was picked up (the lift point), gently, at grasp height.
        lift = [s for s in pick if s.kind == 'move'][-1]
        if held:
            for step in place_steps(lift.x, lift.y, z_grasp, z_hover):
                self.run(step)
        else:
            self.run(Step('open', what='open the gripper (nothing to put back)'))
        self.prompt('home')
        self.arm.home()
        return held, not empty

    def main(self):
        a = self.a
        if a.grasp_height < 0.003:
            raise SystemExit('grasp height under 3 mm would drive the clamp into the table')
        reach = math.hypot(a.x, a.y)
        print(f'object {reach * 100:.0f} cm from the base; table z {a.table_z * 1000:.0f} mm; '
              f'{a.approach} approach, grasp {a.grasp_height * 1000:.0f} mm above the table, '
              f'standoff {a.standoff * 100:.0f} cm, gripper open {a.open:.2f} rad, '
              f'torque {a.torque} %')

        self.arm.connect()
        self.arm.set_grip_torque(a.torque)
        results = []
        try:
            self.prompt('home the arm')
            self.arm.home()
            for n in range(1, a.tries + 1):
                results.append(self.one_try(n))
        except (KeyboardInterrupt, EOFError):
            print('\naborted: the arm holds its current pose')
        except RuntimeError as e:
            print(f'\nstopped: {e}. The arm holds its current pose.')
        finally:
            self.arm.close()
        self.report(results)

    def report(self, results):
        if not results:
            return
        a = self.a
        held = sum(h for h, _ in results)
        agree = sum(h == c for h, c in results)
        print(f'\n=== {held}/{len(results)} held; grip check right {agree}/{len(results)} ===')
        if agree < len(results):
            print('grip check disagreed with you: re-measure empty vs held with probe_arm, '
                  'or the object is too thin for the check')
        print('settings used, for config/intelpick.yaml if they worked:')
        print(f'  sorter:  approach: {a.approach}  standoff: {a.standoff}  '
              f'grasp_offset: {a.offset}  grasp_height: {a.grasp_height}')
        print(f'  arm:     gripper_open: {a.open}  grip_torque: {a.torque}')


def main():
    Trial(parse_args()).main()
